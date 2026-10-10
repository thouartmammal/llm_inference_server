package main

import (
	"context"
	"crypto/rand"
	"crypto/tls"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	pb "github.com/thouartmammal/llm_inference_server/gateway"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
)

const defaultMaxOutputTokens = 512

type gateway struct {
	worker          pb.SchedulerWorkerClient
	tokenizer       tokenizer
	rpcTimeout      time.Duration
	maxOutputTokens int
	generationSlot  chan struct{}
}

type tokenizer interface {
	EncodePrompt(string) ([]int32, error)
	Decode([]int) (string, error)
}

type generateRequest struct {
	Prompt          string `json:"prompt"`
	MaxOutputTokens int    `json:"max_output_tokens,omitempty"`
}

type tokenEvent struct {
	RequestID string `json:"request_id"`
	TokenID   int32  `json:"token_id"`
	Text      string `json:"text"`
	Finished  bool   `json:"finished"`
}

type errorEvent struct {
	Error string `json:"error"`
}

func main() {
	if err := run(); err != nil {
		log.Print(err)
		os.Exit(1)
	}
}

func run() error {
	workerTarget := os.Getenv("WORKER_TARGET")
	if workerTarget == "" {
		return fmt.Errorf("WORKER_TARGET must be set to the worker tunnel host:port")
	}
	pythonExecutable := os.Getenv("PYTHON_EXECUTABLE")
	if pythonExecutable == "" {
		pythonExecutable = "python"
	}
	helperPath := os.Getenv("TOKENIZER_HELPER_PATH")
	if helperPath == "" {
		helperPath = "cmd/gateway/tokenizer_helper.py"
	}
	tokenizerImpl, err := startPythonTokenizer(
		pythonExecutable,
		helperPath,
		"Qwen/Qwen2.5-0.5B-Instruct",
	)
	if err != nil {
		return fmt.Errorf("start Hugging Face tokenizer helper: %w", err)
	}
	defer func() {
		if err := tokenizerImpl.Close(); err != nil {
			log.Printf("close tokenizer helper: %v", err)
		}
	}()

	rpcTimeout := 120 * time.Second
	if raw := os.Getenv("WORKER_RPC_TIMEOUT"); raw != "" {
		rpcTimeout, err = time.ParseDuration(raw)
		if err != nil || rpcTimeout <= 0 {
			return fmt.Errorf("WORKER_RPC_TIMEOUT must be a positive duration such as 120s")
		}
	}

	var transportCredentials credentials.TransportCredentials
	if strings.EqualFold(os.Getenv("WORKER_INSECURE"), "true") {
		transportCredentials = insecure.NewCredentials()
	} else {
		transportCredentials = credentials.NewTLS(&tls.Config{MinVersion: tls.VersionTLS12})
	}
	conn, err := grpc.NewClient(
		workerTarget,
		grpc.WithTransportCredentials(transportCredentials),
	)
	if err != nil {
		return fmt.Errorf("create worker gRPC client: %w", err)
	}
	defer conn.Close()

	app := &gateway{
		worker:          pb.NewSchedulerWorkerClient(conn),
		tokenizer:       tokenizerImpl,
		rpcTimeout:      rpcTimeout,
		maxOutputTokens: defaultMaxOutputTokens,
		generationSlot:  make(chan struct{}, 1),
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", app.health)
	mux.HandleFunc("/generate", app.generate)

	addr := os.Getenv("GATEWAY_ADDR")
	if addr == "" {
		addr = "127.0.0.1:8080"
	}
	server := &http.Server{
		Addr:              addr,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}
	log.Printf("gateway listening on http://%s", addr)
	if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return fmt.Errorf("serve HTTP gateway: %w", err)
	}
	return nil
}

func (g *gateway) health(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (g *gateway) generate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()
	var input generateRequest
	if err := decoder.Decode(&input); err != nil {
		http.Error(w, "request body must be JSON with a prompt string", http.StatusBadRequest)
		return
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		http.Error(w, "request body must contain exactly one JSON object", http.StatusBadRequest)
		return
	}
	if strings.TrimSpace(input.Prompt) == "" {
		http.Error(w, "prompt must not be empty", http.StatusBadRequest)
		return
	}
	if input.MaxOutputTokens == 0 {
		input.MaxOutputTokens = g.maxOutputTokens
	}
	if input.MaxOutputTokens < 1 || input.MaxOutputTokens > g.maxOutputTokens {
		http.Error(
			w,
			fmt.Sprintf("max_output_tokens must be between 1 and %d", g.maxOutputTokens),
			http.StatusBadRequest,
		)
		return
	}

	select {
	case g.generationSlot <- struct{}{}:
		defer func() { <-g.generationSlot }()
	case <-r.Context().Done():
		return
	}

	promptIDs, err := g.tokenizer.EncodePrompt(input.Prompt)
	if err != nil {
		http.Error(w, "tokenize prompt: "+err.Error(), http.StatusInternalServerError)
		return
	}
	if len(promptIDs) == 0 {
		http.Error(w, "tokenizer returned an empty prompt", http.StatusInternalServerError)
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming is not supported by this server", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("X-Prompt-Token-Count", strconv.Itoa(len(promptIDs)))
	w.WriteHeader(http.StatusOK)

	requestID, err := newRequestID()
	if err != nil {
		writeStreamError(w, flusher, fmt.Errorf("create request ID: %w", err))
		return
	}
	workerFinished := false
	defer func() {
		if workerFinished {
			return
		}
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if _, err := g.worker.CancelGeneration(ctx, &pb.CancelGenerationRequest{RequestId: requestID}); err != nil {
			log.Printf("cancel worker request %s: %v", requestID, err)
		}
	}()

	phase := pb.Phase_PREFILL
	tokenIDs := promptIDs
	generatedIDs := make([]int, 0, input.MaxOutputTokens)
	for generatedCount := 1; generatedCount <= input.MaxOutputTokens; generatedCount++ {
		ctx, cancel := context.WithTimeout(r.Context(), g.rpcTimeout)
		response, callErr := g.worker.ScheduleTask(ctx, &pb.ScheduleTaskBatch{
			Requests: []*pb.ScheduleTaskRequest{{
				RequestId: requestID,
				Phase:     phase,
				TokenIds:  tokenIDs,
			}},
		})
		cancel()
		if callErr != nil {
			if r.Context().Err() != nil {
				return
			}
			writeStreamError(w, flusher, fmt.Errorf("worker RPC failed: %w", callErr))
			return
		}
		if len(response.GetSampledTokens()) != 1 {
			writeStreamError(w, flusher, fmt.Errorf(
				"worker returned %d tokens; expected exactly one",
				len(response.GetSampledTokens()),
			))
			return
		}
		sampled := response.GetSampledTokens()[0]
		if sampled.GetRequestId() != requestID {
			writeStreamError(w, flusher, fmt.Errorf(
				"worker returned request ID %q; expected %q",
				sampled.GetRequestId(),
				requestID,
			))
			return
		}
		if sampled.GetFinished() {
			workerFinished = true
		}

		generatedIDs = append(generatedIDs, int(sampled.GetTokenId()))
		text, err := g.tokenizer.Decode(generatedIDs)
		if err != nil {
			writeStreamError(w, flusher, fmt.Errorf("decode generated token IDs: %w", err))
			return
		}
		event := tokenEvent{
			RequestID: requestID,
			TokenID:   sampled.GetTokenId(),
			Text:      text,
			Finished:  sampled.GetFinished() || generatedCount == input.MaxOutputTokens,
		}
		if err := writeEvent(w, flusher, event); err != nil {
			log.Printf("write generation stream for %s: %v", requestID, err)
			return
		}
		if event.Finished {
			return
		}

		phase = pb.Phase_DECODE
		tokenIDs = []int32{sampled.GetTokenId()}
	}

}

func writeEvent(w http.ResponseWriter, flusher http.Flusher, event tokenEvent) error {
	if err := json.NewEncoder(w).Encode(event); err != nil {
		return err
	}
	flusher.Flush()
	return nil
}

func writeStreamError(w http.ResponseWriter, flusher http.Flusher, err error) {
	log.Printf("generation failed: %v", err)
	if writeErr := json.NewEncoder(w).Encode(errorEvent{Error: err.Error()}); writeErr != nil {
		log.Printf("write error event: %v", writeErr)
		return
	}
	flusher.Flush()
}

func newRequestID() (string, error) {
	var bytes [16]byte
	if _, err := rand.Read(bytes[:]); err != nil {
		return "", err
	}
	return hex.EncodeToString(bytes[:]), nil
}
