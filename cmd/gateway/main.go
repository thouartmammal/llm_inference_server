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
	"sync"
	"time"

	pb "github.com/thouartmammal/llm_inference_server/gateway"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
)

const defaultMaxOutputTokens = 512

type gateway struct {
	worker          pb.SchedulerWorkerClient
	workerTarget    string
	workerInsecure  bool
	tokenizer       tokenizer
	metrics         *gatewayTelemetry
	rpcTimeout      time.Duration
	maxOutputTokens int
	schedulerMode   string
	staticBatchSize int
	staticBatchWait time.Duration
	batchQueue      chan *generationRequest
	schedulerOnce   sync.Once
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

type generationRequest struct {
	id              string
	promptIDs       []int32
	maxOutputTokens int
	startedAt       time.Time
	lastTokenAt     time.Time
	tokensEmitted   int
	ctx             context.Context
	writer          http.ResponseWriter
	flusher         http.Flusher
	done            chan struct{}
	finishOnce      sync.Once
	generatedIDs    []int
	nextTokenID     int32
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

	workerInsecure := strings.EqualFold(os.Getenv("WORKER_INSECURE"), "true")
	var transportCredentials credentials.TransportCredentials
	if workerInsecure {
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

	batchSize := 4
	if raw := os.Getenv("STATIC_BATCH_SIZE"); raw != "" {
		batchSize, err = strconv.Atoi(raw)
		if err != nil || batchSize < 1 {
			return fmt.Errorf("STATIC_BATCH_SIZE must be a positive integer")
		}
	}
	batchWait := 50 * time.Millisecond
	if raw := os.Getenv("STATIC_BATCH_WAIT"); raw != "" {
		batchWait, err = time.ParseDuration(raw)
		if err != nil || batchWait < 0 {
			return fmt.Errorf("STATIC_BATCH_WAIT must be a non-negative duration such as 50ms")
		}
	}
	schedulerMode := strings.ToLower(strings.TrimSpace(os.Getenv("SCHEDULER_MODE")))
	if schedulerMode == "" {
		schedulerMode = "static"
	}
	switch schedulerMode {
	case "naive":
		batchSize = 1
		batchWait = 0
	case "static", "continuous":
	default:
		return fmt.Errorf("SCHEDULER_MODE must be naive, static, or continuous")
	}

	app := &gateway{
		worker:          pb.NewSchedulerWorkerClient(conn),
		workerTarget:    workerTarget,
		workerInsecure:  workerInsecure,
		tokenizer:       tokenizerImpl,
		metrics:         defaultGatewayTelemetry,
		rpcTimeout:      rpcTimeout,
		maxOutputTokens: defaultMaxOutputTokens,
		schedulerMode:   schedulerMode,
		staticBatchSize: batchSize,
		staticBatchWait: batchWait,
	}
	app.startBatchScheduler()
	log.Printf(
		"gateway config: worker_target=%s worker_insecure=%t scheduler_mode=%s static_batch_size=%d static_batch_wait=%s rpc_timeout=%s",
		app.workerTarget,
		app.workerInsecure,
		app.schedulerMode,
		app.staticBatchSize,
		app.staticBatchWait,
		app.rpcTimeout,
	)

	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", app.health)
	mux.Handle("/metrics", app.telemetry().handler())
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

func (g *gateway) telemetry() *gatewayTelemetry {
	return telemetryForGateway(g)
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
	requestID, err := newRequestID()
	if err != nil {
		http.Error(w, "create request ID: "+err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("X-Prompt-Token-Count", strconv.Itoa(len(promptIDs)))
	w.WriteHeader(http.StatusOK)
	g.startBatchScheduler()
	request := &generationRequest{
		id:              requestID,
		promptIDs:       promptIDs,
		maxOutputTokens: input.MaxOutputTokens,
		startedAt:       time.Now(),
		ctx:             r.Context(),
		writer:          w,
		flusher:         flusher,
		done:            make(chan struct{}),
		generatedIDs:    make([]int, 0, input.MaxOutputTokens),
	}
	log.Printf(
		"request accepted: request_id=%s prompt_tokens=%d max_output_tokens=%d scheduler_mode=%s",
		request.id,
		len(promptIDs),
		request.maxOutputTokens,
		g.schedulerMode,
	)
	g.telemetry().recordRequestAccepted()
	select {
	case g.batchQueue <- request:
		g.telemetry().recordQueueDepth(len(g.batchQueue))
	case <-r.Context().Done():
		g.finishRequest(request, "client_cancelled")
		return
	}
	<-request.done
}

func (g *gateway) startBatchScheduler() {
	g.schedulerOnce.Do(func() {
		batchSize := g.staticBatchSize
		if batchSize < 1 {
			batchSize = 1
		}
		batchWait := g.staticBatchWait
		if batchWait < 0 {
			batchWait = 50 * time.Millisecond
		}
		g.batchQueue = make(chan *generationRequest, batchSize*16)
		if g.schedulerMode == "continuous" {
			go g.continuousLoop(batchSize, batchWait)
			return
		}
		go g.batchLoop(batchSize, batchWait)
	})
}

func (g *gateway) batchLoop(batchSize int, batchWait time.Duration) {
	for {
		g.runStaticBatch(g.collectBatch(batchSize, batchWait, true))
	}
}

func (g *gateway) continuousLoop(batchSize int, batchWait time.Duration) {
	var active []*generationRequest
	for {
		if len(active) == 0 {
			active = g.runBatchPhase(
				g.collectBatch(batchSize, batchWait, true),
				pb.Phase_PREFILL,
			)
			continue
		}

		availableSlots := batchSize - len(active)
		if availableSlots > 0 {
			newcomers := g.collectBatch(availableSlots, batchWait, false)
			active = g.runContinuousStep(active, newcomers)
			continue
		}
		active = g.runBatchPhase(active, pb.Phase_DECODE)
	}
}

func (g *gateway) collectBatch(
	limit int,
	batchWait time.Duration,
	waitForFirst bool,
) []*generationRequest {
	defer g.telemetry().recordQueueDepth(len(g.batchQueue))
	if limit < 1 {
		return nil
	}
	var first *generationRequest
	if waitForFirst {
		first = <-g.batchQueue
	} else {
		select {
		case first = <-g.batchQueue:
		default:
			return nil
		}
	}
	batch := []*generationRequest{first}
	if limit == 1 || batchWait <= 0 {
		for len(batch) < limit {
			select {
			case request := <-g.batchQueue:
				batch = append(batch, request)
			default:
				return batch
			}
		}
		return batch
	}

	timer := time.NewTimer(batchWait)
	defer timer.Stop()
	for len(batch) < limit {
		select {
		case request := <-g.batchQueue:
			batch = append(batch, request)
		case <-timer.C:
			return batch
		}
	}
	return batch
}

func (g *gateway) runContinuousStep(
	active []*generationRequest,
	newcomers []*generationRequest,
) []*generationRequest {
	if len(newcomers) > 0 {
		active = append(active, g.runBatchPhase(newcomers, pb.Phase_PREFILL)...)
	}
	if len(active) == 0 {
		return nil
	}
	return g.runBatchPhase(active, pb.Phase_DECODE)
}

func (g *gateway) runStaticBatch(batch []*generationRequest) {
	active := g.runBatchPhase(batch, pb.Phase_PREFILL)
	for len(active) > 0 {
		active = g.runBatchPhase(active, pb.Phase_DECODE)
	}
}

func (g *gateway) runBatchPhase(
	requests []*generationRequest,
	phase pb.Phase,
) []*generationRequest {
	active := make([]*generationRequest, 0, len(requests))
	workerRequests := make([]*pb.ScheduleTaskRequest, 0, len(requests))
	for _, request := range requests {
		if request.ctx.Err() != nil {
			if phase == pb.Phase_DECODE {
				g.cancelWorkerRequest(request.id)
			}
			g.finishRequest(request, "client_cancelled")
			continue
		}
		tokens := request.promptIDs
		if phase == pb.Phase_DECODE {
			tokens = []int32{request.nextTokenID}
		}
		workerRequests = append(workerRequests, &pb.ScheduleTaskRequest{
			RequestId: request.id,
			Phase:     phase,
			TokenIds:  tokens,
		})
		active = append(active, request)
	}
	if len(active) == 0 {
		return nil
	}

	g.telemetry().recordBatchSize(phase, len(active))
	log.Printf(
		"dispatch worker batch: phase=%s batch_size=%d worker_target=%s",
		phaseLabel(phase),
		len(active),
		g.workerTarget,
	)
	ctx, cancel := context.WithTimeout(context.Background(), g.rpcTimeout)
	startedAt := time.Now()
	response, err := g.worker.ScheduleTask(ctx, &pb.ScheduleTaskBatch{Requests: workerRequests})
	cancel()
	g.telemetry().recordWorkerRPC(phase, time.Since(startedAt))
	if err != nil {
		log.Printf(
			"worker ScheduleTask failed: phase=%s batch_size=%d worker_target=%s err=%v",
			phaseLabel(phase),
			len(active),
			g.workerTarget,
			err,
		)
		g.failBatch(active, fmt.Errorf("worker RPC failed: %w", err))
		return nil
	}
	sampledByID, err := indexSampledTokens(response.GetSampledTokens(), active)
	if err != nil {
		g.failBatch(active, err)
		return nil
	}

	nextActive := make([]*generationRequest, 0, len(active))
	for _, request := range active {
		if request.ctx.Err() != nil {
			g.cancelWorkerRequest(request.id)
			g.finishRequest(request, "client_cancelled")
			continue
		}
		sampled := sampledByID[request.id]
		request.generatedIDs = append(request.generatedIDs, int(sampled.GetTokenId()))
		text, decodeErr := g.tokenizer.Decode(request.generatedIDs)
		if decodeErr != nil {
			g.writeRequestError(request, fmt.Errorf("decode generated token IDs: %w", decodeErr))
			g.cancelWorkerRequest(request.id)
			g.finishRequest(request, "decode_error")
			continue
		}
		finished := sampled.GetFinished() || len(request.generatedIDs) >= request.maxOutputTokens
		if writeErr := writeEvent(request.writer, request.flusher, tokenEvent{
			RequestID: request.id,
			TokenID:   sampled.GetTokenId(),
			Text:      text,
			Finished:  finished,
		}); writeErr != nil {
			log.Printf("write generation stream for %s: %v", request.id, writeErr)
			g.cancelWorkerRequest(request.id)
			g.finishRequest(request, "stream_error")
			continue
		}
		g.telemetry().recordTokenEmission(request)
		if finished {
			if !sampled.GetFinished() {
				g.cancelWorkerRequest(request.id)
			}
			if len(request.generatedIDs) >= request.maxOutputTokens && !sampled.GetFinished() {
				g.finishRequest(request, "output_limit")
				continue
			}
			g.finishRequest(request, "completed")
			continue
		}
		request.nextTokenID = sampled.GetTokenId()
		nextActive = append(nextActive, request)
	}
	return nextActive
}

func indexSampledTokens(
	sampled []*pb.SampledToken,
	active []*generationRequest,
) (map[string]*pb.SampledToken, error) {
	expected := make(map[string]struct{}, len(active))
	for _, request := range active {
		expected[request.id] = struct{}{}
	}
	indexed := make(map[string]*pb.SampledToken, len(sampled))
	for _, token := range sampled {
		if token == nil {
			return nil, fmt.Errorf("worker returned an empty sampled-token entry")
		}
		if _, ok := expected[token.GetRequestId()]; !ok {
			return nil, fmt.Errorf("worker returned unexpected request ID %q", token.GetRequestId())
		}
		if _, duplicate := indexed[token.GetRequestId()]; duplicate {
			return nil, fmt.Errorf("worker returned duplicate token for request %q", token.GetRequestId())
		}
		indexed[token.GetRequestId()] = token
	}
	if len(indexed) != len(expected) {
		return nil, fmt.Errorf("worker returned %d tokens; expected %d", len(indexed), len(expected))
	}
	return indexed, nil
}

func (g *gateway) failBatch(active []*generationRequest, err error) {
	for _, request := range active {
		if request.ctx.Err() == nil {
			g.writeRequestError(request, err)
		}
		g.cancelWorkerRequest(request.id)
		g.finishRequest(request, "worker_error")
	}
}

func (g *gateway) finishRequest(request *generationRequest, outcome string) {
	if request == nil {
		return
	}
	request.finishOnce.Do(func() {
		g.telemetry().recordRequestFinished(request, outcome)
		close(request.done)
	})
}

func (g *gateway) writeRequestError(request *generationRequest, err error) {
	writeStreamError(request.writer, request.flusher, err)
}

func (g *gateway) cancelWorkerRequest(requestID string) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := g.worker.CancelGeneration(ctx, &pb.CancelGenerationRequest{RequestId: requestID}); err != nil {
		log.Printf(
			"cancel worker request failed: request_id=%s worker_target=%s err=%v",
			requestID,
			g.workerTarget,
			err,
		)
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
