package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	pb "github.com/thouartmammal/llm_inference_server/gateway"
	"google.golang.org/grpc"
)

type fakeTokenizer struct {
	encoded string
}

func (f *fakeTokenizer) EncodePrompt(input string) ([]int32, error) {
	f.encoded = input
	return []int32{10, 11}, nil
}

func (*fakeTokenizer) Decode(ids []int) (string, error) {
	return strings.Repeat("x", len(ids)), nil
}

type fakeWorker struct {
	calls               []*pb.ScheduleTaskBatch
	cancelledIDs        []string
	finishAt            int
	waitForCancellation bool
	started             chan struct{}
}

func (f *fakeWorker) ScheduleTask(ctx context.Context, batch *pb.ScheduleTaskBatch, _ ...grpc.CallOption) (*pb.ScheduleTaskResponse, error) {
	f.calls = append(f.calls, batch)
	if f.waitForCancellation {
		close(f.started)
		<-ctx.Done()
		return nil, ctx.Err()
	}
	finished := f.finishAt != 0 && len(f.calls) >= f.finishAt
	return &pb.ScheduleTaskResponse{SampledTokens: []*pb.SampledToken{{
		RequestId: batch.Requests[0].GetRequestId(),
		TokenId:   int32(20 + len(f.calls)),
		Finished:  finished,
	}}}, nil
}

func (f *fakeWorker) CancelGeneration(_ context.Context, request *pb.CancelGenerationRequest, _ ...grpc.CallOption) (*pb.CancelGenerationResponse, error) {
	f.cancelledIDs = append(f.cancelledIDs, request.GetRequestId())
	return &pb.CancelGenerationResponse{CacheReleased: true}, nil
}

func TestGenerateStreamsTokensAndPassesPromptToTokenizer(t *testing.T) {
	worker := &fakeWorker{finishAt: 2}
	tok := &fakeTokenizer{}
	app := &gateway{
		worker:          worker,
		tokenizer:       tok,
		maxOutputTokens: 512,
		generationSlot:  make(chan struct{}, 1),
	}

	request := httptest.NewRequest(
		http.MethodPost,
		"/generate",
		strings.NewReader(`{"prompt":"What is a KV cache?"}`),
	)
	response := httptest.NewRecorder()
	app.generate(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d; body: %s", response.Code, http.StatusOK, response.Body)
	}
	if got, want := tok.encoded, "What is a KV cache?"; got != want {
		t.Fatalf("tokenizer input = %q, want %q", got, want)
	}
	if len(worker.calls) != 2 {
		t.Fatalf("worker calls = %d, want 2", len(worker.calls))
	}
	if got := worker.calls[0].Requests[0].GetPhase(); got != pb.Phase_PREFILL {
		t.Fatalf("first phase = %v, want PREFILL", got)
	}
	if got := worker.calls[1].Requests[0].GetPhase(); got != pb.Phase_DECODE {
		t.Fatalf("second phase = %v, want DECODE", got)
	}
	if got := worker.calls[1].Requests[0].GetTokenIds(); len(got) != 1 || got[0] != 21 {
		t.Fatalf("decode token IDs = %v, want [21]", got)
	}
	if len(worker.cancelledIDs) != 0 {
		t.Fatalf("completed generation triggered cleanup: %v", worker.cancelledIDs)
	}

	lines := strings.Split(strings.TrimSpace(response.Body.String()), "\n")
	if len(lines) != 2 {
		t.Fatalf("stream events = %d, want 2; body: %s", len(lines), response.Body)
	}
	var first, second tokenEvent
	if err := json.Unmarshal([]byte(lines[0]), &first); err != nil {
		t.Fatalf("decode first event: %v", err)
	}
	if err := json.Unmarshal([]byte(lines[1]), &second); err != nil {
		t.Fatalf("decode second event: %v", err)
	}
	if first.Finished || !second.Finished {
		t.Fatalf("finished values = [%t, %t], want [false, true]", first.Finished, second.Finished)
	}
	if first.RequestID == "" || first.RequestID != second.RequestID {
		t.Fatalf("request IDs = [%q, %q], want same non-empty ID", first.RequestID, second.RequestID)
	}
	if first.Text != "x" || second.Text != "xx" {
		t.Fatalf("streamed text snapshots = [%q, %q], want [x, xx]", first.Text, second.Text)
	}
}

func TestGenerateCancelsWorkerWhenOutputLimitIsReached(t *testing.T) {
	worker := &fakeWorker{}
	app := &gateway{
		worker:          worker,
		tokenizer:       &fakeTokenizer{},
		maxOutputTokens: 8,
		generationSlot:  make(chan struct{}, 1),
	}
	request := httptest.NewRequest(
		http.MethodPost,
		"/generate",
		strings.NewReader(`{"prompt":"test","max_output_tokens":1}`),
	)
	response := httptest.NewRecorder()

	app.generate(response, request)

	if len(worker.calls) != 1 {
		t.Fatalf("worker calls = %d, want 1", len(worker.calls))
	}
	if len(worker.cancelledIDs) != 1 || worker.cancelledIDs[0] != worker.calls[0].Requests[0].GetRequestId() {
		t.Fatalf("cancelled request IDs = %v, want generated worker request ID", worker.cancelledIDs)
	}
	var event tokenEvent
	if err := json.Unmarshal(bytes.TrimSpace(response.Body.Bytes()), &event); err != nil {
		t.Fatalf("decode stream event: %v", err)
	}
	if !event.Finished {
		t.Fatal("final event finished = false, want true at output limit")
	}
}

func TestGenerateCleansWorkerCacheWhenClientCancels(t *testing.T) {
	worker := &fakeWorker{
		waitForCancellation: true,
		started:             make(chan struct{}),
	}
	app := &gateway{
		worker:          worker,
		tokenizer:       &fakeTokenizer{},
		maxOutputTokens: 8,
		generationSlot:  make(chan struct{}, 1),
	}
	ctx, cancel := context.WithCancel(context.Background())
	request := httptest.NewRequest(
		http.MethodPost,
		"/generate",
		strings.NewReader(`{"prompt":"test"}`),
	).WithContext(ctx)
	response := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		defer close(done)
		app.generate(response, request)
	}()

	<-worker.started
	cancel()
	<-done

	if len(worker.cancelledIDs) != 1 || worker.cancelledIDs[0] != worker.calls[0].Requests[0].GetRequestId() {
		t.Fatalf("cancelled request IDs = %v, want active worker request ID", worker.cancelledIDs)
	}
}

func TestGenerateRejectsEmptyPrompt(t *testing.T) {
	app := &gateway{
		worker:         &fakeWorker{},
		tokenizer:      &fakeTokenizer{},
		generationSlot: make(chan struct{}, 1),
	}
	request := httptest.NewRequest(http.MethodPost, "/generate", strings.NewReader(`{"prompt":"  "}`))
	response := httptest.NewRecorder()

	app.generate(response, request)

	if response.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d", response.Code, http.StatusBadRequest)
	}
}
