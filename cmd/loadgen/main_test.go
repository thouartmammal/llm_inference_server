package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"
)

func TestParseOutputMix(t *testing.T) {
	got, err := parseOutputMix("16, 32,128")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 || got[0] != 16 || got[1] != 32 || got[2] != 128 {
		t.Fatalf("parsed mix = %v, want [16 32 128]", got)
	}
	if _, err := parseOutputMix("0,700"); err == nil {
		t.Fatal("expected invalid output lengths to return an error")
	}
}

func TestRunRequestRecordsStreamingMetrics(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("method = %s, want POST", r.Method)
		}
		w.Header().Set("X-Prompt-Token-Count", "27")
		w.Header().Set("Content-Type", "application/x-ndjson")
		fmt.Fprintln(w, `{"token_id":1,"text":"a","finished":false}`)
		fmt.Fprintln(w, `{"token_id":2,"text":"ab","finished":true}`)
	}))
	defer server.Close()

	client := &http.Client{Timeout: time.Second}
	result, intervals := runRequest(
		t.Context(),
		client,
		server.URL,
		1,
		promptProfile{Name: "short", Text: "test prompt"},
		2,
		false,
	)
	if !result.Success {
		t.Fatalf("request failed: %s", result.Error)
	}
	if result.PromptTokens != 27 || result.GeneratedTokens != 2 {
		t.Fatalf("token counts = prompt:%d generated:%d; want 27 and 2", result.PromptTokens, result.GeneratedTokens)
	}
	if result.TTFTMillis <= 0 || result.LatencyMillis < result.TTFTMillis {
		t.Fatalf("invalid TTFT/latency: %f/%f", result.TTFTMillis, result.LatencyMillis)
	}
	if len(intervals) != 1 || result.InterTokenMeanMs < 0 {
		t.Fatalf("inter-token metrics = %v, mean %f", intervals, result.InterTokenMeanMs)
	}
}

func TestRunRequestReportsUnfinishedStreamAsFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("X-Prompt-Token-Count", "3")
		fmt.Fprintln(w, `{"token_id":1,"text":"a","finished":false}`)
	}))
	defer server.Close()

	result, _ := runRequest(
		t.Context(),
		server.Client(),
		server.URL,
		1,
		promptProfile{Name: "short", Text: "x"},
		1,
		false,
	)
	if result.Success || result.Error != "stream ended without a finished token" {
		t.Fatalf("result = success:%t error:%q; want unfinished-stream failure", result.Success, result.Error)
	}
}

func TestIdleBaselineRunsPromptAndOutputCasesSequentially(t *testing.T) {
	var activeRequests atomic.Int32
	var maxActive atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		active := activeRequests.Add(1)
		for {
			maxSeen := maxActive.Load()
			if active <= maxSeen || maxActive.CompareAndSwap(maxSeen, active) {
				break
			}
		}
		defer activeRequests.Add(-1)

		time.Sleep(time.Millisecond)
		w.Header().Set("X-Prompt-Token-Count", "12")
		w.Header().Set("Content-Type", "application/x-ndjson")
		fmt.Fprintln(w, `{"token_id":1,"text":"a","finished":true}`)
	}))
	defer server.Close()

	outputDir := t.TempDir()
	prompts := []promptProfile{
		{Name: "short", Text: "short"},
		{Name: "medium", Text: "medium"},
		{Name: "long", Text: "long"},
	}
	runIdleBaseline(
		server.Client(),
		server.URL,
		"test-model",
		"test-gpu",
		outputDir,
		prompts,
		[]int{16, 32},
		0,
		2,
		7,
	)

	if got := maxActive.Load(); got != 1 {
		t.Fatalf("maximum concurrent requests = %d, want 1", got)
	}
	files, err := os.ReadDir(outputDir)
	if err != nil {
		t.Fatal(err)
	}
	var summaryPath, csvPath string
	for _, file := range files {
		switch filepath.Ext(file.Name()) {
		case ".json":
			summaryPath = filepath.Join(outputDir, file.Name())
		case ".csv":
			csvPath = filepath.Join(outputDir, file.Name())
		}
	}
	if summaryPath == "" || csvPath == "" {
		t.Fatalf("expected JSON and CSV output files; found %v", files)
	}

	summaryBytes, err := os.ReadFile(summaryPath)
	if err != nil {
		t.Fatal(err)
	}
	var result summary
	if err := json.Unmarshal(summaryBytes, &result); err != nil {
		t.Fatal(err)
	}
	if result.Mode != "idle" || result.RequestsCompleted != 12 || len(result.IdleCases) != 6 {
		t.Fatalf("unexpected idle summary: mode=%q completed=%d cases=%d", result.Mode, result.RequestsCompleted, len(result.IdleCases))
	}

	csvFile, err := os.Open(csvPath)
	if err != nil {
		t.Fatal(err)
	}
	defer csvFile.Close()
	records, err := csv.NewReader(csvFile).ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	if got := len(records) - 1; got != 12 {
		t.Fatalf("CSV sample rows = %d, want 12", got)
	}
}
