package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

type promptProfile struct {
	Name string `json:"name"`
	Text string `json:"text"`
}

type generateRequest struct {
	Prompt          string `json:"prompt"`
	MaxOutputTokens int    `json:"max_output_tokens"`
}

type tokenEvent struct {
	TokenID  int32  `json:"token_id"`
	Text     string `json:"text"`
	Finished bool   `json:"finished"`
	Error    string `json:"error"`
}

type requestResult struct {
	Index            int       `json:"index"`
	PromptName       string    `json:"prompt_name"`
	PromptBytes      int       `json:"prompt_bytes"`
	PromptTokens     int       `json:"prompt_tokens"`
	RequestedTokens  int       `json:"requested_output_tokens"`
	GeneratedTokens  int       `json:"generated_tokens"`
	TTFTMillis       float64   `json:"ttft_ms"`
	InterTokenMeanMs float64   `json:"inter_token_mean_ms"`
	LatencyMillis    float64   `json:"latency_ms"`
	StartedAt        time.Time `json:"started_at"`
	CompletedAt      time.Time `json:"completed_at"`
	Success          bool      `json:"success"`
	Error            string    `json:"error,omitempty"`
	WasWarmup        bool      `json:"warmup"`
}

type summary struct {
	Mode                string            `json:"mode"`
	Model               string            `json:"model"`
	GPU                 string            `json:"gpu"`
	GatewayURL          string            `json:"gateway_url"`
	ArrivalRateRPS      float64           `json:"arrival_rate_rps"`
	MeasurementSeconds  float64           `json:"measurement_seconds"`
	ElapsedSeconds      float64           `json:"elapsed_seconds"`
	WarmupRequests      int               `json:"warmup_requests"`
	Seed                int64             `json:"seed"`
	PromptProfiles      []promptProfile   `json:"prompt_profiles"`
	OutputTokenMix      []int             `json:"output_token_mix"`
	RequestsOffered     int               `json:"requests_offered"`
	RequestsStarted     int               `json:"requests_started"`
	RequestsDropped     int               `json:"requests_dropped_at_inflight_limit"`
	RequestsCompleted   int               `json:"requests_completed"`
	RequestsFailed      int               `json:"requests_failed"`
	GeneratedTokens     int               `json:"generated_tokens"`
	SuccessfulReqPerSec float64           `json:"successful_requests_per_second"`
	GeneratedTokPerSec  float64           `json:"generated_tokens_per_second"`
	TTFT                metricSummary     `json:"ttft_ms"`
	InterTokenLatency   metricSummary     `json:"inter_token_latency_ms"`
	EndToEndLatency     metricSummary     `json:"end_to_end_latency_ms"`
	PromptTokenCounts   metricSummary     `json:"prompt_token_counts"`
	IdleCases           []idleCaseSummary `json:"idle_cases,omitempty"`
}

type idleCaseSummary struct {
	PromptName      string        `json:"prompt_name"`
	RequestedTokens int           `json:"requested_output_tokens"`
	Repetitions     int           `json:"repetitions"`
	Completed       int           `json:"completed"`
	Failed          int           `json:"failed"`
	TTFT            metricSummary `json:"ttft_ms"`
	InterToken      metricSummary `json:"inter_token_latency_ms"`
	EndToEnd        metricSummary `json:"end_to_end_latency_ms"`
}

type metricSummary struct {
	P50  float64 `json:"p50"`
	P95  float64 `json:"p95"`
	Mean float64 `json:"mean"`
}

type resultEnvelope struct {
	result      requestResult
	interTokens []float64
}

func main() {
	mode := flag.String("mode", "poisson", "Test mode: poisson or idle")
	gatewayURL := flag.String("url", "http://127.0.0.1:8080/generate", "Gateway generate endpoint")
	rate := flag.Float64("rate", 0.2, "Poisson request arrival rate per second")
	duration := flag.Duration("duration", 60*time.Second, "Measured arrival window")
	warmups := flag.Int("warmup", 1, "Sequential warm-up requests excluded from results")
	outputMixArg := flag.String("output-lengths", "32,64,128", "Comma-separated maximum output-token mix")
	maxInflight := flag.Int("max-inflight", 64, "Maximum outstanding HTTP requests")
	seed := flag.Int64("seed", 1, "Random seed for repeatable prompt/output selection")
	model := flag.String("model", "Qwen/Qwen2.5-0.5B-Instruct", "Model name recorded in results")
	gpu := flag.String("gpu", "unspecified", "GPU model recorded in results (for example, Kaggle T4)")
	outputDir := flag.String("output-dir", "benchmark-results", "Directory for JSON summary and CSV samples")
	repetitions := flag.Int("repetitions", 3, "Sequential samples for each prompt/output length in idle mode")
	flag.Parse()

	if *rate <= 0 || *duration <= 0 || *warmups < 0 || *maxInflight < 1 {
		fail("rate, duration and max-inflight must be positive; warmup cannot be negative")
	}
	if *mode != "poisson" && *mode != "idle" {
		fail("mode must be poisson or idle")
	}
	if *repetitions < 1 {
		fail("repetitions must be at least 1")
	}
	outputMix, err := parseOutputMix(*outputMixArg)
	if err != nil {
		fail("invalid -output-lengths: %v", err)
	}
	prompts := []promptProfile{
		{Name: "short", Text: "Explain KV cache in one sentence."},
		{Name: "medium", Text: "Explain how prefill and decode work in an LLM inference server, and why the KV cache makes decoding more efficient."},
		{Name: "long", Text: "Describe the request lifecycle in a GPU LLM inference server. Cover prompt tokenization, prefill, autoregressive decoding, the KV cache, streaming generated tokens to a client, end-of-sequence handling, and why scheduling multiple requests can improve GPU utilization. Give a concise but technically accurate explanation."},
	}

	client := &http.Client{Timeout: 30 * time.Minute}
	arrivalRNG := rand.New(rand.NewSource(*seed))
	profileRNG := rand.New(rand.NewSource(*seed ^ 0x5deece66d))

	if *mode == "idle" {
		runIdleBaseline(
			client,
			*gatewayURL,
			*model,
			*gpu,
			*outputDir,
			prompts,
			outputMix,
			*warmups,
			*repetitions,
			*seed,
		)
		return
	}

	for i := 0; i < *warmups; i++ {
		prompt := prompts[i%len(prompts)]
		outputTokens := outputMix[i%len(outputMix)]
		warmupResult, _ := runRequest(context.Background(), client, *gatewayURL, -(*warmups - i), prompt, outputTokens, true)
		if !warmupResult.Success {
			fail("warm-up request failed: %s", warmupResult.Error)
		}
		fmt.Printf("warm-up %d/%d complete (%d tokens)\n", i+1, *warmups, warmupResult.GeneratedTokens)
	}

	measureStart := time.Now()
	arrivalEnd := measureStart.Add(*duration)
	results := make(chan resultEnvelope, *maxInflight)
	slots := make(chan struct{}, *maxInflight)
	var workers sync.WaitGroup
	started := 0
	offered := 0
	arrival := measureStart
	requestIndex := 0
	dropped := 0

	for {
		intervalSeconds := arrivalRNG.ExpFloat64() / *rate
		if intervalSeconds > duration.Seconds() {
			break
		}
		interval := time.Duration(intervalSeconds * float64(time.Second))
		if interval < time.Nanosecond {
			interval = time.Nanosecond
		}
		arrival = arrival.Add(interval)
		if arrival.After(arrivalEnd) {
			break
		}
		if delay := time.Until(arrival); delay > 0 {
			time.Sleep(delay)
		}
		prompt := prompts[profileRNG.Intn(len(prompts))]
		outputTokens := outputMix[profileRNG.Intn(len(outputMix))]
		requestIndex++
		offered++
		select {
		case slots <- struct{}{}:
		default:
			dropped++
			continue
		}
		index := requestIndex
		started++
		workers.Add(1)
		go func() {
			defer workers.Done()
			defer func() { <-slots }()
			result, interTokens := runRequest(
				context.Background(),
				client,
				*gatewayURL,
				index,
				prompt,
				outputTokens,
				false,
			)
			results <- resultEnvelope{result: result, interTokens: interTokens}
		}()
	}
	workers.Wait()
	close(results)
	measureEnd := time.Now()

	resultRows := make([]requestResult, 0, started)
	var ttfts, interTokenLatencies, latencies, promptTokenCounts []float64
	completed, failed, generatedTokens := 0, 0, 0
	for item := range results {
		resultRows = append(resultRows, item.result)
		if item.result.Success {
			completed++
			generatedTokens += item.result.GeneratedTokens
			ttfts = append(ttfts, item.result.TTFTMillis)
			latencies = append(latencies, item.result.LatencyMillis)
			promptTokenCounts = append(promptTokenCounts, float64(item.result.PromptTokens))
			interTokenLatencies = append(interTokenLatencies, item.interTokens...)
		} else {
			failed++
		}
	}
	sort.Slice(resultRows, func(i, j int) bool { return resultRows[i].Index < resultRows[j].Index })

	elapsedSeconds := measureEnd.Sub(measureStart).Seconds()
	summaryResult := summary{
		Mode:               "poisson",
		Model:              *model,
		GPU:                *gpu,
		GatewayURL:         *gatewayURL,
		ArrivalRateRPS:     *rate,
		MeasurementSeconds: duration.Seconds(),
		ElapsedSeconds:     elapsedSeconds,
		WarmupRequests:     *warmups,
		Seed:               *seed,
		PromptProfiles:     prompts,
		OutputTokenMix:     outputMix,
		RequestsOffered:    offered,
		RequestsStarted:    started,
		RequestsDropped:    dropped,
		RequestsCompleted:  completed,
		RequestsFailed:     failed,
		GeneratedTokens:    generatedTokens,
		TTFT:               summarize(ttfts),
		InterTokenLatency:  summarize(interTokenLatencies),
		EndToEndLatency:    summarize(latencies),
		PromptTokenCounts:  summarize(promptTokenCounts),
	}
	if elapsedSeconds > 0 {
		summaryResult.SuccessfulReqPerSec = float64(completed) / elapsedSeconds
		summaryResult.GeneratedTokPerSec = float64(generatedTokens) / elapsedSeconds
	}
	if err := writeResults(*outputDir, summaryResult, resultRows); err != nil {
		fail("write benchmark results: %v", err)
	}

	fmt.Printf(
		"measured: offered=%d started=%d dropped=%d completed=%d failed=%d tokens=%d elapsed=%.2fs req/s=%.3f tok/s=%.3f\n",
		offered,
		started,
		dropped,
		completed,
		failed,
		generatedTokens,
		elapsedSeconds,
		summaryResult.SuccessfulReqPerSec,
		summaryResult.GeneratedTokPerSec,
	)
	fmt.Printf(
		"TTFT p50/p95=%.1f/%.1f ms; inter-token p50/p95=%.1f/%.1f ms; latency p50/p95=%.1f/%.1f ms\n",
		summaryResult.TTFT.P50,
		summaryResult.TTFT.P95,
		summaryResult.InterTokenLatency.P50,
		summaryResult.InterTokenLatency.P95,
		summaryResult.EndToEndLatency.P50,
		summaryResult.EndToEndLatency.P95,
	)
	fmt.Printf("results written under %s\n", *outputDir)
}

func runIdleBaseline(
	client *http.Client,
	endpoint string,
	model string,
	gpu string,
	outputDir string,
	prompts []promptProfile,
	outputMix []int,
	warmups int,
	repetitions int,
	seed int64,
) {
	for _, prompt := range prompts {
		for i := 0; i < warmups; i++ {
			result, _ := runRequest(
				context.Background(),
				client,
				endpoint,
				-(i + 1),
				prompt,
				outputMix[0],
				true,
			)
			if !result.Success {
				fail("warm-up for %s prompt failed: %s", prompt.Name, result.Error)
			}
			fmt.Printf("warm-up %s %d/%d complete (%d tokens)\n", prompt.Name, i+1, warmups, result.GeneratedTokens)
		}
	}

	start := time.Now()
	rows := make([]requestResult, 0, len(prompts)*len(outputMix)*repetitions)
	var allTTFT, allInterToken, allLatency, allPromptTokens []float64
	cases := make([]idleCaseSummary, 0, len(prompts)*len(outputMix))
	index := 0
	completedTotal, failedTotal, generatedTotal := 0, 0, 0

	for _, prompt := range prompts {
		for _, outputTokens := range outputMix {
			var caseTTFT, caseInterToken, caseLatency []float64
			caseCompleted, caseFailed := 0, 0
			for repetition := 1; repetition <= repetitions; repetition++ {
				index++
				result, intervals := runRequest(
					context.Background(),
					client,
					endpoint,
					index,
					prompt,
					outputTokens,
					false,
				)
				rows = append(rows, result)
				if !result.Success {
					caseFailed++
					failedTotal++
					fmt.Printf("idle sample failed: prompt=%s output_limit=%d: %s\n", prompt.Name, outputTokens, result.Error)
					continue
				}

				caseCompleted++
				completedTotal++
				generatedTotal += result.GeneratedTokens
				caseTTFT = append(caseTTFT, result.TTFTMillis)
				caseInterToken = append(caseInterToken, intervals...)
				caseLatency = append(caseLatency, result.LatencyMillis)
				allTTFT = append(allTTFT, result.TTFTMillis)
				allInterToken = append(allInterToken, intervals...)
				allLatency = append(allLatency, result.LatencyMillis)
				allPromptTokens = append(allPromptTokens, float64(result.PromptTokens))
				fmt.Printf(
					"idle sample %d/%d: prompt=%s output_limit=%d generated=%d TTFT=%.1fms latency=%.1fms\n",
					repetition,
					repetitions,
					prompt.Name,
					outputTokens,
					result.GeneratedTokens,
					result.TTFTMillis,
					result.LatencyMillis,
				)
			}
			cases = append(cases, idleCaseSummary{
				PromptName:      prompt.Name,
				RequestedTokens: outputTokens,
				Repetitions:     repetitions,
				Completed:       caseCompleted,
				Failed:          caseFailed,
				TTFT:            summarize(caseTTFT),
				InterToken:      summarize(caseInterToken),
				EndToEnd:        summarize(caseLatency),
			})
		}
	}
	elapsedSeconds := time.Since(start).Seconds()
	summaryResult := summary{
		Mode:               "idle",
		Model:              model,
		GPU:                gpu,
		GatewayURL:         endpoint,
		MeasurementSeconds: elapsedSeconds,
		ElapsedSeconds:     elapsedSeconds,
		WarmupRequests:     warmups * len(prompts),
		Seed:               seed,
		PromptProfiles:     prompts,
		OutputTokenMix:     outputMix,
		RequestsOffered:    len(rows),
		RequestsStarted:    len(rows),
		RequestsCompleted:  completedTotal,
		RequestsFailed:     failedTotal,
		GeneratedTokens:    generatedTotal,
		TTFT:               summarize(allTTFT),
		InterTokenLatency:  summarize(allInterToken),
		EndToEndLatency:    summarize(allLatency),
		PromptTokenCounts:  summarize(allPromptTokens),
		IdleCases:          cases,
	}
	if elapsedSeconds > 0 {
		summaryResult.SuccessfulReqPerSec = float64(completedTotal) / elapsedSeconds
		summaryResult.GeneratedTokPerSec = float64(generatedTotal) / elapsedSeconds
	}
	if err := writeResults(outputDir, summaryResult, rows); err != nil {
		fail("write idle baseline results: %v", err)
	}
	fmt.Printf(
		"idle baseline complete: completed=%d failed=%d generated_tokens=%d elapsed=%.2fs\n",
		completedTotal,
		failedTotal,
		generatedTotal,
		elapsedSeconds,
	)
	fmt.Printf("grouped metrics and per-request samples written under %s\n", outputDir)
}

func runRequest(
	parent context.Context,
	client *http.Client,
	endpoint string,
	index int,
	prompt promptProfile,
	outputTokens int,
	isWarmup bool,
) (requestResult, []float64) {
	startedAt := time.Now()
	result := requestResult{
		Index:           index,
		PromptName:      prompt.Name,
		PromptBytes:     len(prompt.Text),
		RequestedTokens: outputTokens,
		StartedAt:       startedAt,
		WasWarmup:       isWarmup,
	}
	body, err := json.Marshal(generateRequest{
		Prompt:          prompt.Text,
		MaxOutputTokens: outputTokens,
	})
	if err != nil {
		return finishFailure(result, err)
	}
	ctx, cancel := context.WithCancel(parent)
	defer cancel()
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return finishFailure(result, err)
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := client.Do(request)
	if err != nil {
		return finishFailure(result, err)
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		responseBody, _ := io.ReadAll(io.LimitReader(response.Body, 4096))
		return finishFailure(result, fmt.Errorf("gateway HTTP %d: %s", response.StatusCode, strings.TrimSpace(string(responseBody))))
	}
	result.PromptTokens, _ = strconv.Atoi(response.Header.Get("X-Prompt-Token-Count"))
	scanner := bufio.NewScanner(response.Body)
	scanner.Buffer(make([]byte, 4096), 4<<20)
	var lastTokenAt time.Time
	interTokenLatencies := make([]float64, 0, outputTokens)
	finished := false
	for scanner.Scan() {
		var event tokenEvent
		if err := json.Unmarshal(scanner.Bytes(), &event); err != nil {
			return finishFailureWithIntervals(result, interTokenLatencies, fmt.Errorf("parse gateway event: %w", err))
		}
		if event.Error != "" {
			return finishFailureWithIntervals(result, interTokenLatencies, errors.New(event.Error))
		}
		now := time.Now()
		if result.GeneratedTokens == 0 {
			result.TTFTMillis = float64(now.Sub(startedAt).Microseconds()) / 1000
		} else {
			interTokenLatencies = append(interTokenLatencies, float64(now.Sub(lastTokenAt).Microseconds())/1000)
		}
		lastTokenAt = now
		result.GeneratedTokens++
		if event.Finished {
			finished = true
		}
	}
	if err := scanner.Err(); err != nil {
		return finishFailureWithIntervals(result, interTokenLatencies, err)
	}
	if !finished {
		return finishFailureWithIntervals(result, interTokenLatencies, errors.New("stream ended without a finished token"))
	}
	result.Success = true
	result.CompletedAt = time.Now()
	result.LatencyMillis = float64(result.CompletedAt.Sub(startedAt).Microseconds()) / 1000
	if len(interTokenLatencies) > 0 {
		for _, interval := range interTokenLatencies {
			result.InterTokenMeanMs += interval
		}
		result.InterTokenMeanMs /= float64(len(interTokenLatencies))
	}
	return result, interTokenLatencies
}

func finishFailure(result requestResult, err error) (requestResult, []float64) {
	return finishFailureWithIntervals(result, nil, err)
}

func finishFailureWithIntervals(result requestResult, intervals []float64, err error) (requestResult, []float64) {
	result.Success = false
	result.Error = err.Error()
	result.CompletedAt = time.Now()
	result.LatencyMillis = float64(result.CompletedAt.Sub(result.StartedAt).Microseconds()) / 1000
	return result, intervals
}

func parseOutputMix(value string) ([]int, error) {
	var values []int
	for _, part := range strings.Split(value, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(part))
		if err != nil || n < 1 || n > 512 {
			return nil, fmt.Errorf("output lengths must be integers between 1 and 512")
		}
		values = append(values, n)
	}
	if len(values) == 0 {
		return nil, fmt.Errorf("provide at least one output length")
	}
	return values, nil
}

func summarize(values []float64) metricSummary {
	if len(values) == 0 {
		return metricSummary{}
	}
	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)
	sum := 0.0
	for _, value := range sorted {
		sum += value
	}
	return metricSummary{
		P50:  percentile(sorted, 0.50),
		P95:  percentile(sorted, 0.95),
		Mean: sum / float64(len(sorted)),
	}
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(math.Ceil(p*float64(len(sorted)))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

func writeResults(dir string, summaryResult summary, rows []requestResult) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	stamp := time.Now().UTC().Format("20060102T150405Z")
	summaryPath := filepath.Join(dir, "summary-"+stamp+".json")
	file, err := os.Create(summaryPath)
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	encodeErr := encoder.Encode(summaryResult)
	closeErr := file.Close()
	if encodeErr != nil {
		return encodeErr
	}
	if closeErr != nil {
		return closeErr
	}

	csvPath := filepath.Join(dir, "requests-"+stamp+".csv")
	csvFile, err := os.Create(csvPath)
	if err != nil {
		return err
	}
	writer := csv.NewWriter(csvFile)
	headers := []string{
		"index", "prompt_name", "prompt_bytes", "prompt_tokens",
		"requested_output_tokens", "generated_tokens", "ttft_ms",
		"inter_token_mean_ms", "latency_ms", "started_at", "completed_at",
		"success", "warmup", "error",
	}
	if err := writer.Write(headers); err != nil {
		_ = csvFile.Close()
		return err
	}
	for _, row := range rows {
		record := []string{
			strconv.Itoa(row.Index),
			row.PromptName,
			strconv.Itoa(row.PromptBytes),
			strconv.Itoa(row.PromptTokens),
			strconv.Itoa(row.RequestedTokens),
			strconv.Itoa(row.GeneratedTokens),
			fmt.Sprintf("%.3f", row.TTFTMillis),
			fmt.Sprintf("%.3f", row.InterTokenMeanMs),
			fmt.Sprintf("%.3f", row.LatencyMillis),
			row.StartedAt.Format(time.RFC3339Nano),
			row.CompletedAt.Format(time.RFC3339Nano),
			strconv.FormatBool(row.Success),
			strconv.FormatBool(row.WasWarmup),
			row.Error,
		}
		if err := writer.Write(record); err != nil {
			_ = csvFile.Close()
			return err
		}
	}
	writer.Flush()
	if err := writer.Error(); err != nil {
		_ = csvFile.Close()
		return err
	}
	return csvFile.Close()
}

func fail(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}
