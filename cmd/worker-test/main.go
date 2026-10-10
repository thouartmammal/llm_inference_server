package main

import (
	"context"
	"crypto/tls"
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"time"

	pb "github.com/thouartmammal/llm_inference_server/gateway"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
)

func main() {
	target := flag.String("target", os.Getenv("WORKER_TARGET"), "Worker tunnel host:port (or set WORKER_TARGET)")
	tokenIDsArg := flag.String("token-ids", "123,345,900", "Comma-separated prompt token IDs")
	useInsecure := flag.Bool("insecure", false, "Use plaintext only if the tunnel requires it")
	timeout := flag.Duration("timeout", 120*time.Second, "Timeout for each worker RPC")
	maxOutputTokens := flag.Int("max-output-tokens", 512, "Maximum tokens to request before stopping")
	flag.Parse()

	if *target == "" {
		log.Fatal("set WORKER_TARGET or pass -target tunnel-host:port")
	}
	if *timeout <= 0 {
		log.Fatal("-timeout must be greater than zero")
	}
	if *maxOutputTokens <= 0 {
		log.Fatal("-max-output-tokens must be greater than zero")
	}

	tokenIDs, err := parseTokenIDs(*tokenIDsArg)
	if err != nil {
		log.Fatalf("invalid -token-ids: %v", err)
	}

	var transportCredentials credentials.TransportCredentials
	if *useInsecure {
		transportCredentials = insecure.NewCredentials()
	} else {
		transportCredentials = credentials.NewTLS(&tls.Config{
			MinVersion: tls.VersionTLS12,
		})
	}

	conn, err := grpc.NewClient(
		*target,
		grpc.WithTransportCredentials(transportCredentials),
	)
	if err != nil {
		log.Fatalf("create gRPC connection: %v", err)
	}
	defer conn.Close()

	client := pb.NewSchedulerWorkerClient(conn)
	requestID := fmt.Sprintf("go-tunnel-test-%d", time.Now().UnixNano())
	phase := pb.Phase_PREFILL

	fmt.Printf("Calling worker at %s with request ID %s\n", *target, requestID)
	for generatedCount := 1; generatedCount <= *maxOutputTokens; generatedCount++ {
		ctx, cancel := context.WithTimeout(context.Background(), *timeout)
		response, callErr := client.ScheduleTask(ctx, &pb.ScheduleTaskBatch{
			Requests: []*pb.ScheduleTaskRequest{{
				RequestId: requestID,
				Phase:     phase,
				TokenIds:  tokenIDs,
			}},
		})
		cancel()
		if callErr != nil {
			log.Fatalf("ScheduleTask RPC failed: %v", callErr)
		}
		if len(response.GetSampledTokens()) != 1 {
			log.Fatalf("expected one sampled token, got %d", len(response.GetSampledTokens()))
		}

		sampled := response.GetSampledTokens()[0]
		if sampled.GetRequestId() != requestID {
			log.Fatalf("worker returned request ID %q; expected %q", sampled.GetRequestId(), requestID)
		}
		fmt.Printf(
			"token %d: token_id=%d finished=%t\n",
			generatedCount,
			sampled.GetTokenId(),
			sampled.GetFinished(),
		)
		if sampled.GetFinished() {
			fmt.Println("Go-to-worker gRPC tunnel test passed.")
			return
		}

		phase = pb.Phase_DECODE
		tokenIDs = []int32{sampled.GetTokenId()}
	}

	log.Fatalf(
		"worker did not finish within %d tokens; ensure this limit is at least the worker's output cap",
		*maxOutputTokens,
	)
}

func parseTokenIDs(value string) ([]int32, error) {
	parts := strings.Split(value, ",")
	tokenIDs := make([]int32, 0, len(parts))
	for _, part := range parts {
		parsed, err := strconv.ParseInt(strings.TrimSpace(part), 10, 32)
		if err != nil {
			return nil, fmt.Errorf("%q is not a valid token ID", part)
		}
		tokenIDs = append(tokenIDs, int32(parsed))
	}
	if len(tokenIDs) == 0 {
		return nil, fmt.Errorf("provide at least one token ID")
	}
	return tokenIDs, nil
}
