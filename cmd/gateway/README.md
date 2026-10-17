# HTTP generation gateway

This development gateway applies the Qwen2.5 single-user chat template,
tokenizes prompts, schedules batched prefill and decode steps in the Python
worker, and streams newline-delimited JSON snapshots to each client as tokens
arrive.

## Prerequisites

- A running Python worker reachable through a gRPC tunnel.
- Python with `transformers` installed. The tokenizer helper downloads and
  caches tokenizer files for Qwen2.5-0.5B-Instruct at startup.
- Go dependencies from the repository's `go.mod`.

Set the worker target in PowerShell:

```powershell
$env:WORKER_TARGET = "your-tunnel-host:port"
```

If the `python` command does not select the environment with `transformers`,
set `PYTHON_EXECUTABLE` to that environment's Python executable. The helper
path defaults to `cmd/gateway/tokenizer_helper.py`; override it with
`TOKENIZER_HELPER_PATH` if needed.

The gateway uses TLS for the worker connection by default. Only if the tunnel
requires plaintext, set:

```powershell
$env:WORKER_INSECURE = "true"
```

Run the gateway from the repository root:

```powershell
go run .\cmd\gateway
```

It listens on `127.0.0.1:8080` by default. Override this with `GATEWAY_ADDR`.
`GET /healthz` is a process health check; it does not verify that the worker is
reachable.

`GET /metrics` exposes Prometheus metrics for the gateway and scheduler path.
The starter dashboard in `monitoring/grafana/dashboards/gateway.json` expects
metrics such as `llm_gateway_time_to_first_token_seconds`,
`llm_gateway_inter_token_latency_seconds`, `llm_scheduler_queue_depth`,
`llm_scheduler_batch_size`, and `llm_scheduler_worker_rpc_seconds`.

## Scheduler modes

Set `SCHEDULER_MODE` to `naive`, `static`, or `continuous`. The default is
`static`. The load generator's `-scheduler` flag labels result files only; it
does not configure the gateway.

### Naive and static

The `naive` mode uses one request per worker batch with no collection wait. It
reproduces the one-request-at-a-time baseline. The `static` mode (the default)
forms fixed cohorts: requests arriving after a cohort starts wait for the next
cohort, and finished or cancelled requests are not replaced during generation.

The default maximum batch size is 4 and the gateway waits up to 50 ms for a
cohort to fill. Set `STATIC_BATCH_SIZE` and `STATIC_BATCH_WAIT` to change these
values. For example:

```powershell
$env:SCHEDULER_MODE = "static"
$env:STATIC_BATCH_SIZE = "4"
$env:STATIC_BATCH_WAIT = "50ms"
go run .\cmd\gateway
```

To reproduce the Week 2 one-request-at-a-time baseline, use batch size 1 and
zero wait:

```powershell
$env:SCHEDULER_MODE = "naive"
go run .\cmd\gateway
```

### Continuous batching

Continuous mode keeps up to `STATIC_BATCH_SIZE` requests active. At each
decode-step boundary, queued requests are prefetched and admitted into free
slots; requests already decoding continue in the next shared decode batch.
Requests that finish, reach their output limit, or are cancelled free their
slots. The worker merges each request's modern Transformers KV cache row,
left-padding cache and attention-mask rows when their prompt/cache lengths
differ. This avoids converting Qwen's cache to the legacy tuple format.

```powershell
$env:SCHEDULER_MODE = "continuous"
$env:STATIC_BATCH_SIZE = "4"
$env:STATIC_BATCH_WAIT = "50ms"
go run .\cmd\gateway
```

The collection wait applies to initial batches and to incoming requests once at
least one free slot has an arrival queued; active decoding is not delayed while
waiting for a new arrival. A newcomer is prefetched separately before joining
the following decode step. Thus this is continuous decode-batch admission, not
simultaneous prefill and decode execution.

The worker left-pads prompts in each prefill batch and supplies attention masks
and position IDs so padding is ignored. The worker retains a separate KV cache
row for every active request and batches decode steps for the requests that
remain active. Update `worker/server.py` and both generated Python protobuf
files on the GPU host before testing this version.

## Request and streamed response

Send a JSON prompt:

```powershell
curl.exe -N -H "Content-Type: application/json" `
  --data-binary '{"prompt":"What is a KV cache?"}' `
  http://127.0.0.1:8080/generate
```

The response is newline-delimited JSON (`application/x-ndjson`). Each line is
an update for one generated token. `text` is the full decoded answer so far;
clients should replace their displayed text with the latest snapshot. A final
token has `"finished": true`. Errors that happen after streaming starts are
returned as an NDJSON object with an `"error"` field.

The gateway's output cap is fixed at 512 to match the worker's current cap.
Requests can set `max_output_tokens` from 1 through 512; when omitted, the
gateway uses 512. If the HTTP client disconnects, a request reaches that
per-request output limit, or generation errors after the worker request starts,
the gateway calls the worker's `CancelGeneration` RPC to release its cache.

Regenerate both language bindings after editing `scheduler-worker.proto`. The
Colab/Kaggle worker must run a version of `worker/server.py` and generated
Python stubs that include `CancelGeneration`.
