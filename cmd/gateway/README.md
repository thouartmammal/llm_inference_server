# HTTP generation gateway

This development gateway applies the Qwen2.5 single-user chat template,
tokenizes prompts, forms fixed cohorts of requests for batched prefill and
decode steps in the Python worker, and streams newline-delimited JSON
snapshots to each client as tokens arrive.

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

## Static batching

The gateway defaults to a maximum batch size of 4 and waits up to 50 ms for a
cohort to fill. A batch starts when it fills or the wait expires. Once started,
it is a fixed cohort: requests that arrive later wait for the next batch, and
finished or cancelled requests are removed without admitting replacements.
Set `STATIC_BATCH_SIZE` and `STATIC_BATCH_WAIT` before starting the gateway to
change these values. For example:

```powershell
$env:STATIC_BATCH_SIZE = "4"
$env:STATIC_BATCH_WAIT = "50ms"
go run .\cmd\gateway
```

To reproduce the Week 2 one-request-at-a-time baseline, use batch size 1 and
zero wait:

```powershell
$env:STATIC_BATCH_SIZE = "1"
$env:STATIC_BATCH_WAIT = "0s"
go run .\cmd\gateway
```

The worker left-pads prompts in each prefill batch and supplies attention masks
and position IDs so padding is ignored. The worker retains a separate KV cache
for every request in the cohort and batches decode steps for the requests that
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
