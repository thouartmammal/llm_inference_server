# HTTP generation gateway

This development gateway accepts one user prompt at a time, applies the
Qwen2.5 single-user chat template, tokenizes it, calls the Python worker for
prefill and decode steps, and streams newline-delimited JSON snapshots as
tokens arrive.

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

The gateway serializes generation requests to provide a simple one-at-a-time
baseline. Its output cap is fixed at 512 to match the worker's current cap.
Requests can set `max_output_tokens` from 1 through 512; when omitted, the
gateway uses 512. If the HTTP client disconnects, a request reaches that
per-request output limit, or generation errors after the worker request starts,
the gateway calls the worker's `CancelGeneration` RPC to release its cache.

Regenerate both language bindings after editing `scheduler-worker.proto`. The
Colab/Kaggle worker must run a version of `worker/server.py` and generated
Python stubs that include `CancelGeneration`.
