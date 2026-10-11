# Gateway benchmarks

`cmd/loadgen` supports Poisson-arrival load tests and sequential idle
measurements. For a scheduler comparison, use Poisson mode with the same model,
GPU, seed, prompt/output mix, arrival rate, duration, and warm-up count for
each run. The examples use `0.2` requests/second for 60 seconds. The offered
rate should build a backlog behind the naive gateway and give batched modes a
chance to form batches. Results compare performance but do not prove what
batch sizes occurred because per-step batch-size distributions are not
currently recorded.

## Naive baseline

Start the gateway in one terminal:

```powershell
$env:SCHEDULER_MODE = "naive"
go run .\cmd\gateway
```

In another terminal, run the load generator:

```powershell
go run .\cmd\loadgen `
  -rate 0.2 `
  -duration 60s `
  -warmup 1 `
  -output-lengths 32,64,128 `
  -seed 1 `
  -scheduler naive `
  -batch-size 1 `
  -batch-wait 0s `
  -gpu "Google Colab T4"
```

## Static batching

Restart the gateway with the static batch settings:

```powershell
$env:SCHEDULER_MODE = "static"
$env:STATIC_BATCH_SIZE = "4"
$env:STATIC_BATCH_WAIT = "50ms"
go run .\cmd\gateway
```

Run the same workload, changing only the scheduler metadata:

```powershell
go run .\cmd\loadgen `
  -rate 0.2 `
  -duration 60s `
  -warmup 1 `
  -output-lengths 32,64,128 `
  -seed 1 `
  -scheduler static `
  -batch-size 4 `
  -batch-wait 50ms `
  -gpu "Google Colab T4"
```

## Continuous batching

Restart the gateway in continuous mode with the same active-batch limit:

```powershell
$env:SCHEDULER_MODE = "continuous"
$env:STATIC_BATCH_SIZE = "4"
$env:STATIC_BATCH_WAIT = "50ms"
go run .\cmd\gateway
```

Run the identical workload, changing only the scheduler label:

```powershell
go run .\cmd\loadgen `
  -rate 0.2 `
  -duration 60s `
  -warmup 1 `
  -output-lengths 32,64,128 `
  -seed 1 `
  -scheduler continuous `
  -batch-size 4 `
  -batch-wait 50ms `
  -gpu "Google Colab T4"
```

The load-generator scheduler flags label the result files; they do not configure
the gateway. Set `SCHEDULER_MODE`, `STATIC_BATCH_SIZE`, and `STATIC_BATCH_WAIT`
to match each label. Each run writes JSON and CSV files under
`benchmark-results`. The JSON records the scheduler label, batch size, and
batch wait alongside throughput and latency metrics.

Compare successful requests/second and generated tokens/second, TTFT,
inter-token latency, end-to-end latency, and failures. A low offered rate may
not fill batches, while a higher rate increases queueing. Use the same `-rate`
and `-duration` for both runs, and allow each command to finish draining its
queue before starting the next run. If the higher rate creates excessive
latency or takes too long to drain, stop and repeat both modes at a lower
matching rate. Keep all Week 2 result files as the naive baseline.

Idle mode is useful for single-request references, but it cannot demonstrate
the throughput benefit of batching because its requests are sequential.
