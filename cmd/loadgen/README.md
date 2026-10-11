# Gateway benchmarks

`cmd/loadgen` supports Poisson-arrival load tests and sequential idle
measurements. For a Week 2 versus Week 3 comparison, use Poisson mode with the
same model, GPU, seed, prompt/output mix, arrival rate, duration, and warm-up
count for both runs. The examples below use `0.2` requests/second for 60
seconds, higher than the earlier `0.03` run. This should build a backlog behind
the one-at-a-time naive gateway and give the static scheduler a better chance
to form batches. The actual batch sizes are not currently recorded, so this
benchmark compares performance but does not prove what batch sizes occurred.

## Naive baseline

Start the gateway in one terminal:

```powershell
$env:STATIC_BATCH_SIZE = "1"
$env:STATIC_BATCH_WAIT = "0s"
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

The load-generator scheduler flags label the result files; they do not configure
the gateway. Set the gateway environment variables to match the labels. Each
run writes JSON and CSV files under `benchmark-results`. The JSON records the
scheduler label, batch size, and batch wait alongside throughput and latency
metrics.

Compare successful requests/second and generated tokens/second, TTFT,
inter-token latency, end-to-end latency, and failures. A low offered rate may
not fill batches, while a higher rate increases queueing. Use the same `-rate`
and `-duration` for both runs, and allow each command to finish draining its
queue before starting the next run. If the higher rate creates excessive
latency or takes too long to drain, stop and repeat both modes at a lower
matching rate. Keep all Week 2 result files as the naive baseline.

Idle mode is useful for single-request references, but it cannot demonstrate
the throughput benefit of batching because its requests are sequential.
