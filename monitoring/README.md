# Monitoring

This directory contains the starter Prometheus and Grafana setup for the Go gateway and scheduler metrics.

## Files
- `prometheus.yml`: scrapes the Go gateway on `/metrics`
- `grafana/provisioning/datasources/datasource.yml`: provisions Prometheus in Grafana
- `grafana/provisioning/dashboards/dashboards.yml`: loads dashboards from disk
- `grafana/dashboards/gateway.json`: starter dashboard for TTFT, inter-token latency, queue depth, batch size, and worker RPC latency

## Expected metric names
- `llm_gateway_requests_total`
- `llm_gateway_requests_active`
- `llm_gateway_request_cancellations_total`
- `llm_gateway_request_errors_total`
- `llm_gateway_streamed_tokens_total`
- `llm_gateway_request_duration_seconds`
- `llm_gateway_time_to_first_token_seconds`
- `llm_gateway_inter_token_latency_seconds`
- `llm_scheduler_queue_depth`
- `llm_scheduler_batch_size`
- `llm_scheduler_worker_rpc_seconds`
