package main

import (
	"net/http"
	"strconv"
	"time"

	pb "github.com/thouartmammal/llm_inference_server/gateway"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

type gatewayTelemetry struct {
	registry *prometheus.Registry

	requestsTotal           prometheus.Counter
	requestsActive          prometheus.Gauge
	requestCancellations    *prometheus.CounterVec
	requestErrors           *prometheus.CounterVec
	streamedTokensTotal     prometheus.Counter
	requestDurationSeconds   prometheus.Histogram
	timeToFirstTokenSeconds  prometheus.Histogram
	interTokenLatencySecond  prometheus.Histogram
	schedulerQueueDepth      prometheus.Gauge
	schedulerBatchSize       *prometheus.HistogramVec
	workerRPCSeconds         *prometheus.HistogramVec
}

var defaultGatewayTelemetry = newGatewayTelemetry()

func newGatewayTelemetry() *gatewayTelemetry {
	registry := prometheus.NewRegistry()
	telemetry := &gatewayTelemetry{
		registry: registry,
		requestsTotal: prometheus.NewCounter(prometheus.CounterOpts{
			Namespace: "llm",
			Subsystem: "gateway",
			Name:      "requests_total",
			Help:      "Total HTTP generation requests accepted by the gateway.",
		}),
		requestsActive: prometheus.NewGauge(prometheus.GaugeOpts{
			Namespace: "llm",
			Subsystem: "gateway",
			Name:      "requests_active",
			Help:      "Current number of in-flight generation requests.",
		}),
		requestCancellations: prometheus.NewCounterVec(prometheus.CounterOpts{
			Namespace: "llm",
			Subsystem: "gateway",
			Name:      "request_cancellations_total",
			Help:      "Total generation requests canceled before natural completion.",
		}, []string{"reason"}),
		requestErrors: prometheus.NewCounterVec(prometheus.CounterOpts{
			Namespace: "llm",
			Subsystem: "gateway",
			Name:      "request_errors_total",
			Help:      "Total generation request failures.",
		}, []string{"reason"}),
		streamedTokensTotal: prometheus.NewCounter(prometheus.CounterOpts{
			Namespace: "llm",
			Subsystem: "gateway",
			Name:      "streamed_tokens_total",
			Help:      "Total generated tokens streamed to clients.",
		}),
		requestDurationSeconds: prometheus.NewHistogram(prometheus.HistogramOpts{
			Namespace: "llm",
			Subsystem: "gateway",
			Name:      "request_duration_seconds",
			Help:      "End-to-end request duration from accept to finish.",
			Buckets:   generationLatencyBuckets(),
		}),
		timeToFirstTokenSeconds: prometheus.NewHistogram(prometheus.HistogramOpts{
			Namespace: "llm",
			Subsystem: "gateway",
			Name:      "time_to_first_token_seconds",
			Help:      "Time from request accept to first streamed token.",
			Buckets:   generationLatencyBuckets(),
		}),
		interTokenLatencySecond: prometheus.NewHistogram(prometheus.HistogramOpts{
			Namespace: "llm",
			Subsystem: "gateway",
			Name:      "inter_token_latency_seconds",
			Help:      "Time between consecutive streamed tokens for a request.",
			Buckets:   generationLatencyBuckets(),
		}),
		schedulerQueueDepth: prometheus.NewGauge(prometheus.GaugeOpts{
			Namespace: "llm",
			Subsystem: "scheduler",
			Name:      "queue_depth",
			Help:      "Current number of queued generation requests waiting for a batch.",
		}),
		schedulerBatchSize: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Namespace: "llm",
			Subsystem: "scheduler",
			Name:      "batch_size",
			Help:      "Number of requests processed in a single scheduler batch.",
			Buckets:   []float64{1, 2, 4, 8, 16, 32},
		}, []string{"phase"}),
		workerRPCSeconds: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Namespace: "llm",
			Subsystem: "scheduler",
			Name:      "worker_rpc_seconds",
			Help:      "Time spent waiting for a worker ScheduleTask RPC response.",
			Buckets:   generationLatencyBuckets(),
		}, []string{"phase"}),
	}

	registry.MustRegister(
		telemetry.requestsTotal,
		telemetry.requestsActive,
		telemetry.requestCancellations,
		telemetry.requestErrors,
		telemetry.streamedTokensTotal,
		telemetry.requestDurationSeconds,
		telemetry.timeToFirstTokenSeconds,
		telemetry.interTokenLatencySecond,
		telemetry.schedulerQueueDepth,
		telemetry.schedulerBatchSize,
		telemetry.workerRPCSeconds,
	)

	return telemetry
}

func generationLatencyBuckets() []float64 {
	return []float64{0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60}
}

func (m *gatewayTelemetry) handler() http.Handler {
	return promhttp.HandlerFor(m.registry, promhttp.HandlerOpts{})
}

func (m *gatewayTelemetry) recordRequestAccepted() {
	m.requestsTotal.Inc()
	m.requestsActive.Inc()
}

func (m *gatewayTelemetry) recordRequestFinished(request *generationRequest, outcome string) {
	if request == nil {
		return
	}
	m.requestsActive.Dec()
	if !request.startedAt.IsZero() {
		duration := time.Since(request.startedAt).Seconds()
		m.requestDurationSeconds.Observe(duration)
	}
	switch outcome {
	case "completed":
		return
	case "client_cancelled", "output_limit":
		m.requestCancellations.WithLabelValues(outcome).Inc()
	default:
		m.requestErrors.WithLabelValues(outcome).Inc()
	}
}

func (m *gatewayTelemetry) recordTokenEmission(request *generationRequest) {
	if request == nil {
		return
	}
	now := time.Now()
	if request.tokensEmitted == 0 && !request.startedAt.IsZero() {
		m.timeToFirstTokenSeconds.Observe(now.Sub(request.startedAt).Seconds())
	} else if !request.lastTokenAt.IsZero() {
		m.interTokenLatencySecond.Observe(now.Sub(request.lastTokenAt).Seconds())
	}
	request.tokensEmitted++
	request.lastTokenAt = now
	m.streamedTokensTotal.Inc()
}

func (m *gatewayTelemetry) recordQueueDepth(depth int) {
	m.schedulerQueueDepth.Set(float64(depth))
}

func (m *gatewayTelemetry) recordBatchSize(phase pb.Phase, size int) {
	m.schedulerBatchSize.WithLabelValues(phaseLabel(phase)).Observe(float64(size))
}

func (m *gatewayTelemetry) recordWorkerRPC(phase pb.Phase, elapsed time.Duration) {
	m.workerRPCSeconds.WithLabelValues(phaseLabel(phase)).Observe(elapsed.Seconds())
}

func phaseLabel(phase pb.Phase) string {
	switch phase {
	case pb.Phase_PREFILL:
		return "prefill"
	case pb.Phase_DECODE:
		return "decode"
	default:
		return strconv.Itoa(int(phase))
	}
}

func telemetryForGateway(g *gateway) *gatewayTelemetry {
	if g != nil && g.metrics != nil {
		return g.metrics
	}
	return defaultGatewayTelemetry
}
