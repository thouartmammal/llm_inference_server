"""Minimal Qwen inference worker implementing scheduler-worker.proto."""

import logging
import os
import threading
from concurrent import futures

import grpc

import scheduler_worker_pb2
import scheduler_worker_pb2_grpc


MODEL_NAME =  "Qwen/Qwen2.5-0.5B-Instruct"
MAX_OUTPUT_TOKENS = 512
GRPC_PORT = 50051


class SchedulerWorkerServicer(
    scheduler_worker_pb2_grpc.SchedulerWorkerServicer
):
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        self.model.eval()
        self.stop_token_ids = {self.tokenizer.eos_token_id}
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None and im_end_id != self.tokenizer.unk_token_id:
            self.stop_token_ids.add(im_end_id)
        self.caches = {}
        self.lock = threading.Lock()
        self.cache_layout_logged = False

    def ScheduleTask(self, batch, context):
        try:
            return self._schedule_task(batch, context)
        except grpc.RpcError:
            raise
        except Exception:
            logging.exception("ScheduleTask failed")
            raise

    def _schedule_task(self, batch, context):
        with self.lock, self.torch.no_grad():
            requests = list(batch.requests)
            if not requests:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "batch must contain at least one request",
                )
            phase = requests[0].phase
            request_ids = set()
            for request in requests:
                if not request.request_id:
                    context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        "request_id must not be empty",
                    )
                if request.request_id in request_ids:
                    context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        "request IDs in a batch must be unique",
                    )
                request_ids.add(request.request_id)
                if not request.token_ids:
                    context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        "token_ids must not be empty",
                    )
                if request.phase != phase:
                    context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        "all requests in a batch must use the same phase",
                    )

            if not context.is_active():
                self._discard_caches(requests)
                context.abort(
                    grpc.StatusCode.CANCELLED,
                    "generation batch was cancelled",
                )

            if phase == scheduler_worker_pb2.PREFILL:
                logits, past_key_values, attention_mask = self._prefill(requests)
                current_counts = [1] * len(requests)
                previous_cohort = None
            elif phase == scheduler_worker_pb2.DECODE:
                for request in requests:
                    if request.request_id not in self.caches:
                        context.abort(
                            grpc.StatusCode.FAILED_PRECONDITION,
                            "DECODE requires a previous PREFILL for this request_id",
                        )
                (
                    logits,
                    past_key_values,
                    attention_mask,
                    current_counts,
                    previous_cohort,
                ) = self._decode(requests)
            else:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "phase must be PREFILL or DECODE",
                )

            if not context.is_active():
                self._discard_caches(requests)
                context.abort(
                    grpc.StatusCode.CANCELLED,
                    "generation batch was cancelled",
                )

            sampled = []
            surviving_indices = []
            for index, request in enumerate(requests):
                next_token_id = int(
                    self.torch.argmax(logits[index, -1, :], dim=-1).item()
                )
                finished = (
                    next_token_id in self.stop_token_ids
                    or current_counts[index] >= MAX_OUTPUT_TOKENS
                )
                if not finished:
                    surviving_indices.append(index)
                sampled.append(
                    scheduler_worker_pb2.SampledToken(
                        request_id=request.request_id,
                        token_id=next_token_id,
                        finished=finished,
                    )
                )
            self._replace_cohort(
                previous_cohort,
                [request.request_id for request in requests],
                past_key_values,
                attention_mask,
                current_counts,
                surviving_indices,
            )
        return scheduler_worker_pb2.ScheduleTaskResponse(sampled_tokens=sampled)

    def _prefill(self, requests):
        max_length = max(len(request.token_ids) for request in requests)
        input_rows = []
        mask_rows = []
        for request in requests:
            padding = max_length - len(request.token_ids)
            input_rows.append([0] * padding + list(request.token_ids))
            mask_rows.append([0] * padding + [1] * len(request.token_ids))

        input_ids = self.torch.tensor(
            input_rows,
            dtype=self.torch.long,
            device="cuda",
        )
        attention_mask = self.torch.tensor(
            mask_rows,
            dtype=self.torch.long,
            device="cuda",
        )
        position_ids = attention_mask.cumsum(dim=-1) - 1
        position_ids = self.torch.where(attention_mask == 0, 0, position_ids)
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        self._log_cache_layout(output.past_key_values)
        return output.logits, output.past_key_values, attention_mask

    def _log_cache_layout(self, past_key_values):
        if getattr(self, "cache_layout_logged", False):
            return
        cache = past_key_values
        if hasattr(cache, "to_legacy_cache"):
            cache = cache.to_legacy_cache()

        layers = []
        for layer_index, layer in enumerate(cache):
            entries = []
            for item in layer:
                if item is None:
                    entries.append("None")
                elif hasattr(item, "shape"):
                    entries.append(f"{type(item).__name__}{tuple(item.shape)}")
                else:
                    entries.append(type(item).__name__)
            layers.append(f"layer {layer_index}: [{', '.join(entries)}]")

        logging.info(
            "Initial model KV cache layout (%s): %s",
            type(past_key_values).__name__,
            "; ".join(layers),
        )
        self.cache_layout_logged = True

    def _decode(self, requests):
        cohort = self.caches[requests[0].request_id]
        if any(self.caches[request.request_id] is not cohort for request in requests):
            raise ValueError("all decode requests must belong to the same static batch")

        cohort_indices = [
            cohort["request_ids"].index(request.request_id)
            for request in requests
        ]
        self._select_cohort_rows(cohort, cohort_indices)
        cohort["request_ids"] = [request.request_id for request in requests]

        past_key_values = cohort["kv"]
        previous_mask = cohort["attention_mask"]
        attention_mask = previous_mask
        next_mask_column = self.torch.ones(
            (len(requests), 1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_mask = self.torch.cat(
            [attention_mask, next_mask_column],
            dim=1,
        )
        position_ids = attention_mask.sum(dim=1, keepdim=True) - 1
        input_ids = self.torch.tensor(
            [[request.token_ids[-1]] for request in requests],
            dtype=self.torch.long,
            device="cuda",
        )
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        current_counts = [cohort["counts"][index] + 1 for index in range(len(requests))]
        return (
            output.logits,
            output.past_key_values,
            attention_mask,
            current_counts,
            cohort,
        )

    def _select_cohort_rows(self, cohort, indices):
        if not hasattr(cohort["kv"], "batch_select_indices"):
            raise TypeError(
                "model returned a KV cache without batch_select_indices; "
                "use a Transformers Cache implementation for batched decoding"
            )
        index_tensor = self.torch.tensor(
            indices,
            dtype=self.torch.long,
            device=cohort["attention_mask"].device,
        )
        cohort["kv"].batch_select_indices(index_tensor)
        cohort["attention_mask"] = cohort["attention_mask"].index_select(
            0,
            index_tensor,
        )
        cohort["counts"] = [cohort["counts"][index] for index in indices]
        cohort["request_ids"] = [
            cohort["request_ids"][index] for index in indices
        ]

    def _replace_cohort(
        self,
        previous_cohort,
        request_ids,
        past_key_values,
        attention_mask,
        counts,
        surviving_indices,
    ):
        if previous_cohort is not None:
            for request_id in previous_cohort["request_ids"]:
                if self.caches.get(request_id) is previous_cohort:
                    self.caches.pop(request_id, None)
        if not surviving_indices:
            return

        cohort = {
            "request_ids": list(request_ids),
            "kv": past_key_values,
            "attention_mask": attention_mask,
            "counts": list(counts),
        }
        if len(surviving_indices) != len(request_ids):
            self._select_cohort_rows(cohort, surviving_indices)
        for request_id in cohort["request_ids"]:
            self.caches[request_id] = cohort

    def _discard_caches(self, requests):
        for request in requests:
            self._remove_cache_request(request.request_id)

    def _remove_cache_request(self, request_id):
        cohort = self.caches.get(request_id)
        if cohort is None:
            return
        keep_indices = [
            index
            for index, cached_id in enumerate(cohort["request_ids"])
            if cached_id != request_id
        ]
        for cached_id in cohort["request_ids"]:
            self.caches.pop(cached_id, None)
        if not keep_indices:
            return
        self._select_cohort_rows(cohort, keep_indices)
        for cached_id in cohort["request_ids"]:
            self.caches[cached_id] = cohort

    def CancelGeneration(self, request, context):
        if not request.request_id:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "request_id must not be empty",
            )
        with self.lock:
            existed = request.request_id in self.caches
            self._remove_cache_request(request.request_id)
        return scheduler_worker_pb2.CancelGenerationResponse(
            cache_released=existed
        )


def serve():
    if MAX_OUTPUT_TOKENS < 1:
        raise ValueError("MAX_OUTPUT_TOKENS must be a positive integer")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    scheduler_worker_pb2_grpc.add_SchedulerWorkerServicer_to_server(
        SchedulerWorkerServicer(),
        server,
    )
    if server.add_insecure_port(f"[::]:{GRPC_PORT}") == 0:
        raise RuntimeError(f"could not bind gRPC server on port {GRPC_PORT}")
    server.start()
    print(f"gRPC server started and listening on port {GRPC_PORT}...", flush=True)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
