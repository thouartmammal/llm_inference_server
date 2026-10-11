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
                for request in requests:
                    if request.request_id in self.caches:
                        context.abort(
                            grpc.StatusCode.ALREADY_EXISTS,
                            "PREFILL request_id already has an active cache",
                        )
                logits, past_key_values, attention_mask = self._prefill(requests)
                current_counts = [1] * len(requests)
                previous_cohorts = []
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

        layers = []
        for layer_index, layer in enumerate(self._cache_layers(past_key_values)):
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
        previous_cohorts = []
        previous_cohort_ids = set()
        cache_rows = []
        max_cache_length = 0
        for request in requests:
            cohort = self.caches[request.request_id]
            cohort_identity = id(cohort)
            if cohort_identity not in previous_cohort_ids:
                previous_cohort_ids.add(cohort_identity)
                previous_cohorts.append(cohort)

            row_index = cohort["request_ids"].index(request.request_id)
            legacy_cache = self._cache_layers(cohort["kv"])
            cache_rows.append(
                (
                    legacy_cache,
                    row_index,
                    cohort["attention_mask"][row_index : row_index + 1],
                    cohort["counts"][row_index],
                )
            )
            max_cache_length = max(
                max_cache_length,
                cohort["attention_mask"].shape[1],
            )

        layer_count = len(cache_rows[0][0])
        if any(len(row[0]) != layer_count for row in cache_rows):
            raise ValueError("decode caches have different numbers of layers")

        merged_layers = []
        for layer_index in range(layer_count):
            keys = []
            values = []
            optional_entries = []
            for legacy_cache, row_index, _, _ in cache_rows:
                layer = legacy_cache[layer_index]
                key, value = layer[:2]
                padding = max_cache_length - key.shape[-2]
                if padding:
                    key = self.torch.cat(
                        [
                            key.new_zeros(
                                (*key.shape[:-2], padding, key.shape[-1])
                            ),
                            key,
                        ],
                        dim=-2,
                    )
                    value = self.torch.cat(
                        [
                            value.new_zeros(
                                (*value.shape[:-2], padding, value.shape[-1])
                            ),
                            value,
                        ],
                        dim=-2,
                    )
                keys.append(key[row_index : row_index + 1])
                values.append(value[row_index : row_index + 1])
                optional_entries.append(layer[2:])

            for entries in optional_entries[1:]:
                if len(entries) != len(optional_entries[0]):
                    raise ValueError(
                        f"decode cache metadata differs at layer {layer_index}"
                    )
                for expected, actual in zip(optional_entries[0], entries):
                    if expected is actual:
                        continue
                    equal = (
                        self.torch.equal(expected, actual)
                        if hasattr(expected, "shape") and hasattr(actual, "shape")
                        else expected == actual
                    )
                    if not equal:
                        raise ValueError(
                            f"decode cache metadata differs at layer {layer_index}"
                        )
            merged_layers.append(
                (
                    self.torch.cat(keys, dim=0),
                    self.torch.cat(values, dim=0),
                    *optional_entries[0],
                )
            )

        cache_type = type(previous_cohorts[0]["kv"])
        past_key_values = cache_type(
            ddp_cache_data=tuple(merged_layers),
            config=getattr(self.model, "config", None),
        )
        attention_rows = []
        current_counts = []
        for _, _, mask, count in cache_rows:
            padding = max_cache_length - mask.shape[1]
            if padding:
                mask = self.torch.cat(
                    [
                        mask.new_zeros((1, padding)),
                        mask,
                    ],
                    dim=1,
                )
            attention_rows.append(mask)
            current_counts.append(count + 1)
        previous_mask = self.torch.cat(attention_rows, dim=0)
        next_mask_column = self.torch.ones(
            (len(requests), 1),
            dtype=previous_mask.dtype,
            device=previous_mask.device,
        )
        attention_mask = self.torch.cat(
            [previous_mask, next_mask_column],
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
        current_counts = [row[3] + 1 for row in cache_rows]
        return (
            output.logits,
            output.past_key_values,
            attention_mask,
            current_counts,
        )

    @staticmethod
    def _cache_layers(cache):
        to_legacy_cache = getattr(cache, "to_legacy_cache", None)
        if callable(to_legacy_cache):
            return tuple(to_legacy_cache())
        return tuple(cache)

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
        request_ids,
        past_key_values,
        attention_mask,
        counts,
        surviving_indices,
    ):
        for request_id in request_ids:
            self._remove_cache_request(request_id)
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
