"""Minimal Qwen inference worker implementing scheduler-worker.proto."""

import os
import threading
from concurrent import futures

import grpc

import scheduler_worker_pb2
import scheduler_worker_pb2_grpc


MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "512"))
GRPC_PORT = int(os.environ.get("GRPC_PORT", "50051"))


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

    def ScheduleTask(self, batch, context):
        sampled = []
        with self.lock, self.torch.no_grad():
            for request in batch.requests:
                if not context.is_active():
                    self.caches.pop(request.request_id, None)
                    context.abort(
                        grpc.StatusCode.CANCELLED,
                        "generation request was cancelled",
                    )
                if not request.request_id:
                    context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        "request_id must not be empty",
                    )
                if not request.token_ids:
                    context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        "token_ids must not be empty",
                    )

                if request.phase == scheduler_worker_pb2.PREFILL:
                    self.caches.pop(request.request_id, None)
                    token_tensor = self.torch.tensor(
                        [request.token_ids],
                        dtype=self.torch.long,
                        device="cuda",
                    )
                    output = self.model(input_ids=token_tensor, use_cache=True)
                    current_count = 1
                elif request.phase == scheduler_worker_pb2.DECODE:
                    cache = self.caches.get(request.request_id)
                    if cache is None:
                        context.abort(
                            grpc.StatusCode.FAILED_PRECONDITION,
                            "DECODE requires a previous PREFILL for this request_id",
                        )
                    token_tensor = self.torch.tensor(
                        [[request.token_ids[-1]]],
                        dtype=self.torch.long,
                        device="cuda",
                    )
                    output = self.model(
                        input_ids=token_tensor,
                        past_key_values=cache["kv"],
                        use_cache=True,
                    )
                    current_count = cache["count"] + 1
                else:
                    context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        "phase must be PREFILL or DECODE",
                    )

                if not context.is_active():
                    self.caches.pop(request.request_id, None)
                    context.abort(
                        grpc.StatusCode.CANCELLED,
                        "generation request was cancelled",
                    )

                next_token_id = int(
                    self.torch.argmax(output.logits[:, -1, :], dim=-1).item()
                )
                finished = (
                    next_token_id in self.stop_token_ids
                    or current_count >= MAX_OUTPUT_TOKENS
                )
                if finished:
                    self.caches.pop(request.request_id, None)
                else:
                    self.caches[request.request_id] = {
                        "kv": output.past_key_values,
                        "count": current_count,
                    }

                sampled.append(
                    scheduler_worker_pb2.SampledToken(
                        request_id=request.request_id,
                        token_id=next_token_id,
                        finished=finished,
                    )
                )
        return scheduler_worker_pb2.ScheduleTaskResponse(sampled_tokens=sampled)

    def CancelGeneration(self, request, context):
        if not request.request_id:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "request_id must not be empty",
            )
        with self.lock:
            cache_released = self.caches.pop(request.request_id, None) is not None
        return scheduler_worker_pb2.CancelGenerationResponse(
            cache_released=cache_released
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
