"""Exercise the worker directly through its gRPC tunnel.
"""

import argparse
import os
import sys
import uuid

import grpc
import scheduler_worker_pb2
import scheduler_worker_pb2_grpc
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        default=os.environ.get("WORKER_TARGET"),
        help="Tunnel host:port (or set WORKER_TARGET).",
    )
    parser.add_argument(
        "--prompt",
        default="What is a KV cache?",
        help="User prompt to send to the model.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Use plaintext only if the tunnel provider requires it.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds for each worker RPC (default: 120).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=512,
        help="Client-side safety bound; keep it at least as large as the worker cap.",
    )
    args = parser.parse_args()

    if not args.target:
        parser.error("provide --target or set WORKER_TARGET to the tunnel host:port")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    if args.max_output_tokens <= 0:
        parser.error("--max-output-tokens must be greater than zero")

    return args


def main():
    args = parse_args()
    request_id = f"rpc-test-{uuid.uuid4().hex}"

    if args.insecure:
        channel = grpc.insecure_channel(args.target)
    else:
        channel = grpc.secure_channel(
            args.target,
            grpc.ssl_channel_credentials(),
        )

    stub = scheduler_worker_pb2_grpc.SchedulerWorkerStub(channel)
    phase = scheduler_worker_pb2.PREFILL

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    messages = [{"role": "user", "content": args.prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_token_ids = tokenizer.encode(
        formatted_prompt,
        add_special_tokens=False,
    )
    generated_token_ids = []
    token_ids = [int(token_id) for token_id in prompt_token_ids]

    try:
        for generated_count in range(1, args.max_output_tokens + 1):
            request = scheduler_worker_pb2.ScheduleTaskRequest(
                request_id=request_id,
                phase=phase,
                token_ids=token_ids,
            )
            batch = scheduler_worker_pb2.ScheduleTaskBatch(requests=[request])

            try:
                response = stub.ScheduleTask(batch, timeout=args.timeout)
            except grpc.RpcError as error:
                print(
                    f"Worker RPC failed: {error.code().name}: {error.details()}",
                    file=sys.stderr,
                )
                return 1

            if len(response.sampled_tokens) != 1:
                raise RuntimeError(
                    "Expected exactly one sampled token, got "
                    f"{len(response.sampled_tokens)}"
                )

            sampled = response.sampled_tokens[0]
            if sampled.request_id != request_id:
                raise RuntimeError(
                    f"Worker returned request_id {sampled.request_id!r}; "
                    f"expected {request_id!r}"
                )

            print(
                f"token {generated_count}: request_id={sampled.request_id}, "
                f"token_id={sampled.token_id}, finished={sampled.finished}"
            )
            generated_token_ids.append(sampled.token_id)
            decoded_output = tokenizer.decode(
                    sampled.token_id,
                    skip_special_tokens=True,
                )

            print(f"decoded_output: {decoded_output}\n")
            
            if sampled.finished:
                answer = tokenizer.decode(
                    generated_token_ids,
                    skip_special_tokens=True,
                )
                print(f"Generated answer:\n{answer}")
                print(f"Worker RPC test passed ({generated_count} generated token(s)).")
                return 0

            # The first call prefills the prompt; later calls feed back the
            # previous sampled token so the worker can continue from its cache.
            phase = scheduler_worker_pb2.DECODE
            token_ids = [sampled.token_id]

        answer = tokenizer.decode(
            generated_token_ids,
            skip_special_tokens=True,
        )
        print(f"Generated answer so far (client limit reached):\n{answer}")
        raise RuntimeError(
            "Worker did not report finished within --max-output-tokens. "
            "Check that this bound is at least as large as the worker's cap."
        )
    finally:
        channel.close()


if __name__ == "__main__":
    raise SystemExit(main())
