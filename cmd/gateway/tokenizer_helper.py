"""Keep the model's Hugging Face tokenizer loaded for Go gateway requests."""

import json
import sys

from transformers import AutoTokenizer


def main():
    model_name = sys.argv[1]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(json.dumps({"ready": True}), flush=True)

    for line in sys.stdin:
        try:
            command = json.loads(line)
            operation = command.get("operation")

            if operation == "encode":
                messages = [{"role": "user", "content": command["prompt"]}]
                encoded = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
                token_ids = getattr(encoded, "input_ids", encoded)
                response = {"token_ids": list(token_ids)}
            elif operation == "decode":
                response = {
                    "text": tokenizer.decode(
                        command["token_ids"],
                        skip_special_tokens=True,
                    )
                }
            else:
                raise ValueError(f"unsupported tokenizer operation: {operation!r}")
        except Exception as error:
            response = {"error": str(error)}

        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"error": str(error)}), flush=True)
        raise
