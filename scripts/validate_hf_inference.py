#!/usr/bin/env python3
"""Validate Hugging Face Inference API credentials + model callability."""

import os
import sys
from huggingface_hub import InferenceClient
from dotenv import load_dotenv


def fail(msg: str, code: int = 1):
    print(f"❌ {msg}")
    sys.exit(code)


def main() -> int:
    load_dotenv()
    token = (
        os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
        or os.getenv("HF_TOKEN", "").strip()
        or os.getenv("HUGGINGFACE_API_TOKEN", "").strip()
    )
    model = os.getenv("HF_INFERENCE_API", "mistralai/Mistral-7B-Instruct-v0.2").strip()
    fallbacks = [
        candidate.strip()
        for candidate in os.getenv("HF_INFERENCE_FALLBACKS", "").split(",")
        if candidate.strip()
    ]
    timeout = float(os.getenv("HF_API_TIMEOUT", "45"))
    task = os.getenv("HF_INFERENCE_TASK", "conversational").strip() or "conversational"
    task_fallbacks = [
        t.strip()
        for t in os.getenv("HF_INFERENCE_TASK_FALLBACKS", "text-generation,conversational").split(",")
        if t.strip()
    ]

    if not token:
        fail("Missing Hugging Face token. Set HUGGINGFACEHUB_API_TOKEN (or HF_TOKEN / HUGGINGFACE_API_TOKEN).")

    if not model:
        fail("HF_INFERENCE_API is empty.")

    prompt = "Reply with exactly: ok"
    candidates = [model, *fallbacks]
    tasks = [task, *[t for t in task_fallbacks if t != task]]
    errors = []

    for candidate in candidates:
        for task_name in tasks:
            print(f"🔎 Checking model callability: {candidate} (task={task_name})")
            client = InferenceClient(model=candidate, token=token, timeout=timeout)

            try:
                if task_name == "conversational":
                    output = client.chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=16,
                        temperature=0.0,
                    )
                    sample = output.choices[0].message.content if output.choices else ""
                else:
                    sample = client.text_generation(prompt, max_new_tokens=8, temperature=0.0)

                print(f"✅ Inference succeeded with {candidate} (task={task_name}). Sample output: {sample!r}")
                print("✅ Hugging Face token + model/task combination appears callable.")
                return 0
            except Exception as exc:
                errors.append(f"{candidate}|{task_name}: {type(exc).__name__}: {exc}")

    fail(
        "Inference call failed for all configured model/task combinations. This usually means the token is invalid, "
        "the model is not available for your account/provider, or rate limits are hit.\n"
        + "\n".join(errors),
        code=2,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
