#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Benchmark local vLLM or SGLang serving throughput via /v1/completions.

The script can either launch the backend server itself or benchmark an already
running endpoint. By default it uses a built-in synthetic prompt set so that it
does not depend on external benchmark datasets.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import math
import os
import pathlib
import random
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


DEFAULT_CONCURRENCY_CANDIDATES = [1, 2, 4, 8, 16, 32, 64, 128, 256]

PROMPT_TASKS = [
    "Summarize the document in four concise bullet points.",
    "Answer the question using only the document content.",
    "Extract the key entities and explain their roles briefly.",
    "Write a short technical abstract for the document.",
    "List the main risks and the mitigation ideas.",
    "Explain the core optimization strategy in plain language.",
    "Produce a compact changelog style summary.",
    "Identify the most likely bottleneck and justify it.",
]

PROMPT_QUESTIONS = [
    "What are the most important details a performance engineer should notice?",
    "Which parts of the text are most relevant for scaling inference?",
    "What trade-offs does the text imply between latency and throughput?",
    "Which implementation choices appear to matter the most?",
    "What should be monitored during a production rollout?",
    "What would you test next after reading this context?",
    "Which claims in the text need stronger evidence?",
    "What is the shortest reliable conclusion you can draw?",
]

CORPUS_SENTENCES = [
    "Modern inference servers batch requests aggressively to improve accelerator utilization while still trying to keep tail latency bounded.",
    "Long prompts spend more time in prefill, whereas long generations shift pressure toward decode scheduling and KV cache management.",
    "Kernel launch overhead, memory bandwidth limits, and host side request handling can all become throughput bottlenecks at scale.",
    "Tensor parallelism increases aggregate memory and compute capacity, but it also adds communication overhead that can dominate for smaller models.",
    "Pipeline parallelism helps fit larger models, although it may reduce efficiency when request sizes are highly variable.",
    "Prefill heavy workloads and decode heavy workloads often favor different scheduler settings and different concurrency targets.",
    "Prefix caching can improve effective throughput when many requests share a long common context, but the gains depend on cache hit rate.",
    "Quantized checkpoints reduce memory pressure and may raise throughput, yet the exact effect depends on kernels, hardware, and model architecture.",
    "A reliable benchmark should keep prompts deterministic, report failures clearly, and separate warmup behavior from steady state behavior.",
    "OpenAI compatible endpoints make it easier to swap serving backends, but backend specific launch flags still matter for final performance.",
    "The best client side concurrency is usually not obvious in advance, so a short sweep is often needed to find the saturation point.",
    "Production benchmarks should record both request throughput and token throughput because they can move in opposite directions.",
    "Large generation limits can expose scheduler contention, queue growth, and timeout settings that do not appear in short synthetic tests.",
    "Server memory fractions should be chosen carefully because pushing them too high can produce unstable out of memory failures during prefill.",
    "Reasoning tuned models, tool use parsers, and multimodal adapters often require backend specific extra arguments at launch time.",
    "Benchmark outputs are most useful when they include the exact launch command, endpoint configuration, prompt lengths, and concurrency values.",
]

SHARED_PREFIX_SENTENCES = [
    "The serving platform receives a mixture of batch analytics prompts, interactive chat turns, and code generation requests.",
    "Operators want to maximize accelerator occupancy without creating unstable latency spikes for medium sized prompts.",
    "The deployment must support quantized checkpoints, configurable parallelism, and reproducible benchmark settings across machines.",
    "A good benchmark should measure request throughput, prompt token throughput, completion token throughput, and robust latency percentiles.",
    "Shared system prompts can improve cache locality when a large number of requests use the same context but differ in a short suffix.",
]


class BenchmarkError(RuntimeError):
    """Raised when benchmark execution fails."""


@dataclass
class RequestResult:
    ok: bool
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    output_chars: int
    error: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM or SGLang serving throughput.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--backend", choices=("vllm", "sglang"), required=True)
    parser.add_argument("--model", required=True, help="Model path or model id to serve.")
    parser.add_argument(
        "--served-model-name",
        default="",
        help="Model name exposed through the API. Defaults to the model basename.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Serving port. Defaults to 8000 for vLLM and 30000 for SGLang.",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Optional API key used for both server launch and client requests.",
    )
    parser.add_argument(
        "--skip-launch",
        action="store_true",
        help="Benchmark an already running server instead of starting a new one.",
    )
    parser.add_argument(
        "--keep-server",
        action="store_true",
        help="Do not stop the launched server after the benchmark finishes.",
    )
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument(
        "--tensor-parallel",
        default="auto",
        help="auto or a positive integer.",
    )
    parser.add_argument(
        "--pipeline-parallel",
        default="1",
        help="auto or a positive integer.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vLLM only. Passed as --gpu-memory-utilization when launching vLLM.",
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=0.9,
        help="SGLang only. Passed as --mem-fraction-static when launching SGLang.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional model context limit. Maps to backend specific flags.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass --trust-remote-code to the backend launcher.",
    )
    parser.add_argument(
        "--server-extra-args",
        default="",
        help="Extra backend specific launch args, written as one shell style string.",
    )
    parser.add_argument("--num-requests", type=int, default=256)
    parser.add_argument("--warmup-requests", type=int, default=4)
    parser.add_argument(
        "--dataset",
        choices=("mixed", "shared-prefix"),
        default="mixed",
        help="Built-in prompt generator used when --prompts-file is not supplied.",
    )
    parser.add_argument(
        "--approx-prompt-tokens",
        type=int,
        default=1024,
        help="Approximate prompt length used by the built-in prompt generator.",
    )
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument(
        "--max-concurrency",
        default="auto",
        help="auto or a positive integer. Auto runs a short sweep and picks the best throughput.",
    )
    parser.add_argument(
        "--auto-concurrency-candidates",
        default="",
        help="Comma separated candidate list for auto mode, e.g. 1,2,4,8,16,32,64.",
    )
    parser.add_argument(
        "--auto-tune-requests",
        type=int,
        default=64,
        help="Number of requests used during auto concurrency tuning.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260413)
    parser.add_argument(
        "--prompts-file",
        default="",
        help="Optional TXT/JSON/JSONL prompt file. JSON can be a list[str] or {'prompts': [...]}.",
    )
    parser.add_argument(
        "--extra-request-body",
        default="{}",
        help="JSON object merged into every /v1/completions request body.",
    )
    parser.add_argument(
        "--results-dir",
        default="",
        help="Directory where logs and benchmark results will be written.",
    )
    parser.add_argument("--tag", default="", help="Optional run tag added to the output directory name.")
    args = parser.parse_args()

    if args.port is None:
        args.port = 8000 if args.backend == "vllm" else 30000

    for name in ("num_requests", "warmup_requests", "approx_prompt_tokens", "output_tokens", "auto_tune_requests"):
        value = getattr(args, name)
        if value < 0:
            parser.error(f"--{name.replace('_', '-')} must be >= 0")

    if args.num_requests < 1:
        parser.error("--num-requests must be >= 1")

    for name in ("startup_timeout", "request_timeout"):
        value = getattr(args, name)
        if value < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")

    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        parser.error("--gpu-memory-utilization must be in (0, 1]")

    if not 0.0 < args.mem_fraction_static <= 1.0:
        parser.error("--mem-fraction-static must be in (0, 1]")

    return args


def parse_json_object(raw: str, flag_name: str) -> Dict[str, Any]:
    value = raw.strip()
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"{flag_name} must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise BenchmarkError(f"{flag_name} must decode to a JSON object.")
    return parsed


def parse_positive_int_or_auto(raw: str, name: str, auto_value: int) -> int:
    value = str(raw).strip().lower()
    if value == "auto":
        return max(1, auto_value)
    try:
        parsed = int(value)
    except ValueError as exc:
        raise BenchmarkError(f"{name} must be 'auto' or a positive integer, got {raw!r}.") from exc
    if parsed < 1:
        raise BenchmarkError(f"{name} must be >= 1.")
    return parsed


def parse_candidate_list(raw: str, limit: int) -> List[int]:
    if raw.strip():
        candidates: List[int] = []
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                value = int(item)
            except ValueError as exc:
                raise BenchmarkError(
                    f"--auto-concurrency-candidates must be a comma separated integer list, got {raw!r}."
                ) from exc
            if value < 1:
                raise BenchmarkError("--auto-concurrency-candidates values must be >= 1.")
            candidates.append(value)
    else:
        candidates = list(DEFAULT_CONCURRENCY_CANDIDATES)
    filtered = sorted(set(value for value in candidates if value <= limit))
    return filtered or [1]


def detect_gpu_count() -> int:
    for env_name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        raw = os.environ.get(env_name, "").strip()
        if raw and raw.lower() not in {"-1", "none", "void"}:
            return len([token for token in raw.split(",") if token.strip()])

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            count = int(torch.cuda.device_count())
            if count > 0:
                return count
    except Exception:
        pass

    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
            )
            count = len([line for line in result.stdout.splitlines() if line.strip()])
            if count > 0:
                return count
        except Exception:
            pass

    if shutil.which("rocm-smi"):
        try:
            result = subprocess.run(
                ["rocm-smi", "-i"],
                check=True,
                capture_output=True,
                text=True,
            )
            matches = re.findall(r"GPU\[(\d+)\]", result.stdout)
            if matches:
                return len(set(matches))
        except Exception:
            pass

    return 1


def default_served_model_name(model: str) -> str:
    normalized = model.rstrip("/").replace("\\", "/")
    candidate = normalized.split("/")[-1] if normalized else model
    return candidate or "served-model"


def slugify(value: str) -> str:
    output = []
    for char in value:
        if char.isalnum() or char in {"-", "_", "."}:
            output.append(char)
        else:
            output.append("_")
    collapsed = "".join(output).strip("._")
    return collapsed or "run"


def resolve_results_dir(args: argparse.Namespace, served_model_name: str) -> pathlib.Path:
    if args.results_dir:
        result_dir = pathlib.Path(args.results_dir).expanduser().resolve()
        result_dir.mkdir(parents=True, exist_ok=True)
        return result_dir

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    base = repo_root / "results" / "benchmark"
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [args.backend]
    if args.tag:
        parts.append(slugify(args.tag))
    parts.append(slugify(served_model_name))
    parts.append(timestamp)
    result_dir = base / "_".join(parts)
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def build_headers(api_key: str) -> Dict[str, str]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Connection": "close",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def extract_output_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            text = first.get("text")
            if isinstance(text, str):
                return text
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
    text = payload.get("text")
    return text if isinstance(text, str) else ""


def safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def approx_token_count(text: str) -> int:
    return max(1, len(text.split()))


def http_json(
    method: str,
    url: str,
    headers: Dict[str, str],
    timeout_s: int,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise BenchmarkError(f"{method} {url} failed with HTTP {exc.code}: {detail[:500]}") from exc
    except urllib.error.URLError as exc:
        raise BenchmarkError(f"{method} {url} failed: {exc.reason}") from exc

    if not raw.strip():
        return {}

    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"{method} {url} returned invalid JSON: {raw[:300]!r}") from exc

    if not isinstance(decoded, dict):
        raise BenchmarkError(f"{method} {url} returned a non-object JSON payload.")
    return decoded


def fetch_available_models(base_url: str, headers: Dict[str, str], timeout_s: int) -> List[str]:
    payload = http_json("GET", f"{base_url}/v1/models", headers=headers, timeout_s=timeout_s)
    models = []
    for item in payload.get("data", []):
        if isinstance(item, dict):
            model_id = item.get("id")
            if isinstance(model_id, str) and model_id:
                models.append(model_id)
    return models


def wait_for_server_ready(
    base_url: str,
    headers: Dict[str, str],
    timeout_s: int,
    process: Optional[subprocess.Popen[str]],
) -> List[str]:
    deadline = time.time() + timeout_s
    last_error = "server is not ready yet"
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise BenchmarkError(f"Server exited before becoming ready with code {process.returncode}.")
        try:
            return fetch_available_models(base_url, headers=headers, timeout_s=10)
        except Exception as exc:
            last_error = str(exc)
        time.sleep(2)
    raise BenchmarkError(f"Timed out waiting for {base_url}/v1/models: {last_error}")


def load_prompt_file(path: pathlib.Path) -> List[str]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    if suffix == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            payload = payload.get("prompts")
        if not isinstance(payload, list):
            raise BenchmarkError(f"{path} JSON must be a list of strings or an object with a 'prompts' list.")
        prompts = [item for item in payload if isinstance(item, str) and item.strip()]
    elif suffix == ".jsonl":
        prompts = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, str):
                prompts.append(payload)
                continue
            if isinstance(payload, dict):
                for key in ("prompt", "text", "input"):
                    value = payload.get(key)
                    if isinstance(value, str) and value.strip():
                        prompts.append(value)
                        break
    else:
        prompts = [line.strip() for line in text.splitlines() if line.strip()]

    if not prompts:
        raise BenchmarkError(f"No prompts found in {path}.")
    return prompts


def expand_prompts(source_prompts: Sequence[str], total: int, seed: int) -> List[str]:
    if len(source_prompts) >= total:
        return list(source_prompts[:total])
    rng = random.Random(seed)
    return [rng.choice(source_prompts) for _ in range(total)]


def build_repeated_context(target_tokens: int, rng: random.Random, sentence_pool: Sequence[str]) -> str:
    pieces: List[str] = []
    words = 0
    while words < target_tokens:
        sentence = rng.choice(sentence_pool)
        pieces.append(sentence)
        words += len(sentence.split())
    return " ".join(pieces)


def build_mixed_prompts(total: int, approx_prompt_tokens: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    prompts: List[str] = []
    for idx in range(total):
        task = PROMPT_TASKS[idx % len(PROMPT_TASKS)]
        question = PROMPT_QUESTIONS[idx % len(PROMPT_QUESTIONS)]
        prefix = (
            "You are participating in an inference throughput benchmark. "
            "Follow the instruction carefully and answer in plain text.\n\n"
            f"Task: {task}\n\n"
        )
        fixed_words = len(prefix.split()) + len(question.split()) + 12
        context_budget = max(48, approx_prompt_tokens - fixed_words)
        context = build_repeated_context(context_budget, rng, CORPUS_SENTENCES)
        prompt = (
            f"{prefix}"
            f"Document:\n{context}\n\n"
            f"Question: {question}\n"
            "Response:"
        )
        prompts.append(prompt)
    return prompts


def build_shared_prefix_prompts(total: int, approx_prompt_tokens: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    base_prefix = (
        "You are participating in an inference throughput benchmark. "
        "Use only the shared context to answer the question.\n\n"
    )
    fixed_words = len(base_prefix.split()) + 18
    shared_budget = max(64, approx_prompt_tokens - fixed_words)
    shared_context = build_repeated_context(shared_budget, rng, SHARED_PREFIX_SENTENCES + CORPUS_SENTENCES)

    prompts: List[str] = []
    for idx in range(total):
        question = PROMPT_QUESTIONS[idx % len(PROMPT_QUESTIONS)]
        prompts.append(
            f"{base_prefix}"
            f"Shared context:\n{shared_context}\n\n"
            f"Request {idx + 1}: {question}\n"
            "Response:"
        )
    return prompts


def prepare_prompts(args: argparse.Namespace) -> List[str]:
    total_needed = args.warmup_requests + max(args.num_requests, args.auto_tune_requests, 1)
    if args.prompts_file:
        prompt_path = pathlib.Path(args.prompts_file).expanduser().resolve()
        return expand_prompts(load_prompt_file(prompt_path), total_needed, args.seed)

    if args.dataset == "shared-prefix":
        return build_shared_prefix_prompts(total_needed, args.approx_prompt_tokens, args.seed)
    return build_mixed_prompts(total_needed, args.approx_prompt_tokens, args.seed)


def percentile(values: Sequence[float], pct: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * (pct / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (rank - low)


def tail_text(path: pathlib.Path, max_lines: int = 40) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def launch_server(
    args: argparse.Namespace,
    served_model_name: str,
    tensor_parallel: int,
    pipeline_parallel: int,
    result_dir: pathlib.Path,
) -> tuple[subprocess.Popen[str], pathlib.Path, List[str]]:
    if args.backend == "vllm":
        cli = shutil.which("vllm")
        if cli:
            command = [
                cli,
                "serve",
                args.model,
                "--host",
                args.host,
                "--port",
                str(args.port),
                "--served-model-name",
                served_model_name,
                "--tensor-parallel-size",
                str(tensor_parallel),
                "--gpu-memory-utilization",
                str(args.gpu_memory_utilization),
            ]
            if pipeline_parallel > 1:
                command.extend(["--pipeline-parallel-size", str(pipeline_parallel)])
            if args.max_model_len is not None:
                command.extend(["--max-model-len", str(args.max_model_len)])
        else:
            command = [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                args.model,
                "--host",
                args.host,
                "--port",
                str(args.port),
                "--served-model-name",
                served_model_name,
                "--tensor-parallel-size",
                str(tensor_parallel),
                "--gpu-memory-utilization",
                str(args.gpu_memory_utilization),
            ]
            if pipeline_parallel > 1:
                command.extend(["--pipeline-parallel-size", str(pipeline_parallel)])
            if args.max_model_len is not None:
                command.extend(["--max-model-len", str(args.max_model_len)])
    else:
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            args.model,
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--served-model-name",
            served_model_name,
            "--tensor-parallel-size",
            str(tensor_parallel),
            "--mem-fraction-static",
            str(args.mem_fraction_static),
        ]
        if pipeline_parallel > 1:
            command.extend(["--pipeline-parallel-size", str(pipeline_parallel)])
        if args.max_model_len is not None:
            command.extend(["--context-length", str(args.max_model_len)])

    if args.api_key:
        command.extend(["--api-key", args.api_key])

    if args.trust_remote_code:
        command.append("--trust-remote-code")

    if args.server_extra_args:
        command.extend(shlex.split(args.server_extra_args))

    log_path = result_dir / f"{args.backend}_server.log"
    log_handle = log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    try:
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )
    except Exception:
        log_handle.close()
        raise
    log_handle.close()
    return process, log_path, command


def stop_server(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait(timeout=10)


def send_completion_request(
    base_url: str,
    model_name: str,
    prompt: str,
    output_tokens: int,
    temperature: float,
    top_p: float,
    timeout_s: int,
    headers: Dict[str, str],
    extra_request_body: Dict[str, Any],
) -> RequestResult:
    payload = dict(extra_request_body)
    payload.update(
        {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": output_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
    )

    started = time.perf_counter()
    try:
        response = http_json(
            "POST",
            f"{base_url}/v1/completions",
            headers=headers,
            timeout_s=timeout_s,
            payload=payload,
        )
        latency = time.perf_counter() - started
        usage = response.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        prompt_tokens = safe_int(usage.get("prompt_tokens"))
        completion_tokens = safe_int(usage.get("completion_tokens"))
        output_text = extract_output_text(response)
        if prompt_tokens < 1:
            prompt_tokens = approx_token_count(prompt)
        if completion_tokens < 1 and output_text:
            completion_tokens = approx_token_count(output_text)
        return RequestResult(
            ok=True,
            latency_s=latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            output_chars=len(output_text),
        )
    except Exception as exc:
        latency = time.perf_counter() - started
        return RequestResult(
            ok=False,
            latency_s=latency,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            output_chars=0,
            error=str(exc),
        )


def summarize_round(
    name: str,
    requested_requests: int,
    requested_concurrency: int,
    wall_time_s: float,
    results: Sequence[RequestResult],
) -> Dict[str, Any]:
    successes = [result for result in results if result.ok]
    failures = [result for result in results if not result.ok]
    latencies_ms = sorted(result.latency_s * 1000.0 for result in successes)
    prompt_tokens = sum(result.prompt_tokens for result in successes)
    completion_tokens = sum(result.completion_tokens for result in successes)
    total_tokens = sum(result.total_tokens for result in successes)
    output_chars = sum(result.output_chars for result in successes)

    error_examples: List[str] = []
    for result in failures:
        if result.error and result.error not in error_examples:
            error_examples.append(result.error)
        if len(error_examples) >= 5:
            break

    return {
        "name": name,
        "requested_requests": requested_requests,
        "requested_concurrency": requested_concurrency,
        "effective_workers": min(requested_concurrency, max(1, requested_requests)),
        "wall_time_s": round(wall_time_s, 6),
        "completed_requests": len(successes),
        "failed_requests": len(failures),
        "success_rate": round(len(successes) / max(1, requested_requests), 6),
        "request_throughput_rps": round(len(successes) / wall_time_s, 6) if wall_time_s > 0 else 0.0,
        "prompt_token_throughput_tps": round(prompt_tokens / wall_time_s, 6) if wall_time_s > 0 else 0.0,
        "completion_token_throughput_tps": round(completion_tokens / wall_time_s, 6) if wall_time_s > 0 else 0.0,
        "total_token_throughput_tps": round(total_tokens / wall_time_s, 6) if wall_time_s > 0 else 0.0,
        "total_prompt_tokens": prompt_tokens,
        "total_completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "total_output_chars": output_chars,
        "latency_ms": {
            "mean": round(statistics.fmean(latencies_ms), 6) if latencies_ms else None,
            "p50": round(percentile(latencies_ms, 50) or 0.0, 6) if latencies_ms else None,
            "p90": round(percentile(latencies_ms, 90) or 0.0, 6) if latencies_ms else None,
            "p95": round(percentile(latencies_ms, 95) or 0.0, 6) if latencies_ms else None,
            "p99": round(percentile(latencies_ms, 99) or 0.0, 6) if latencies_ms else None,
            "max": round(max(latencies_ms), 6) if latencies_ms else None,
        },
        "error_examples": error_examples,
    }


def run_round(
    name: str,
    prompts: Sequence[str],
    max_concurrency: int,
    base_url: str,
    model_name: str,
    args: argparse.Namespace,
    headers: Dict[str, str],
    extra_request_body: Dict[str, Any],
) -> Dict[str, Any]:
    print(
        f"[benchmark] {name}: requests={len(prompts)}, max_concurrency={max_concurrency}, "
        f"prompt_tokens~={args.approx_prompt_tokens}, output_tokens={args.output_tokens}",
        flush=True,
    )
    started = time.perf_counter()
    results: List[RequestResult] = []
    workers = min(max_concurrency, len(prompts))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [
            executor.submit(
                send_completion_request,
                base_url,
                model_name,
                prompt,
                args.output_tokens,
                args.temperature,
                args.top_p,
                args.request_timeout,
                headers,
                extra_request_body,
            )
            for prompt in prompts
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    summary = summarize_round(
        name=name,
        requested_requests=len(prompts),
        requested_concurrency=max_concurrency,
        wall_time_s=time.perf_counter() - started,
        results=results,
    )
    print(
        f"[benchmark] {name} done: completed={summary['completed_requests']}/{summary['requested_requests']}, "
        f"rps={summary['request_throughput_rps']:.2f}, "
        f"tok/s={summary['total_token_throughput_tps']:.2f}, "
        f"p95={summary['latency_ms']['p95']}",
        flush=True,
    )
    return summary


def auto_tune_concurrency(
    prompts: Sequence[str],
    base_url: str,
    model_name: str,
    args: argparse.Namespace,
    headers: Dict[str, str],
    extra_request_body: Dict[str, Any],
) -> Dict[str, Any]:
    tune_prompts = list(prompts[: max(1, args.auto_tune_requests)])
    candidates = parse_candidate_list(args.auto_concurrency_candidates, limit=len(tune_prompts))
    rounds: List[Dict[str, Any]] = []

    print(f"[auto] tuning client concurrency with candidates={candidates}", flush=True)
    for candidate in candidates:
        sample_count = min(len(tune_prompts), max(candidate * 2, 8))
        sample_prompts = tune_prompts[:sample_count]
        summary = run_round(
            name=f"auto_tune_c{candidate}",
            prompts=sample_prompts,
            max_concurrency=candidate,
            base_url=base_url,
            model_name=model_name,
            args=args,
            headers=headers,
            extra_request_body=extra_request_body,
        )
        rounds.append(summary)

    successful = [round_info for round_info in rounds if round_info["completed_requests"] > 0]
    if not successful:
        raise BenchmarkError("Auto concurrency tuning failed because every candidate produced zero successful requests.")

    best = max(
        successful,
        key=lambda round_info: (
            round_info["total_token_throughput_tps"],
            round_info["request_throughput_rps"],
            -round_info["requested_concurrency"],
        ),
    )
    chosen = int(best["requested_concurrency"])
    print(
        f"[auto] selected max_concurrency={chosen} "
        f"(tok/s={best['total_token_throughput_tps']:.2f}, rps={best['request_throughput_rps']:.2f})",
        flush=True,
    )
    return {
        "enabled": True,
        "chosen_max_concurrency": chosen,
        "rounds": rounds,
    }


def resolve_model_name(
    launched_name: str,
    discovered_models: Sequence[str],
    user_value: str,
) -> str:
    if user_value:
        return user_value
    if launched_name:
        return launched_name
    if discovered_models:
        return discovered_models[0]
    raise BenchmarkError("Unable to determine the model name to send in /v1/completions.")


def format_command(command: Sequence[str]) -> str:
    return shlex.join(list(command))


def main() -> int:
    args = parse_args()
    extra_request_body = parse_json_object(args.extra_request_body, "--extra-request-body")

    gpu_count = detect_gpu_count()
    tensor_parallel = parse_positive_int_or_auto(args.tensor_parallel, "--tensor-parallel", gpu_count)
    pipeline_parallel = parse_positive_int_or_auto(args.pipeline_parallel, "--pipeline-parallel", 1)
    served_model_name = args.served_model_name.strip() or default_served_model_name(args.model)
    result_dir = resolve_results_dir(args, served_model_name)
    base_url = f"http://{args.host}:{args.port}"
    headers = build_headers(args.api_key)
    prompts = prepare_prompts(args)
    warmup_prompts = prompts[: args.warmup_requests]
    benchmark_prompts = prompts[args.warmup_requests : args.warmup_requests + args.num_requests]
    auto_prompts = prompts[args.warmup_requests :]

    process: Optional[subprocess.Popen[str]] = None
    server_log_path: Optional[pathlib.Path] = None
    launch_command: List[str] = []
    warmup_summary: Optional[Dict[str, Any]] = None
    auto_tune_summary: Optional[Dict[str, Any]] = None

    try:
        if args.skip_launch:
            print(f"[server] using existing {args.backend} server at {base_url}", flush=True)
        else:
            print(
                f"[server] launching {args.backend} server at {base_url} "
                f"with tp={tensor_parallel}, pp={pipeline_parallel}",
                flush=True,
            )
            process, server_log_path, launch_command = launch_server(
                args=args,
                served_model_name=served_model_name,
                tensor_parallel=tensor_parallel,
                pipeline_parallel=pipeline_parallel,
                result_dir=result_dir,
            )
            print(f"[server] launch command: {format_command(launch_command)}", flush=True)

        discovered_models = wait_for_server_ready(
            base_url=base_url,
            headers=headers,
            timeout_s=args.startup_timeout,
            process=process,
        )
        launched_name_for_client = served_model_name if not args.skip_launch or not discovered_models else ""
        client_model_name = resolve_model_name(
            launched_name=launched_name_for_client,
            discovered_models=discovered_models,
            user_value=args.served_model_name.strip(),
        )
        print(
            f"[server] ready. discovered_models={discovered_models or [client_model_name]} "
            f"client_model={client_model_name}",
            flush=True,
        )

        if warmup_prompts:
            warmup_summary = run_round(
                name="warmup",
                prompts=warmup_prompts,
                max_concurrency=min(4, max(1, len(warmup_prompts))),
                base_url=base_url,
                model_name=client_model_name,
                args=args,
                headers=headers,
                extra_request_body=extra_request_body,
            )

        if str(args.max_concurrency).strip().lower() == "auto":
            auto_tune_summary = auto_tune_concurrency(
                prompts=auto_prompts,
                base_url=base_url,
                model_name=client_model_name,
                args=args,
                headers=headers,
                extra_request_body=extra_request_body,
            )
            max_concurrency = int(auto_tune_summary["chosen_max_concurrency"])
        else:
            max_concurrency = parse_positive_int_or_auto(args.max_concurrency, "--max-concurrency", 1)

        benchmark_summary = run_round(
            name="benchmark",
            prompts=benchmark_prompts,
            max_concurrency=max_concurrency,
            base_url=base_url,
            model_name=client_model_name,
            args=args,
            headers=headers,
            extra_request_body=extra_request_body,
        )

        results_path = result_dir / "benchmark_results.json"
        result_payload = {
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "backend": args.backend,
            "base_url": base_url,
            "model": args.model,
            "served_model_name": served_model_name,
            "client_model_name": client_model_name,
            "available_models": discovered_models,
            "gpu_count_detected": gpu_count,
            "launch": {
                "skip_launch": args.skip_launch,
                "launch_command": launch_command,
                "launch_command_str": format_command(launch_command) if launch_command else "",
                "server_log_path": str(server_log_path) if server_log_path else "",
                "keep_server": args.keep_server,
                "tensor_parallel": tensor_parallel,
                "pipeline_parallel": pipeline_parallel,
            },
            "prompt_source": {
                "prompts_file": args.prompts_file,
                "dataset": args.dataset,
                "approx_prompt_tokens": args.approx_prompt_tokens,
                "output_tokens": args.output_tokens,
                "seed": args.seed,
            },
            "request_config": {
                "num_requests": args.num_requests,
                "warmup_requests": args.warmup_requests,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "request_timeout": args.request_timeout,
                "extra_request_body": extra_request_body,
            },
            "auto_tuning": auto_tune_summary,
            "warmup": warmup_summary,
            "benchmark": benchmark_summary,
        }
        results_path.write_text(json.dumps(result_payload, indent=2, sort_keys=True), encoding="utf-8")

        print(f"[output] results written to {results_path}", flush=True)
        if server_log_path is not None:
            print(f"[output] server log written to {server_log_path}", flush=True)
        print(
            "[summary] "
            f"completed={benchmark_summary['completed_requests']}/{benchmark_summary['requested_requests']} "
            f"rps={benchmark_summary['request_throughput_rps']:.2f} "
            f"tok/s={benchmark_summary['total_token_throughput_tps']:.2f} "
            f"p95_ms={benchmark_summary['latency_ms']['p95']}",
            flush=True,
        )
        return 0
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr, flush=True)
        if server_log_path is not None:
            log_tail = tail_text(server_log_path)
            if log_tail:
                print("[error] last server log lines:", file=sys.stderr, flush=True)
                print(log_tail, file=sys.stderr, flush=True)
        return 1
    finally:
        if process is not None and not args.keep_server:
            stop_server(process)


if __name__ == "__main__":
    raise SystemExit(main())
