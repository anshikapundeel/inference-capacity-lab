#!/usr/bin/env python3
"""Sweep vLLM offered request rates and report measured latency goodput."""
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--rates", type=float, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--input-tokens", type=int, nargs="+", default=[128, 1024])
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--prompts", type=int, default=200)
    parser.add_argument("--ttft-ms", type=float, default=500)
    parser.add_argument("--tpot-ms", type=float, default=50)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    if any(not math.isfinite(x) or x <= 0 for x in [*args.rates, args.ttft_ms, args.tpot_ms]):
        parser.error("rates and latency objectives must be positive and finite")
    if min(args.prompts, *args.input_tokens) < 1 or args.output_tokens < 2:
        parser.error("use positive prompt and input lengths, and at least two output tokens")
    args.results.mkdir(parents=True, exist_ok=False)
    writer = csv.writer(sys.stdout)
    writer.writerow(["input_tokens", "request_rate", "completed", "p99_ttft_ms", "p99_tpot_ms", "goodput_req_s"])
    for index, (length, rate) in enumerate(
            ((length, rate) for length in args.input_tokens for rate in args.rates), start=1):
        filename = f"point-{index}.json"
        command = [
            "vllm", "bench", "serve", "--backend", "vllm", "--model", args.model,
            "--base-url", args.base_url, "--dataset-name", "random",
            "--random-input-len", str(length), "--random-output-len", str(args.output_tokens),
            "--num-prompts", str(args.prompts), "--request-rate", str(rate),
            "--seed", str(index), "--percentile-metrics", "ttft,tpot", "--metric-percentiles", "99",
            "--goodput", f"ttft:{args.ttft_ms}", f"tpot:{args.tpot_ms}",
            "--save-result", "--result-dir", str(args.results), "--result-filename", filename,
        ]
        subprocess.run(command, check=True, stdout=sys.stderr)
        result = json.loads((args.results / filename).read_text())
        writer.writerow([length, rate, result["completed"], result["p99_ttft_ms"],
                         result["p99_tpot_ms"], result["request_goodput"]])
        sys.stdout.flush()


if __name__ == "__main__":
    main()
