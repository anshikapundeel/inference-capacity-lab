#!/usr/bin/env python3
"""Sweep vLLM offered request rates and report measured latency goodput."""
import argparse
import csv
import json
import math
import random
from pathlib import Path
import subprocess
import sys


def trial_plan(lengths, rates, repeats, seed):
    rng = random.Random(seed)
    for repeat in range(1, repeats + 1):
        points = [(length, rate) for length in lengths for rate in rates]
        rng.shuffle(points)
        for length, rate in points:
            yield repeat, length, rate


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
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if any(not math.isfinite(x) or x <= 0 for x in [*args.rates, args.ttft_ms, args.tpot_ms]):
        parser.error("rates and latency objectives must be positive and finite")
    if min(args.repeats, args.prompts, *args.input_tokens) < 1 or args.output_tokens < 2:
        parser.error("use positive repeat counts, prompt counts and input lengths, and at least two output tokens")
    args.results.mkdir(parents=True, exist_ok=False)
    metadata = vars(args) | {"results": str(args.results)}
    (args.results / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    writer = csv.writer(sys.stdout)
    writer.writerow(["repeat", "input_tokens", "request_rate", "completed", "failed", "p99_ttft_ms", "p99_tpot_ms", "goodput_req_s"])
    trials = {}
    for index, (repeat, length, rate) in enumerate(
            trial_plan(args.input_tokens, args.rates, args.repeats, args.seed), start=1):
        filename = f"point-{index}.json"
        command = [
            "vllm", "bench", "serve", "--backend", "vllm", "--model", args.model,
            "--base-url", args.base_url, "--dataset-name", "random",
            "--random-input-len", str(length), "--random-output-len", str(args.output_tokens),
            "--num-prompts", str(args.prompts), "--request-rate", str(rate),
            "--seed", str(args.seed + index), "--ignore-eos", "--percentile-metrics", "ttft,tpot", "--metric-percentiles", "99",
            "--goodput", f"ttft:{args.ttft_ms}", f"tpot:{args.tpot_ms}",
            "--save-result", "--result-dir", str(args.results), "--result-filename", filename,
        ]
        subprocess.run(command, check=True, stdout=sys.stderr)
        result = json.loads((args.results / filename).read_text())
        writer.writerow([repeat, length, rate, result["completed"], result["failed"], result["p99_ttft_ms"],
                         result["p99_tpot_ms"], result["request_goodput"]])
        passed = (result["completed"] == args.prompts and result["failed"] == 0
                  and result["p99_ttft_ms"] <= args.ttft_ms and result["p99_tpot_ms"] <= args.tpot_ms)
        trials.setdefault((length, rate), []).append((passed, result["request_goodput"]))
        sys.stdout.flush()

    lines = ["# Inference capacity experiment", "",
             f"Model: {args.model}. Each passing trial completed all {args.prompts} requests "
             f"with p99 first-token latency ≤ {args.ttft_ms} ms and p99 time per output token ≤ {args.tpot_ms} ms.", "",
             "| Input tokens | Offered req/s | Passing trials | Goodput range (req/s) |",
             "|---:|---:|---:|---:|"]
    for (length, rate), points in sorted(trials.items()):
        goodput = [value for _, value in points]
        successes = sum(passed for passed, _ in points)
        lines.append(f"| {length} | {rate:g} | {successes}/{len(points)} | {min(goodput):.2f}–{max(goodput):.2f} |")
    for length in sorted(set(args.input_tokens)):
        passing = [rate for (tokens, rate), points in trials.items()
                   if tokens == length and all(passed for passed, _ in points)]
        bound = f"{max(passing):g} req/s" if passing else "no passing rate"
        lines.append(f"\nFor {length} input tokens, the highest tested offered rate passing every repeat was **{bound}**.")
    (args.results / "capacity.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
