# Inference Capacity Lab

Find the offered load at which a vLLM deployment misses its latency objectives. The lab sweeps two prompt lengths and multiple request rates, saving vLLM's results and a CSV summary of completion, p99 time to first token, p99 time per output token and SLO goodput.

Install vLLM on a supported machine and start its OpenAI-compatible server with a model that fits your hardware. Use the same vLLM release for client and server.

```sh
python3 run.py --model YOUR_MODEL --results run-01 > summary.csv
```

Goodput is the number of requests per second that meet both configured latency objectives. Compare it against offered rate and review completed requests at each point. For comparable trials, hold model, hardware, server configuration and output length fixed, control prefix caching, and repeat with fresh run directories. Report actual hardware and server settings alongside any results. This repository ships no measured GPU results.
