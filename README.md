# Inference Capacity Lab

Find the offered load at which a vLLM deployment misses its latency objectives. The lab sweeps two prompt lengths and multiple request rates, saving vLLM's results, experiment parameters and a CSV summary of completed and failed requests, p99 time to first token, p99 time per output token and SLO goodput.

Install vLLM on a supported machine and start its OpenAI-compatible server with a model that fits your hardware. Use the same vLLM release for client and server.

```sh
python3 run.py --model YOUR_MODEL --results run-01 > summary.csv
```

The runner randomizes trial order reproducibly and repeats each point three times by default. It requests a fixed output length with `--ignore-eos` and uses a different synthetic prompt seed at each point. Disable prefix caching on the server if measuring uncached capacity.

Goodput is the number of requests per second that meet both configured latency objectives. Compare it against offered rate and review failed requests and the spread across repeats at each point. For comparable trials, hold model, hardware, server configuration and output length fixed, control prefix caching, and repeat with fresh run directories. Report actual hardware and server settings alongside any results. This repository ships no measured GPU results.

After every planned point finishes, capacity.md lists the passing repeats, observed goodput range and highest tested rate meeting both p99 thresholds in every repeat with all requests completed. The numbers apply to the tested grid and hardware.

## Run the case study on a GPU runner

The manual Inference capacity case study workflow uses a Linux NVIDIA runner labeled gpu and vllm-lab. Start the model server on that runner at port 8000 before dispatch and enter its served model ID. The job records GPU model, memory, driver and vLLM client version alongside the raw outputs. Keep the server version, startup flags and cache configuration with the experiment when sharing results.
