"""
Cost every pipeline from MEASURED tokens, not from guesses.

Each extraction run logs its real prompt/completion token totals for the 16 gold
pages. This scales those to the two corpus sizes that matter and prices them at
each provider's list rate, so "corpus $" stops being a hand-wave:

  test        the 16 gold pages -- what was actually spent getting these numbers
  cached      2,192 pages, the LOC pages already downloaded and sitting in cache/
  continuous  ~23,000 pages, the projected 1900-1963 monthly scrape (P3)

Two-stage pipelines are priced as extraction + judging, each at its own model's
rate. Gemini batch mode halves both input and output, so a bulk corpus run costs
half the figures below; DeepInfra has no equivalent published discount, so its
numbers are not adjusted.
"""

import json
import re
from pathlib import Path

GOLD_PAGES = 16
CACHED_PAGES = 2192
CONTINUOUS_PAGES = 23000

# $ per million tokens (input, output).
# Gemini rates from ai.google.dev/gemini-api/docs/pricing (checked 2026-07-22):
# 3.5 Flash is 5x Flash-Lite, which is easy to miss and dominates the ranking.
PRICES = {
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "openai/gpt-oss-120b": (0.04, 0.17),
    "deepseek-ai/DeepSeek-V3.1-Terminus": (0.27, 0.95),
    "meta-llama/Llama-3.3-70B-Instruct": (0.23, 0.40),
    "Qwen/Qwen3-235B-A22B-Instruct-2507": (0.09, 0.55),
    "Qwen/Qwen3-Next-80B-A3B-Instruct": (0.09, 1.10),
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": (0.05, 0.10),
    "meta-llama/Meta-Llama-3.1-8B-Instruct": (0.02, 0.05),
    "regex": (0.0, 0.0),
}

# label -> (result file stem, [(model, log file), ...])
PIPELINES = [
    ("gemini-3.5-flash 1-window", "gem35flash_1chunk",
     [("gemini-3.5-flash", "log_gem1chunk.txt")]),
    ("gemini-3.5-flash 8k", "gemini35flash",
     [("gemini-3.5-flash", "log_gemini35flash.txt")]),
    ("gpt-oss-120b -> gem-flash verify", "oss120b_vgemflash",
     [("openai/gpt-oss-120b", "log_gptoss120b.txt"),
      ("gemini-3.5-flash", "log_v_gemflash.txt")]),
    ("gpt-oss-120b -> gem-lite verify", "oss120b_vgemlite",
     [("openai/gpt-oss-120b", "log_gptoss120b.txt"),
      ("gemini-3.5-flash-lite", "log_v_gemlite.txt")]),
    ("gpt-oss-120b -> gpt-oss verify", "oss120b_voss",
     [("openai/gpt-oss-120b", "log_gptoss120b.txt"),
      ("openai/gpt-oss-120b", "log_v_oss.txt")]),
    ("gpt-oss-120b", "gptoss120b", [("openai/gpt-oss-120b", "log_gptoss120b.txt")]),
    ("DeepSeek-V3.1-Terminus", "deepseekv3",
     [("deepseek-ai/DeepSeek-V3.1-Terminus", "log_deepseekv3.txt")]),
    ("Llama-3.3-70B", "llama70b",
     [("meta-llama/Llama-3.3-70B-Instruct", "log_llama70b.txt")]),
    ("gemini-3.5-flash-lite 1-window", "gem35flashlite_1chunk",
     [("gemini-3.5-flash-lite", "log_gemlite1chunk.txt")]),
    ("gemini-3.5-flash-lite 8k", "gemini35fl",
     [("gemini-3.5-flash-lite", "log_gemini35fl.txt")]),
    ("Qwen3-235B-A22B", "qwen3_235b",
     [("Qwen/Qwen3-235B-A22B-Instruct-2507", "log_qwen3_235b.txt")]),
    ("Qwen3-Next-80B-A3B", "qwen3next80b",
     [("Qwen/Qwen3-Next-80B-A3B-Instruct", "log_qwen3next80b.txt")]),
    ("Mistral-Small-3.1-24B", "mistral24b",
     [("mistralai/Mistral-Small-3.1-24B-Instruct-2503", "log_mistral24b.txt")]),
    ("Llama-3.1-8B", "llama8b",
     [("meta-llama/Meta-Llama-3.1-8B-Instruct", "log_llama8b.txt")]),
    ("regex (current pipeline)", "regex", []),
]

TOKENS_RE = re.compile(r"tokens:\s*([\d,]+)\s*in\s*/\s*([\d,]+)\s*out")


def tokens_from_log(path):
    text = Path("gold_extraction", path).read_text(encoding="utf-8", errors="replace")
    m = TOKENS_RE.search(text)
    if not m:
        return 0, 0
    return int(m.group(1).replace(",", "")), int(m.group(2).replace(",", ""))


def cost_for(stages, pages):
    total = 0.0
    for model, log in stages:
        tin, tout = tokens_from_log(log)
        pin, pout = PRICES[model]
        scale = pages / GOLD_PAGES
        total += (tin * scale / 1e6) * pin + (tout * scale / 1e6) * pout
    return total


def gemini_share(stages):
    return any(m.startswith("gemini") for m, _ in stages)


if __name__ == "__main__":
    rows = []
    for label, stem, stages in PIPELINES:
        try:
            r = json.load(open(f"gold_extraction/result_{stem}.json"))
            h = json.load(open(f"gold_extraction/hard_{stem}.json"))
        except FileNotFoundError:
            continue
        rows.append((r["f1"], label, r["precision"], r["recall"], h["overall"],
                     cost_for(stages, GOLD_PAGES),
                     cost_for(stages, CACHED_PAGES),
                     cost_for(stages, CONTINUOUS_PAGES),
                     gemini_share(stages)))
    rows.sort(reverse=True)

    print(f"{'pipeline':<34}{'prec':>6}{'rec':>6}{'F1':>7}{'hard':>6}"
          f"{'16pg':>8}{'2,192pg':>10}{'23,000pg':>11}")
    print("-" * 88)
    for f1, label, p, r, h, c_test, c_cached, c_cont, gem in rows:
        print(f"{label:<34}{p:>6.2f}{r:>6.2f}{f1:>7.3f}{h:>5.0%}"
              f"{c_test:>8.2f}{c_cached:>10.2f}{c_cont:>11.2f}")

    print("\nAll figures in USD, from measured token counts scaled linearly by page count.")
    print("Gemini batch mode is 50% off input and output -- halve any Gemini row for a")
    print("bulk corpus run. DeepInfra publishes no equivalent discount.")
    spent = sum(r[5] for r in rows)
    print(f"\nActually spent across every run in this bake-off: ~${spent:.2f}")
