"""Visualize a Hugging Face translation dataset.

Streams a dataset (default: WMT14 de-en, matching download.py) and prints an
overview of its splits/features, a few random sample pairs, and a summary of
the sentence-length distribution. If matplotlib is available it also saves a
length histogram to PNG.

Streaming is used so this stays fast and memory-light even on the multi-million
row WMT corpora -- only the first --max-rows examples are scanned.

Usage:
    python visualize.py
    python visualize.py --dataset wmt/wmt14 --config de-en --split train
    python visualize.py --num-samples 10 --max-rows 50000
"""

import argparse
import random
import statistics

from datasets import load_dataset, load_dataset_builder


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default="wmt/wmt14", help="HF dataset id")
    p.add_argument("--config", default="de-en", help="config / subset name")
    p.add_argument("--split", default="train", help="split to inspect")
    p.add_argument("--num-samples", type=int, default=5,
                   help="number of random example pairs to print")
    p.add_argument("--max-rows", type=int, default=20000,
                   help="cap rows scanned for length stats (keeps it fast)")
    p.add_argument("--save-plot", default="lengths.png",
                   help="where to save the length histogram (PNG)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def text_extractors(features, sample_row):
    """Return [(label, fn(example) -> str)] for the text fields in a row.

    Translation datasets nest languages under a single ``translation`` dict
    (e.g. {"translation": {"de": ..., "en": ...}}); everything else is treated
    as a flat collection of string columns.
    """
    if "translation" in features:
        langs = getattr(features["translation"], "languages", None)
        if not langs:
            langs = sorted(sample_row["translation"].keys())
        return [(lang, lambda ex, l=lang: ex["translation"][l]) for lang in langs]

    extractors = []
    for name, feat in features.items():
        if getattr(feat, "dtype", None) == "string":
            extractors.append((name, lambda ex, n=name: ex[n]))
    return extractors


def print_overview(dataset, config, info):
    print("=" * 70)
    print(f"Dataset: {dataset}   config: {config}")
    print("=" * 70)

    print("Splits:")
    for name, split_info in (info.splits or {}).items():
        print(f"  {name:<12}{split_info.num_examples:>14,} examples")

    print("\nFeatures:")
    for name, feat in info.features.items():
        print(f"  {name}: {feat}")


def print_samples(samples, extractors):
    print(f"\nRandom samples ({len(samples)}):")
    for k, ex in enumerate(samples, 1):
        print(f"\n[{k}]")
        for label, fn in extractors:
            text = fn(ex)
            if len(text) > 200:
                text = text[:200] + " ..."
            print(f"  {label:>4}: {text}")


def print_length_stats(lengths):
    # Whitespace-token counts -- a fine proxy for de/en. For Japanese you'd
    # want character or subword counts instead.
    print("\nSentence length (whitespace tokens):")
    print(f"  {'field':<6}{'min':>6}{'median':>8}{'mean':>8}{'p95':>6}{'max':>7}")
    for label, vals in lengths.items():
        ordered = sorted(vals)
        p95 = ordered[int(0.95 * (len(ordered) - 1))]
        print(f"  {label:<6}{min(vals):>6}{statistics.median(vals):>8.0f}"
              f"{statistics.mean(vals):>8.1f}{p95:>6}{max(vals):>7}")


def plot_lengths(lengths, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed -- skipping plot)")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, vals in lengths.items():
        ax.hist(vals, bins=50, alpha=0.5, label=label)
    ax.set_xlabel("sentence length (whitespace tokens)")
    ax.set_ylabel("count")
    ax.set_title("Sentence-length distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\nSaved length histogram -> {path}")


def main():
    args = parse_args()
    random.seed(args.seed)

    # Split sizes / features come from dataset metadata -- no row download.
    info = load_dataset_builder(args.dataset, args.config).info
    print_overview(args.dataset, args.config, info)

    stream = load_dataset(args.dataset, args.config,
                          split=args.split, streaming=True)

    first = next(iter(stream))  # peek to detect the text structure
    extractors = text_extractors(info.features, first)

    # Single streaming pass: reservoir-sample a few rows + collect lengths.
    lengths = {label: [] for label, _ in extractors}
    samples = []
    scanned = 0
    for i, ex in enumerate(stream):
        if i >= args.max_rows:
            break
        scanned = i + 1
        for label, fn in extractors:
            lengths[label].append(len(fn(ex).split()))
        if len(samples) < args.num_samples:
            samples.append(ex)
        elif random.randint(0, i) < args.num_samples:
            samples[random.randint(0, args.num_samples - 1)] = ex

    print(f"\nScanned {scanned:,} rows of split '{args.split}' for stats.")
    print_samples(samples, extractors)
    print_length_stats(lengths)
    plot_lengths(lengths, args.save_plot)

if __name__ == "__main__":
    main()
