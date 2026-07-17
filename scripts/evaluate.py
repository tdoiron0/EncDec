import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from sacrebleu.metrics import BLEU, CHRF

from src.model.enc_dec import EncoderDecoder
from src.tokenizer.tokenizer import Tokenizer
from constants import PROJECT_ROOT, PAD_TOKEN
from index import DATASET_INDEX

from utils import get_best_device, load_config

device = get_best_device()

def make_console_utf8_safe() -> None:
    """Force stdout/stderr to UTF-8.

    The evaluation prints Japanese source text. On a default Windows console
    (cp1252) that raises UnicodeEncodeError, so re-encode the streams up front.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def resolve_checkpoint_path(run_dir: str, checkpoint_name: str = None) -> str:
    """Locate a checkpoint inside a run directory saved by scripts/train.py.

    The run's train_config.yaml names the checkpoint subdirectory. Without an
    explicit checkpoint name, the latest checkpoint wins — files are named
    ``checkpoint_epoch_{E}_iter_{I}.pt``, so latest means highest (epoch, iter).
    """
    train_config = load_config(os.path.join(run_dir, "train_config.yaml"))
    checkpoint_dir = os.path.join(run_dir, train_config.checkpoint_dir)

    if checkpoint_name:
        return os.path.join(checkpoint_dir, checkpoint_name)

    pattern = re.compile(r"checkpoint_epoch_(\d+)_iter_(\d+)\.pt$")
    candidates = [
        (int(m.group(1)), int(m.group(2)), f.path)
        for f in os.scandir(checkpoint_dir)
        if (m := pattern.match(f.name))
    ]
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return max(candidates)[2]


def load_checkpoint(checkpoint_path: str) -> dict:
    """Load a training checkpoint saved by scripts/train.py."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    iter_num = checkpoint.get("iter_num", "unknown")
    loss = checkpoint.get("loss", "unknown")
    logger.info(f"Loaded checkpoint from iteration {iter_num} with train loss {loss}")
    return checkpoint


def build_model_from_checkpoint(checkpoint: dict) -> torch.nn.Module:
    """Rebuild the EncoderDecoder from the ModelConfig stored at save time.

    train.py persists the exact ModelConfig used to construct the model, so the
    architecture is rebuilt directly rather than inferred from tensor shapes.
    """
    logger = logging.getLogger(__name__)
    if "config" not in checkpoint:
        raise KeyError(
            "Checkpoint has no 'model_config'; it was not saved by the current "
            "EncoderDecoder train.py and cannot be rebuilt."
        )
    model_config = checkpoint["config"]

    model = EncoderDecoder(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Built EncoderDecoder: {model_config.n_layer} layers/stack, "
        f"{model_config.n_embd} embd dim, {model_config.n_head} heads, "
        f"vocab {model_config.vocab_size}, block {model_config.block_size} "
        f"({param_count:,} params)"
    )
    return model


def load_tokenizer(run_dir: str) -> Tokenizer:
    """Load the tokenizer saved alongside the run."""
    logger = logging.getLogger(__name__)
    model_path = os.path.join(run_dir, "tokenizer.model")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Tokenizer model not found: {model_path}")
    logger.info(f"Loading tokenizer from {model_path}")
    return Tokenizer.load(model_path)


def _strip_padding(seq: torch.Tensor) -> List[int]:
    """Drop the trailing PAD the Dataset appends to reach block_size.

    PAD is the tokenizer's reserved pad id, so it never appears inside real
    content — only as right-padding — and removing it recovers the original
    unpadded sequence.
    """
    ids = seq.tolist()
    while ids and ids[-1] == PAD_TOKEN:
        ids.pop()
    return ids


def load_eval_pairs(
    dataset_name: str, split: str, block_size: int
) -> List[Tuple[List[int], List[int]]]:
    """Load the raw (src_ids, tgt_ids) pairs for a split via DATASET_INDEX.

    The dataset registered for this checkpoint owns how splits are read from
    disk, so it is instantiated through DATASET_INDEX rather than reloading the
    .pt by hand. It returns src/tgt padded to block_size; evaluation needs the
    source and target kept apart and unpadded — the source is the generation
    prompt and the target is the reference — so the padding is stripped here.
    """
    logger = logging.getLogger(__name__)
    if dataset_name not in DATASET_INDEX:
        raise KeyError(
            f"Dataset '{dataset_name}' not in DATASET_INDEX; cannot load splits."
        )
    dataset = DATASET_INDEX[dataset_name](split, block_size)
    pairs = [
        (_strip_padding(src), _strip_padding(tgt))
        for src, tgt in (dataset[i] for i in range(len(dataset)))
    ]
    logger.info(f"Loaded {len(pairs)} pairs from '{split}' split")
    return pairs


def decode_reference(sp: Tokenizer, tgt: List[int], eos_id: int) -> str:
    """Decode a target sequence to text, dropping the framing BOS/EOS.

    Targets are stored as ``[BOS] + encode(en) + [EOS]``; the hypothesis from
    generation carries neither, so both ends are stripped before decoding to keep
    references and hypotheses comparable for BLEU / chrF / exact match.
    """
    ids = tgt
    if ids and ids[0] == sp.bos_id():
        ids = ids[1:]
    if ids and ids[-1] == eos_id:
        ids = ids[:-1]
    return sp.decode(ids).strip()


@torch.no_grad()
def translate(
    model: EncoderDecoder,
    sp: Tokenizer,
    src_ids: List[int],
    max_new_tokens: int,
    eos_id: int,
    temperature: float,
) -> str:
    """Generate an English hypothesis for one tokenized Japanese source.

    The EncoderDecoder encodes the source and decodes the target autoregressively
    starting from BOS; ``model.generate`` returns the target ids with the leading
    BOS and terminating EOS already stripped.
    """
    idx = torch.tensor([src_ids], dtype=torch.long, device=device)
    gen_ids = model.generate(idx, max_new_tokens, eos_id=eos_id, temperature=temperature)
    return sp.decode(gen_ids).strip()


@torch.no_grad()
def teacher_forced_loss(
    model: EncoderDecoder, src: List[int], tgt: List[int], block_size: int
) -> Tuple[float, int]:
    """Teacher-forced cross-entropy on the target, summed and token-weighted.

    The EncoderDecoder computes the loss internally (averaged over the supervised
    target tokens with ignore_index=PAD). A single unpadded target of length L is
    supervised on L-1 positions — the decoder input drops the final token — so the
    averaged loss is reweighted by that count for a corpus-level perplexity.
    """
    src = src[:block_size]
    tgt = tgt[: block_size + 1]
    n_tokens = max(len(tgt) - 1, 0)
    if n_tokens == 0:
        return 0.0, 0

    src_t = torch.tensor([src], dtype=torch.long, device=device)
    tgt_t = torch.tensor([tgt], dtype=torch.long, device=device)
    _, loss = model(src_t, tgt_t)  # cross_entropy averages over the n_tokens
    return loss.item() * n_tokens, n_tokens


def evaluate_model(
    model: torch.nn.Module,
    pairs: List[Tuple[List[int], List[int]]],
    sp: Tokenizer,
    eos_id: int,
    max_new_tokens: int,
    temperature: float,
    log_interval: int,
) -> Dict[str, float]:
    """Run the full evaluation: perplexity + generation-based metrics."""
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating {len(pairs)} samples (greedy={temperature == 0})...")

    hypotheses: List[str] = []
    references: List[str] = []
    exact = 0
    total_loss = 0.0
    total_tokens = 0

    start = time.time()
    for i, (src, tgt) in enumerate(pairs, start=1):
        # Teacher-forced loss (the training objective) -> perplexity.
        loss_sum, n_tok = teacher_forced_loss(model, src, tgt, model.block_size)
        total_loss += loss_sum
        total_tokens += n_tok

        # Free-running generation -> BLEU / chrF / exact match.
        reference = decode_reference(sp, tgt, eos_id)
        hypothesis = translate(model, sp, src, max_new_tokens, eos_id, temperature)

        references.append(reference)
        hypotheses.append(hypothesis)
        exact += int(hypothesis == reference)

        if log_interval > 0 and i % log_interval == 0:
            rate = i / (time.time() - start)
            logger.info(f"  {i}/{len(pairs)} ({rate:.1f} samples/sec)")

    bleu = BLEU().corpus_score(hypotheses, [references]).score
    chrf = CHRF().corpus_score(hypotheses, [references]).score
    exact_match = 100.0 * exact / len(pairs)
    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = float(torch.exp(torch.tensor(avg_loss)))

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "bleu": bleu,
        "chrf": chrf,
        "exact_match": exact_match,
        "total_samples": len(pairs),
    }


def show_samples(
    model: torch.nn.Module,
    pairs: List[Tuple[List[int], List[int]]],
    sp: Tokenizer,
    eos_id: int,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
) -> None:
    """Print a handful of source / reference / hypothesis triples."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info(f"SAMPLE TRANSLATIONS (first {num_samples})")
    logger.info("=" * 60)

    for i, (src, tgt) in enumerate(pairs[:num_samples], start=1):
        src_ids = src[1:-1] if len(src) >= 2 else src  # drop BOS/EOS for display
        source = sp.decode(src_ids).strip()
        reference = decode_reference(sp, tgt, eos_id)
        hypothesis = translate(model, sp, src, max_new_tokens, eos_id, temperature)

        logger.info(f"[{i}] JA   : {source}")
        logger.info(f"    REF  : {reference}")
        logger.info(f"    HYP  : {hypothesis}")
        logger.info("-" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a Jap2Eng training run."
    )
    parser.add_argument("run_dir", type=str, help="Path to the run directory")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint filename within the run (default: latest epoch/iter)",
    )
    parser.add_argument(
        "--split", type=str, default="val", help="Eval split file stem (val/test)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 = greedy (default); >0 = sample",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=64, help="Max tokens to generate"
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=0,
        help="Cap on samples evaluated (0 = all)",
    )
    parser.add_argument(
        "--num_samples", type=int, default=5, help="Sample translations to print"
    )
    parser.add_argument(
        "--no_samples", action="store_true", help="Skip sample printing"
    )
    parser.add_argument(
        "--log_interval", type=int, default=200, help="Progress log interval"
    )
    args = parser.parse_args()

    make_console_utf8_safe()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting Jap2Eng evaluation")
    logger.info(f"Run directory: {args.run_dir}")
    logger.info(f"Device: {device} | temperature: {args.temperature}")

    try:
        run_dir = os.path.join(PROJECT_ROOT, args.run_dir)
        checkpoint_path = resolve_checkpoint_path(run_dir, args.checkpoint)
        checkpoint = load_checkpoint(checkpoint_path)
        model = build_model_from_checkpoint(checkpoint)
        dataset_name = checkpoint["config"].dataset
        sp = load_tokenizer(run_dir)
        eos_id = sp.eos_id()

        pairs = load_eval_pairs(dataset_name, args.split, model.block_size)
        if args.max_eval_samples and args.max_eval_samples < len(pairs):
            pairs = pairs[: args.max_eval_samples]
            logger.info(f"Capped evaluation to {len(pairs)} samples")

        if not args.no_samples:
            show_samples(
                model,
                pairs,
                sp,
                eos_id,
                args.num_samples,
                args.max_new_tokens,
                args.temperature,
            )

        metrics = evaluate_model(
            model,
            pairs,
            sp,
            eos_id,
            args.max_new_tokens,
            args.temperature,
            args.log_interval,
        )

        logger.info("=" * 60)
        logger.info("FINAL EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Split             : {args.split}")
        logger.info(f"Samples           : {metrics['total_samples']}")
        logger.info(f"Cross-entropy loss: {metrics['loss']:.4f}")
        logger.info(f"Perplexity        : {metrics['perplexity']:.2f}")
        logger.info(f"BLEU              : {metrics['bleu']:.2f}")
        logger.info(f"chrF              : {metrics['chrf']:.2f}")
        logger.info(f"Exact match       : {metrics['exact_match']:.2f}%")
        logger.info("=" * 60)
        logger.info("Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
