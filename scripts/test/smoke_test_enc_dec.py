"""
Smoke test for the EncoderDecoder model.

This does NOT need the dataset, tokenizer, or training pipeline. It builds the
model from a tiny config and checks that the architecture is wired together
correctly:

  1. construction            - the module builds without error
  2. forward / output shape  - forward runs with DIFFERENT src/tgt lengths and
                               returns (B, tgt_len, vocab_size) logits
  3. causal masking          - changing a future target token does not leak into
                               earlier decoder positions (the mask is threaded
                               through and actually applied)
  4. gradient flow           - loss.backward() reaches every parameter, and in
                               particular the ENCODER receives gradient (proves
                               cross-attention connects the decoder loss back to
                               the encoder)
  5. overfit one batch       - a few hundred steps on a single fixed batch drives
                               the loss down (forward + backward + update compose
                               into real learning)

Run (from anywhere) with the CS-4644 conda env:
    conda run -n CS-4644 python Models/Jap2Eng/scripts/smoke_test_enc_dec.py
"""

import sys
import traceback
from pathlib import Path

# --- make `src`, `config`, ... importable regardless of cwd -------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Models/Jap2Eng
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from config.model_config import ModelConfig
from src.model.enc_dec import EncoderDecoder


# Small but non-trivial config. Dropout is set to 0 so the checks are
# deterministic and eval()/train() behave identically.
CONFIG = ModelConfig(
    vocab_size=32,
    block_size=16,
    n_layer=2,
    n_embd=16,
    n_head=4,
    hidden_pdrop=0.0,
    attn_pdrop=0.0,
)

B = 4          # batch size
SRC_LEN = 9    # source length  (deliberately != target length)
TGT_LEN = 7    # target length
DEVICE = "cpu"  # tiny model; CPU keeps the run deterministic and portable


def causal_mask(batch: int, seq_len: int) -> torch.Tensor:
    """Lower-triangular mask, shape (B, 1, T, T). 1 = keep, 0 = mask.

    Matches the convention in GenericSelfAttention: scores.masked_fill(mask == 0, -inf).
    """
    m = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.long, device=DEVICE))
    return m.view(1, 1, seq_len, seq_len).expand(batch, 1, seq_len, seq_len)


def make_batch():
    src = torch.randint(0, CONFIG.vocab_size, (B, SRC_LEN), device=DEVICE)
    tgt = torch.randint(0, CONFIG.vocab_size, (B, TGT_LEN), device=DEVICE)
    mask = causal_mask(B, TGT_LEN)
    return src, tgt, mask


# -----------------------------------------------------------------------------
# Individual checks. Each returns a human-readable detail string and raises
# AssertionError on failure.
# -----------------------------------------------------------------------------
def check_construction():
    model = EncoderDecoder(CONFIG).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    return model, f"built EncoderDecoder with {n_params:,} parameters"


def check_forward_shapes(model):
    model.eval()
    src, tgt, mask = make_batch()
    with torch.no_grad():
        out = model(src, tgt, mask)

    expected = (B, TGT_LEN, CONFIG.vocab_size)
    assert tuple(out.shape) == expected, (
        f"expected logits of shape {expected}, got {tuple(out.shape)}"
    )
    assert torch.isfinite(out).all(), "forward produced NaN/Inf logits"
    return (
        f"src={tuple(src.shape)}, tgt={tuple(tgt.shape)} -> logits={tuple(out.shape)} "
        f"(src_len {SRC_LEN} != tgt_len {TGT_LEN}, all finite)"
    )


def check_causal_masking(model):
    """Changing the LAST target token must not change outputs at earlier positions."""
    model.eval()
    src, tgt, mask = make_batch()
    with torch.no_grad():
        out1 = model(src, tgt, mask)

        tgt2 = tgt.clone()
        tgt2[:, -1] = (tgt2[:, -1] + 1) % CONFIG.vocab_size  # perturb only the final token
        out2 = model(src, tgt2, mask)

    prefix_unchanged = torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-5)
    last_changed = not torch.allclose(out1[:, -1], out2[:, -1], atol=1e-5)

    assert prefix_unchanged, (
        "future token leaked into earlier positions -> decoder self-attention is "
        "NOT causal (mask not applied / not threaded through)"
    )
    assert last_changed, (
        "perturbing the last token changed nothing at the last position -> the "
        "target tokens may not be influencing the output at all"
    )
    return "perturbing tgt[-1] left positions 0..T-2 unchanged and only moved position T-1"


def check_gradient_flow(model):
    model.train()
    model.zero_grad(set_to_none=True)
    src, tgt, mask = make_batch()

    logits = model(src, tgt, mask)
    loss = F.cross_entropy(logits.reshape(-1, CONFIG.vocab_size), tgt.reshape(-1))
    loss.backward()

    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"these parameters received no gradient: {missing}"

    nonfinite = [
        n for n, p in model.named_parameters()
        if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    assert not nonfinite, f"non-finite gradients in: {nonfinite}"

    # The encoder must receive gradient through the decoder's cross-attention,
    # otherwise the "encoder" contributes nothing to the loss.
    enc_grad = sum(
        p.grad.abs().sum().item()
        for n, p in model.named_parameters()
        if n.startswith("encoder.") and p.grad is not None
    )
    assert enc_grad > 0, (
        "encoder received zero gradient -> cross-attention is not connecting the "
        "decoder loss back to the encoder output"
    )
    return f"loss={loss.item():.4f}, all params have finite grad, encoder grad-norm-ish={enc_grad:.3e}"


def check_overfit_single_batch(model, steps=400, lr=3e-3):
    """A model that works should be able to memorize one fixed batch."""
    model.train()
    src, tgt, mask = make_batch()  # fixed batch reused every step
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    first = last = None
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        logits = model(src, tgt, mask)
        loss = F.cross_entropy(logits.reshape(-1, CONFIG.vocab_size), tgt.reshape(-1))
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()

    assert last < first * 0.25 and last < 1.0, (
        f"loss did not drop enough: {first:.4f} -> {last:.4f} over {steps} steps"
    )
    return f"loss {first:.4f} -> {last:.4f} over {steps} steps (memorized fixed batch)"


def main():
    torch.manual_seed(0)
    print(f"Device: {DEVICE}")
    print(f"Config: {CONFIG}\n")

    results = []

    # Construction is special: it produces the model the other checks need.
    try:
        model, detail = check_construction()
        results.append(("construction", True, detail))
    except Exception:
        results.append(("construction", False, traceback.format_exc()))
        _report(results)
        return 1

    checks = [
        ("forward / output shape", check_forward_shapes),
        ("causal masking", check_causal_masking),
        ("gradient flow", check_gradient_flow),
        ("overfit one batch", check_overfit_single_batch),
    ]
    for name, fn in checks:
        try:
            detail = fn(model)
            results.append((name, True, detail))
        except Exception:
            results.append((name, False, traceback.format_exc()))

    return _report(results)


def _report(results):
    print("=" * 70)
    n_pass = sum(1 for _, ok, _ in results if ok)
    for name, ok, detail in results:
        tag = "[PASS]" if ok else "[FAIL]"
        print(f"{tag} {name}")
        if ok:
            print(f"       {detail}")
        else:
            # indent the traceback
            for line in detail.rstrip().splitlines():
                print(f"       {line}")
    print("=" * 70)
    print(f"{n_pass}/{len(results)} checks passed")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
