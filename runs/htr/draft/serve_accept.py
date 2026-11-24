"""What vLLM would accept of a DFlash2 draft, worked out from a cache (``make_cache.py``) without a server.

The trainer's validation draws anchors at random and wants a whole block after each. vLLM drafts in sequence: the
anchor is the token the reader just gave, ``k`` masks follow it, the drafted tokens are kept up to the first that
is not the reader's own, and the next anchor is the reader's token after those. As the page's final text is the
reader's own answer whatever the draft says, the same walk can be made over a cached answer: the draft is run with
every answer token as the anchor (the context is the reader's states before it, as cached), the count of drafted
tokens that match the answer is kept for each, and the walk then steps through them. Printed as one JSON line:

- ``accepted_per_draft``: accepted tokens over drafts along the walk (vLLM: Accepted / (Drafted / k)); vLLM's
  "mean acceptance length" is this plus one;
- ``position_rates``: the share of the walk's drafts accepted through position 1, 2, ... (vLLM's "per-position
  acceptance rate");
- the same over every answer token as an anchor (``any_anchor_*``), and without the selector (``base_*``).

The draft, the tables and the block come from the recipe's YAML as in ``train_draft.py``; ``--k`` is vLLM's
``num_speculative_tokens`` (the draft is run with 1 + k queries a block, as vLLM runs it).

    CUDA_VISIBLE_DEVICES=2 /mnt/scratch/gabriel/envs/automodel/bin/python runs/htr/draft/serve_accept.py \\
        -c runs/htr/draft/dflash2_parse.yaml --k 16 --cache CACHE [--recipe_args.init_draft_path DRAFT] ...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from nemo_automodel.components.attention.dflash_mask import create_dflash_sdpa_mask
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config

sys.path.insert(0, str(Path(__file__).parent))
from train_draft import ParseDFlash2Recipe  # noqa: E402


def taken(argv: list[str], flag: str, default: str | None = None) -> str | None:
    """Pop ``flag VALUE`` from ``argv``."""
    if flag not in argv:
        return default
    at = argv.index(flag)
    value = argv[at + 1]
    del argv[at : at + 2]
    return value


@torch.no_grad()
def accepted_from(recipe: ParseDFlash2Recipe, row: dict[str, torch.Tensor], k: int, chunk: int) -> torch.Tensor:
    """For each answer token as the anchor: how many of ``k`` drafted tokens are the answer's, with and without the
    selector. Returns ``[2, answer tokens]``."""
    module, draft, device = recipe.trainer_module, recipe.draft_model, recipe.device
    block = 1 + k
    ids = row["input_ids"][None].to(device)
    states = row["hidden_states"][None].to(device, recipe.compute_dtype)
    seq = ids.shape[1]
    anchors_all = row["loss_mask"].nonzero()[:, 0].to(device)
    offsets = torch.arange(block, device=device)
    counts = []
    for anchors in anchors_all.split(chunk):
        anchors = anchors[None]
        keep = torch.ones_like(anchors, dtype=torch.bool)
        noise = torch.full((1, anchors.shape[1], block), module.mask_token_id, device=device)
        noise[..., 0] = ids[0, anchors[0]]
        embedded = module._embed_noise_ids(noise.view(1, -1))  # noqa: SLF001  the trainer's own embedding path
        positions = torch.cat([torch.arange(seq, device=device)[None], (anchors[..., None] + offsets).view(1, -1)], 1)
        mask = create_dflash_sdpa_mask(anchors, keep, seq, block, device, dtype=embedded.dtype)
        hidden = draft(
            position_ids=positions,
            noise_embedding=embedded,
            target_hidden=states,
            attention_mask=mask,
            conv_block_size=block,
        )
        logits = draft.compute_logits(hidden, module.lm_head).view(1, -1, block, module.lm_head.weight.shape[0])
        hidden = hidden.view(1, -1, block, hidden.shape[-1])[:, :, 1:]
        logits = logits[:, :, 1:]
        label_at = anchors[..., None] + offsets
        inside = label_at < seq
        labels = ids[0, label_at.clamp(max=seq - 1)]
        selector = draft.candidate_selector
        unary, candidates = torch.topk(logits, selector.top_k, dim=-1)
        # the walk scores a position after the token it picked before; up to the first miss that is the answer's
        scores = selector.pair_scores(hidden, unary, candidates, labels[..., :-1])
        picked = candidates.gather(-1, scores.argmax(-1, keepdim=True)).squeeze(-1)
        per = []
        for guess in (picked, logits.argmax(-1)):
            right = (guess == labels[..., 1:]) & inside[..., 1:]
            per.append(right.long().cumprod(-1).sum(-1)[0])
        counts.append(torch.stack(per))
    return torch.cat(counts, dim=1).cpu()


def main() -> None:
    """Print the acceptance of the YAML's draft over ``--cache`` for ``--k`` drafted tokens."""
    argv = sys.argv[1:]
    k = int(taken(argv, "--k", "16"))
    cache = Path(taken(argv, "--cache"))
    chunk = int(taken(argv, "--chunk", "96"))
    manifest = json.loads((cache / "manifest.json").read_text("utf-8"))
    argv += ["--recipe_args.train_cache_path", str(cache), "--recipe_args.val_cache_path", str(cache)]
    argv += ["--recipe_args.target_model_name_or_path", manifest["target_model_name_or_path"]]
    recipe = ParseDFlash2Recipe(parse_args_and_load_config(argv=argv))
    recipe.setup()
    recipe.trainer_module.eval()

    sums = {name: [0, 0] for name in ("walk", "walk_base", "any", "any_base")}
    through = torch.zeros(k)
    drafts = 0
    paths = sorted(cache.glob("*.ckpt"))
    for path in paths:
        row = torch.load(path, map_location="cpu", weights_only=True)
        counts = accepted_from(recipe, row, k, chunk)
        answer = counts.shape[1]
        # the anchor of the last draft may be any token but the end token: nothing is drafted after it
        for name, line in (("any", counts[0, :-1]), ("any_base", counts[1, :-1])):
            sums[name][0] += int(line.sum())
            sums[name][1] += len(line)
        for name, line in (("walk", counts[0]), ("walk_base", counts[1])):
            at = 0
            while at < answer - 1:
                got = int(line[at])
                sums[name][0] += got
                sums[name][1] += 1
                if name == "walk":
                    through[:got] += 1
                    drafts += 1
                at += got + 1
    result = {
        "cache": str(cache),
        "draft": str(recipe.cfg.recipe_args.get("init_draft_path", None)),
        "pages": len(paths),
        "k": k,
        "drafts": drafts,
        "accepted_per_draft": round(sums["walk"][0] / sums["walk"][1], 3),
        "base_accepted_per_draft": round(sums["walk_base"][0] / sums["walk_base"][1], 3),
        "any_anchor_accepted": round(sums["any"][0] / sums["any"][1], 3),
        "any_anchor_base_accepted": round(sums["any_base"][0] / sums["any_base"][1], 3),
        "position_rates": [round(rate, 3) for rate in (through / drafts).tolist()],
    }
    print("SERVING " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
