"""Read pages with a checkpoint in offline vLLM and score every answer with the reward's own function.

Follows ``nemotron/recipes/parse/dental/evaluate.py`` (a Ray worker of its own that gives the GPU back, the same
engine, ``n`` sampled answers a page), with three differences:

* the score is ours (``htr_reward.components``), and here the end is known: an answer ``ended`` when its last token
  is EOS and is ``capped`` when vLLM's finish reason is ``length``;
* the sampled answers use the rollout temperature of the training YAML (1.0 as theirs unless changed), so mining
  measures the spread RLVR will see;
* with ``greedy`` penalties, each page is also read greedily at each repetition penalty (1.0, and the 1.1 we serve
  with), so before/after shows whether RLVR removes the need for the penalty.

``answers.json`` in the output keeps every answer with its parts, the page's image and its truth.
"""

from __future__ import annotations

import gc
import hashlib
import json
import sys
from pathlib import Path

from htr_reward import components

HERE = Path(__file__).resolve().parent
MEANS = ("reward", "text", "cer", "boxes", "box_f1", "looped", "capped", "ended", "gate")


def evaluation_protocol(samples: int, settings: dict, greedy: list[float]) -> dict:
    """The generation and scoring contract an evaluation is identified by."""
    scorers = hashlib.sha256()
    for name in ("htr_reward.py", "htr_evaluate.py"):
        scorers.update((HERE / name).read_bytes())
    return {
        "schema_version": 1,
        "task": "htr",
        "scorer_sha256": scorers.hexdigest(),
        "weights": settings["weights"],
        "loop": settings["loop"],
        "generation": {
            "n": samples,
            "temperature": settings["temperature"],
            "top_p": settings["top_p"],
            "seed": settings["seed"],
            "max_tokens": settings["max_new_tokens"],
            "skip_special_tokens": False,
            "repetition_penalty": 1.0,
        },
        "greedy": [
            {
                "n": 1,
                "temperature": 0.0,
                "max_tokens": settings["max_new_tokens"],
                "skip_special_tokens": False,
                "repetition_penalty": penalty,
            }
            for penalty in greedy
        ],
        "engine": {
            "dtype": "bfloat16",
            "distributed_executor_backend": "uni",
            "max_model_len": settings["max_model_len"],
            "max_num_seqs": 32,
            "max_num_batched_tokens": settings["max_num_batched_tokens"],
            "kv_cache_memory_bytes": settings["kv_cache_memory_bytes"],
            "enforce_eager": True,
            "enable_prefix_caching": False,
            "limit_mm_per_prompt": {"image": 1},
            "generation_config": "vllm",
        },
    }


def evaluation_key(model, rows, protocol, processor_source=None) -> str:
    """Identify an evaluation by its model files, its pages and its protocol."""
    model = Path(model).absolute()
    processor = Path(processor_source or model).absolute()
    files = list(model.glob("*.safetensors")) + list(model.glob("*.json"))
    files += list(processor.glob("*token*.json"))
    files += [Path(row["image"]) for row in rows]
    states = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    payload = {
        "version": 1,
        "files": sorted(set(states)),
        "rows": [dict(row, image=str(Path(row["image"]).resolve())) for row in rows],
        "protocol": protocol,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def scored(answer, row, tokenizer, weights, loop) -> dict:
    """One vLLM completion as a record: its text, how it ended and the reward's parts."""
    bool(answer.token_ids and answer.token_ids[-1] == tokenizer.eos_token_id)
    capped = answer.finish_reason == "length"
    parts = components(
        answer.text,
        row["target"],
        weights=weights,
        loop=loop,
        token_ids=list(answer.token_ids),
        eos_token_id=tokenizer.eos_token_id,
        capped=capped,
    )
    return dict(answer=answer.text, tokens=len(answer.token_ids), **parts)


def evaluate(model, rows, output, protocol, processor_source=None):
    """Generate with vLLM in this process, score, and write ``request.json`` and ``answers.json``."""
    from PIL import Image
    from vllm import LLM, SamplingParams

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "request.json").write_text(
        json.dumps({"key": evaluation_key(model, rows, protocol, processor_source), "protocol": protocol})
    )
    view = output / "model"
    view.mkdir()
    model = Path(model).absolute()
    for source in model.iterdir():
        if source.is_file() and source.name != "config.json":
            (view / source.name).symlink_to(source)
    config = json.loads((model / "config.json").read_text())
    config["architectures"] = ["RadioMBartPrefixForCausalLM"]
    (view / "config.json").write_text(json.dumps(config))
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "preprocessor_config.json",
        "processor_config.json",
    ):
        source = Path(processor_source or model) / name
        if source.is_file() and not (view / name).exists():
            (view / name).symlink_to(source.absolute())
    llm = LLM(model=str(view), trust_remote_code=True, **protocol["engine"])
    tokenizer = llm.get_tokenizer()
    requests = []
    for row in rows:
        with Image.open(row["image"]) as image:
            image = image.convert("RGB")
        requests.append(
            {
                "prompt_token_ids": [tokenizer.pad_token_id]
                + tokenizer.encode(row["prompt"], add_special_tokens=False),
                "multi_modal_data": {"image": image},
            }
        )
    weights, loop = protocol["weights"], protocol["loop"]
    sampled = llm.generate(requests, SamplingParams(**protocol["generation"]))
    results = [
        {
            "id": row["id"],
            "image": row["image"],
            "context_id": row["context_id"],
            "target": row["target"],
            "answers": [scored(answer, row, tokenizer, weights, loop) for answer in generated.outputs],
            "greedy": {},
        }
        for row, generated in zip(rows, sampled, strict=True)
    ]
    for params in protocol["greedy"]:
        outputs = llm.generate(requests, SamplingParams(**params))
        for row, result, generated in zip(rows, results, outputs, strict=True):
            result["greedy"][f"{params['repetition_penalty']:g}"] = scored(
                generated.outputs[0], row, tokenizer, weights, loop
            )
    (output / "answers.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    del llm
    gc.collect()
    return results


def evaluate_isolated(model, rows, output, *, samples, processor_source, settings, greedy):
    """Run ``evaluate`` in a Ray worker that releases its GPU when finished; reuse a finished identical one."""
    import ray
    from nemotron.recipes.parse.training.config import worker_env

    output = Path(output)
    protocol = evaluation_protocol(samples, settings, greedy)
    if (output / "answers.json").is_file():
        saved_key = json.loads((output / "request.json").read_text())["key"]
        if saved_key != evaluation_key(model, rows, protocol, processor_source):
            raise ValueError("Evaluation inputs changed; choose a new output directory")
        print(f"Reusing completed evaluation: {len(rows)} pages × {samples} answers")
        return json.loads((output / "answers.json").read_text())
    owns_ray = not ray.is_initialized()
    if owns_ray:
        ray.init(num_cpus=4, num_gpus=1, include_dashboard=False)
    worker = ray.remote(num_gpus=1, num_cpus=4, max_calls=1)(evaluate).options(
        runtime_env={"py_executable": sys.executable, "env_vars": worker_env()}
    )
    try:
        return ray.get(worker.remote(model, rows, output, protocol, processor_source))
    finally:
        if owns_ray:
            ray.shutdown()


def averaged(answers: list[dict]) -> dict:
    """Means of the reward's parts over answers, and the CER of their pages pooled (edits over characters)."""
    if not answers:
        raise ValueError("No answers to summarise")
    known = [a for a in answers if a["boxes"] >= 0]
    means = {key: sum(a[key] for a in answers) / len(answers) for key in MEANS if key not in ("boxes", "box_f1")}
    for key in ("boxes", "box_f1"):
        means[key] = sum(a[key] for a in known) / len(known) if known else None
    means["cer_pooled"] = sum(a["cer"] * a["chars"] for a in answers) / max(1.0, sum(a["chars"] for a in answers))
    means["tokens"] = sum(a["tokens"] for a in answers) / len(answers)
    return means


def summarize(groups: list[dict]) -> dict:
    """The sampled answers' means, how much a page's answers differ, and each greedy reading's means."""
    counts = [len(group["answers"]) for group in groups]
    if not counts or min(counts) == 0 or len(set(counts)) != 1:
        raise ValueError("Evaluation requires equal, nonzero sample counts per page")
    flattened = [answer for group in groups for answer in group["answers"]]
    penalties = sorted({penalty for group in groups for penalty in group.get("greedy", {})})
    return {
        "pages": len(groups),
        "answers": len(flattened),
        "samples": counts[0],
        "sampled": averaged(flattened),
        "pages_with_a_looped_or_capped_answer": sum(
            any(a["looped"] > 0 or a["capped"] > 0 for a in group["answers"]) for group in groups
        ),
        "greedy": {penalty: averaged([group["greedy"][penalty] for group in groups]) for penalty in penalties},
    }
