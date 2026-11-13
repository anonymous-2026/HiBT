"""Adapt the genuine LLM-BT DistilBERT Parser on the shared non-test pool.

BIO spans derive only from the coordinator's 240/48 training/development
teacher records. No benchmark instruction, evaluator goal, or test identifier
is opened by this script. Original checkpoint labels and architecture remain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

LABELS = ["O", "B-Action", "B-Target", "I-Target", "B-Destination", "I-Destination", "B-Location", "I-Location"]
SOURCE = "henryhaotian/LLM-BT@c69c18d0cf4b78f166ed352fc0fa8470823b32f6/LLMBT/Parser/keywords_extraction"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _role_parts(action):
    name, args = action["name"], action["args"]
    if name in {"insert", "place", "screw", "detach", "pull", "unscrew"}:
        return name, args[-2], args[-1]
    if name == "pick_up":
        return name, args[-1], args[-2]
    if name in {"load_tool", "change_tool"}:
        return name, args[-1], args[0]
    if name in {"unload_tool", "put_down"}:
        return name, args[1], None
    if name in {"place_on", "place_in"}:
        return name, args[0], args[1]
    if name in {"open", "close", "turnon", "turnoff"}:
        return name, args[0], None
    raise ValueError(f"No declared domain ATL mapping for {name}")


def _example(name, target, destination, style):
    # Exact registered operation identifiers are used by the language Reasoning
    # prompt too. BIO labels train real neural extraction, not a test parser.
    preposition = {"insert": "into", "place": "on", "screw": "into", "pick_up": "with",
                   "load_tool": "onto", "change_tool": "onto", "detach": "from", "pull": "from",
                   "unscrew": "from", "place_on": "on", "place_in": "in"}.get(name, "to")
    prefix = "" if style == 0 else "Then "
    text = prefix + name + " " + target
    spans = [{"start": len(prefix), "end": len(prefix) + len(name), "role": "Action"},
             {"start": len(prefix) + len(name) + 1, "end": len(text), "role": "Target"}]
    if destination is not None:
        text += " " + preposition + " "
        start = len(text)
        text += destination
        spans.append({"start": start, "end": len(text), "role": "Destination"})
    return text + ".", spans


def build_examples(rows):
    examples = []
    for row in rows:
        if not row.get("teacher_actions"):
            raise ValueError("Require shared teacher rows with actual ground actions")
        for index, action in enumerate(row["teacher_actions"]):
            name, target, destination = _role_parts(action)
            for style in [0, 1]:
                text, spans = _example(name, target, destination, style)
                examples.append({"sample_id": f"{row['sample_id']}/action{index}/style{style}",
                                 "source_sample_id": row["sample_id"], "text": text,
                                 "spans": spans, "source_action": action,
                                 "generation": "BIO rendering of shared non-test teacher action"})
    return examples


def prepare(train, validation):
    train_ids, val_ids = {x["sample_id"] for x in train}, {x["sample_id"] for x in validation}
    if len(train_ids) != len(train) or len(val_ids) != len(validation) or train_ids & val_ids:
        raise ValueError("Source sample IDs must be unique and split-disjoint")
    def objects(rows):
        return {o["name"] for row in rows for o in row["world_state"]["objects"]}
    if objects(train) & objects(validation):
        raise ValueError("Expected shared pool with disjoint train/validation object namespaces")
    output = build_examples(train), build_examples(validation)
    if {x["text"] for x in output[0]} & {x["text"] for x in output[1]}:
        raise ValueError("NER text overlaps train/validation")
    return output


def auxiliary_teacher_rows(path):
    """Read only the shared non-Long XML teacher files, never evaluation rows."""
    rows = []
    for row in read_rows(path):
        if not row.get("registry") or not row.get("output"):
            raise ValueError("Auxiliary rows must be shared XML training/validation records")
        actions, seen = [], set()
        for node in ET.fromstring(row["output"]).iter():
            name = node.tag
            if name not in row["registry"]["actions"]:
                continue
            spec = row["registry"]["actions"][name]
            args = [node.attrib[arg] for arg in spec["arguments"]]
            key = name, tuple(args)
            if key not in seen:
                actions.append({"name": name, "args": args}); seen.add(key)
        if not actions:
            raise ValueError(f"Auxiliary teacher has no registered action: {row['sample_id']}")
        rows.append({"sample_id": row["sample_id"], "world_state": row["world_state"],
                     "teacher_actions": actions, "source": "shared non-Long XML teacher"})
    return rows


def align_labels(tokenizer, row, max_length):
    encoded = tokenizer(row["text"], truncation=False, return_offsets_mapping=True)
    if len(encoded["input_ids"]) > max_length:
        raise ValueError(f"NER sample exceeds declared length: {row['sample_id']}")
    labels, observed = [], set()
    for start, end in encoded.pop("offset_mapping"):
        if start == end:
            labels.append(-100)
            continue
        span = next((x for x in row["spans"] if start < x["end"] and end > x["start"]), None)
        if span is None:
            labels.append(0)
            continue
        role = span["role"]
        # The official label set has no I-Action. Each labeled subword of one
        # action identifier receives B-Action; inference merges same word spans.
        begin = role == "Action" or role not in observed
        label = ("B-" if begin else "I-") + role
        labels.append(LABELS.index(label))
        observed.add(role)
    encoded["labels"] = labels
    return encoded


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-data", type=Path, required=True)
    p.add_argument("--validation-data", type=Path, required=True)
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--aux-train-data", type=Path)
    p.add_argument("--aux-validation-data", type=Path)
    p.add_argument("--aux-split-manifest", type=Path)
    p.add_argument("--model", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=41714)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    a = p.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise ValueError("Use a new output directory; do not overwrite a run")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    train_source, val_source = read_rows(a.train_data), read_rows(a.validation_data)
    train, val = prepare(train_source, val_source)
    split = json.loads(a.split_manifest.read_text())
    if set(split["train_ids"]) != {x["sample_id"] for x in train_source} or set(split["validation_ids"]) != {x["sample_id"] for x in val_source}:
        raise ValueError("Training inputs do not match the frozen shared split manifest")
    auxiliary = {}
    if any([a.aux_train_data, a.aux_validation_data, a.aux_split_manifest]):
        if not all([a.aux_train_data, a.aux_validation_data, a.aux_split_manifest]):
            raise ValueError("All three auxiliary input/manifest arguments are required")
        aux_train_source = auxiliary_teacher_rows(a.aux_train_data)
        aux_val_source = auxiliary_teacher_rows(a.aux_validation_data)
        aux_manifest = json.loads(a.aux_split_manifest.read_text())
        ids_train = {x["sample_id"] for x in aux_train_source}
        ids_val = {x["sample_id"] for x in aux_val_source}
        if ids_train & ids_val or ids_train != set(aux_manifest["train_ids"]) or ids_val != set(aux_manifest["validation_ids"]):
            raise ValueError("Auxiliary teacher split differs from frozen shared non-Long split")
        if not aux_manifest.get("all_Long_targets_excluded_from_supervision"):
            raise ValueError("Auxiliary supervision must explicitly exclude Long test targets")
        aux_train, aux_val = build_examples(aux_train_source), build_examples(aux_val_source)
        auxiliary = {"source_train_count": len(aux_train_source), "source_validation_count": len(aux_val_source),
                     "ner_train_examples": len(aux_train), "ner_validation_examples": len(aux_val),
                     "train_sha256": digest(a.aux_train_data), "validation_sha256": digest(a.aux_validation_data),
                     "split_sha256": digest(a.aux_split_manifest),
                     "primitive_types": sorted({x["source_action"]["name"] for x in aux_train}),
                     "validation_primitive_texts_shared_with_train": len({x["text"] for x in aux_train} & {x["text"] for x in aux_val}),
                     "isolation": "source task IDs; known primitive vocabulary may be shared"}
        train += aux_train; val += aux_val
    for name, rows in [("train_examples.jsonl", train), ("validation_examples.jsonl", val)]:
        with (a.output_dir / name).open("w") as f:
            for row in rows: f.write(json.dumps(row) + "\n")
    config = {**vars(a), "source": SOURCE, "labels": LABELS,
              "source_train_count": len(train_source), "source_validation_count": len(val_source),
              "ner_train_examples": len(train), "ner_validation_examples": len(val),
              "train_sha256": digest(a.train_data), "validation_sha256": digest(a.validation_data),
              "split_sha256": digest(a.split_manifest), "base_checkpoint": str(a.model),
              "auxiliary_nonLong_domain": auxiliary,
              "selection": "minimum non-test validation loss; test cases never read"}
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    (a.output_dir / "training_config.json").write_text(json.dumps(config, indent=2) + "\n")
    if a.prepare_only:
        print(json.dumps({"status": "NER_DATA_PREPARED_NO_TRAINING", **{k: config[k] for k in ["source_train_count", "source_validation_count", "ner_train_examples", "ner_validation_examples"]}}, indent=2))
        return
    if a.model is None or not (a.model / "config.json").exists():
        raise ValueError("Require centrally localized original LLM-BT NER checkpoint")
    if a.device == "cuda" and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Coordinator must bind CUDA_VISIBLE_DEVICES to the reviewed GPU UUID")
    import numpy as np
    import torch
    import transformers
    from torch.utils.data import Dataset
    from transformers import (AutoTokenizer, AutoModelForTokenClassification, Trainer,
                              TrainingArguments, DataCollatorForTokenClassification, TrainerCallback, set_seed)
    set_seed(a.seed)
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(a.model, local_files_only=True)
    if [model.config.id2label[i] for i in range(len(LABELS))] != LABELS:
        raise ValueError("Original checkpoint BIO label mapping differs from expected source")
    class Samples(Dataset):
        def __init__(self, rows): self.rows = [align_labels(tokenizer, x, a.max_length) for x in rows]
        def __len__(self): return len(self.rows)
        def __getitem__(self, i): return self.rows[i]
    train_set, val_set = Samples(train), Samples(val)
    def metrics(prediction):
        logits, labels = prediction
        guessed = logits.argmax(-1)
        mask = labels != -100
        correct = guessed == labels
        tp = ((guessed == labels) & (labels != 0) & mask).sum()
        predicted = ((guessed != 0) & mask).sum()
        gold = ((labels != 0) & mask).sum()
        precision = float(tp / max(int(predicted), 1)); recall = float(tp / max(int(gold), 1))
        return {"token_accuracy": float(correct[mask].mean()), "entity_token_precision": precision,
                "entity_token_recall": recall, "entity_token_f1": 2 * precision * recall / max(precision + recall, 1e-12)}
    class LocalLog(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            with (a.output_dir / "metrics.jsonl").open("a") as f:
                f.write(json.dumps({"step": state.global_step, "epoch": state.epoch, "time": time.time(), **(logs or {})}) + "\n")
    args = TrainingArguments(output_dir=str(a.output_dir), num_train_epochs=a.epochs,
                             learning_rate=a.learning_rate, per_device_train_batch_size=a.batch_size,
                             per_device_eval_batch_size=a.batch_size, weight_decay=0.01,
                             eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
                             load_best_model_at_end=True, metric_for_best_model="eval_loss",
                             greater_is_better=False, use_cpu=a.device == "cpu", report_to=[],
                             seed=a.seed, logging_steps=10, dataloader_num_workers=0)
    trainer = Trainer(model=model, args=args, train_dataset=train_set, eval_dataset=val_set,
                      data_collator=DataCollatorForTokenClassification(tokenizer), compute_metrics=metrics,
                      callbacks=[LocalLog()])
    (a.output_dir / "environment.json").write_text(json.dumps({"python": sys.version,
        "platform": platform.platform(), "torch": torch.__version__, "transformers": transformers.__version__,
        "device": a.device}, indent=2) + "\n")
    result = trainer.train()
    final = a.output_dir / "ner_model"
    trainer.save_model(str(final)); tokenizer.save_pretrained(final)
    m = {**result.metrics, **trainer.evaluate()}
    (a.output_dir / "final_metrics.json").write_text(json.dumps(m, indent=2) + "\n")
    (final / "llmbt_ner_provenance.json").write_text(json.dumps({"method": "LLM-BT",
        "source_checkpoint": SOURCE, "shared_pool_train_rows": len(train_source),
        "shared_pool_validation_rows": len(val_source), "train_sha256": digest(a.train_data),
        "validation_sha256": digest(a.validation_data), "split_sha256": digest(a.split_manifest),
        "auxiliary_nonLong_domain": auxiliary,
        "status": "GENUINE_NER_NEURAL_DOMAIN_ADAPTATION_COMPLETE", "metrics": m}, indent=2) + "\n")
    print(json.dumps({"status": "NER_TRAINING_COMPLETE", "metrics": m}, indent=2))


if __name__ == "__main__":
    main()
