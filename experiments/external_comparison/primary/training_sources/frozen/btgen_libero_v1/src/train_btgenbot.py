"""Train a domain-adapted BTGenBot LoRA using auditable non-test supervision.

No GPU work is started on import. Model paths must already exist locally;
the coordinator handles downloads and authorized device/tmux execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

from methods.btgenbot import METHOD, SOURCE_VERSION, format_messages, xml_to_tree


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_digest(row: dict) -> str:
    """Exact instance hash shared with split preparation; no target/label needed."""
    value = {'instruction': row['instruction'], 'world_state': row.get('world_state', {})}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_rows(path: Path) -> list[dict]:
    text = path.read_text()
    rows = json.loads(text) if text.lstrip().startswith('[') else [json.loads(x) for x in text.splitlines() if x.strip()]
    if not rows:
        raise ValueError(f'Empty supervised split: {path}')
    ids = []
    for row in rows:
        for key in ['sample_id', 'instruction', 'registry', 'output']:
            if key not in row:
                raise ValueError(f'Missing {key} in {path}')
        xml_to_tree(row['output'], row['registry'])
        ids.append(str(row['sample_id']))
    if len(set(ids)) != len(ids):
        raise ValueError(f'Duplicate training sample IDs: {path}')
    return rows


def validate_splits(train: list[dict], validation: list[dict], manifest: dict) -> dict:
    if manifest.get('isolation') not in {'instance', 'family'}:
        raise ValueError('Manifest must declare isolation=instance or family')
    sets = {key: {str(x) for x in manifest[key + '_ids']}
            for key in ['train', 'validation', 'evaluation']}
    if sets['train'] != {str(x['sample_id']) for x in train}:
        raise ValueError('Training IDs do not match frozen split manifest')
    if sets['validation'] != {str(x['sample_id']) for x in validation}:
        raise ValueError('Validation IDs do not match frozen split manifest')
    if not sets['evaluation']:
        raise ValueError('Explicit evaluation IDs required')
    for left, right in [('train', 'validation'), ('train', 'evaluation'), ('validation', 'evaluation')]:
        if sets[left] & sets[right]:
            raise ValueError(f'{left}/{right} IDs overlap')
    train_hashes, val_hashes = {input_digest(x) for x in train}, {input_digest(x) for x in validation}
    eval_hashes = set(manifest.get('evaluation_input_hashes', []))
    if not eval_hashes:
        raise ValueError('Frozen evaluation_input_hashes required for exact instance audit')
    if train_hashes & val_hashes or train_hashes & eval_hashes or val_hashes & eval_hashes:
        raise ValueError('Training, validation, or evaluation input instances overlap')
    family_overlap = None
    if manifest['isolation'] == 'family':
        family_sets = {key: set(manifest[key + '_family_ids'])
                       for key in ['train', 'validation', 'evaluation']}
        for key, rows in [('train', train), ('validation', validation)]:
            if {str(x['family_id']) for x in rows} != family_sets[key]:
                raise ValueError(f'{key} family IDs mismatch')
        if any(family_sets[a] & family_sets[b] for a, b in [('train', 'validation'), ('train', 'evaluation'), ('validation', 'evaluation')]):
            raise ValueError('Declared family-disjoint split contains family overlap')
        family_overlap = False
    return {'isolation': manifest['isolation'], 'n_train': len(train), 'n_validation': len(validation),
            'n_evaluation_ids': len(sets['evaluation']), 'exact_instance_overlap': False,
            'family_overlap': family_overlap,
            'family_claim': 'verified family-disjoint' if manifest['isolation'] == 'family' else 'instance-disjoint; no structural-family exclusion claim'}


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path)
    p.add_argument('--initial-adapter', type=Path)
    p.add_argument('--train-data', type=Path, required=True)
    p.add_argument('--validation-data', type=Path, required=True)
    p.add_argument('--split-manifest', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--epochs', type=float, default=3.0)
    p.add_argument('--micro-batch-size', type=int, default=4)
    p.add_argument('--gradient-accumulation', type=int, default=64)
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--max-length', type=int, default=4096)
    p.add_argument('--lora-r', type=int, default=8)
    p.add_argument('--lora-alpha', type=int, default=16)
    p.add_argument('--lora-dropout', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--logging-steps', type=int, default=1)
    return p.parse_args()


def main():
    a = arguments()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise ValueError('Output must be new/empty; do not overwrite trained checkpoints')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    train, validation = load_rows(a.train_data), load_rows(a.validation_data)
    split = validate_splits(train, validation, json.loads(a.split_manifest.read_text()))
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(a).items()}
    config.update(method=METHOD, source_version=SOURCE_VERSION,
                  training_variant='domain-adapted response-only causal-LM SFT with LoRA',
                  target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                  epoch_provenance='User-configurable domain-adaptation setting; original paper states increased epochs without exact value',
                  effective_batch=a.micro_batch_size * a.gradient_accumulation,
                  split_audit=split, train_data_sha256=digest(a.train_data),
                  validation_data_sha256=digest(a.validation_data), split_manifest_sha256=digest(a.split_manifest))
    (a.output_dir / 'training_config.json').write_text(json.dumps(config, indent=2) + '\n')
    if a.validate_only:
        print(json.dumps({'status': 'DATA_VALIDATED_NO_TRAINING', 'split_audit': split}, indent=2))
        return
    if a.model is None or not a.model.exists():
        raise ValueError('--model must name an already-localized base checkpoint')
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Coordinator must explicitly bind CUDA_VISIBLE_DEVICES before training')
    import torch
    import transformers
    import peft
    from torch.utils.data import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback, set_seed
    if not torch.cuda.is_available():
        raise ValueError('Training requires the coordinator-authorized GPU')
    set_seed(a.seed)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    class Supervision(Dataset):
        def __init__(self, rows):
            self.records = []
            self.lengths = []
            for row in rows:
                messages = format_messages(row, row['registry'])
                prefix_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                # Render the exact inference prefix once, then append the response.
                # Some Mistral templates reposition system text when an assistant
                # turn is present; this keeps train/inference context identical.
                full_text = prefix_text + row['output'] + tokenizer.eos_token
                encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
                tokens = encoded['input_ids']
                if len(tokens) > a.max_length:
                    raise ValueError(f'Sample {row["sample_id"]} has {len(tokens)} tokens > max_length; labels cannot be silently truncated')
                # Tokenization may merge whitespace at the user/assistant boundary
                # (notably Mistral); mask using exact text offsets rather than
                # requiring separately tokenized prefixes to be identical.
                boundary = len(prefix_text)
                labels = [token if end > boundary else -100
                          for token, (start, end) in zip(tokens, encoded['offset_mapping'])]
                if not any(x != -100 for x in labels):
                    raise ValueError('No assistant supervision tokens after response mask')
                self.records.append({'input_ids': tokens, 'attention_mask': [1] * len(tokens), 'labels': labels})
                self.lengths.append(len(tokens))

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            return self.records[index]

    train_set, val_set = Supervision(train), Supervision(validation)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                               local_files_only=True, attn_implementation='sdpa')
    model.config.use_cache = False
    lora = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
                      target_modules=config['target_modules'], bias='none', task_type='CAUSAL_LM')
    if a.initial_adapter:
        from peft import PeftModel
        if not (a.initial_adapter / 'btgenbot_provenance.json').exists():
            raise ValueError('Initial adapter must have verified BTGenBot training provenance')
        model = PeftModel.from_pretrained(model, str(a.initial_adapter), is_trainable=True)
    else:
        model = get_peft_model(model, lora)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    def collate(rows):
        longest = max(len(row['input_ids']) for row in rows)
        return {key: torch.tensor([row[key] + [(-100 if key == 'labels' else tokenizer.pad_token_id if key == 'input_ids' else 0)] * (longest - len(row[key])) for row in rows], dtype=torch.long)
                for key in ['input_ids', 'attention_mask', 'labels']}

    class LocalLog(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.is_world_process_zero:
                row = {'step': state.global_step, 'epoch': state.epoch, 'time': time.time(), **(logs or {})}
                with (a.output_dir / 'metrics.jsonl').open('a') as out:
                    out.write(json.dumps(row) + '\n')

    env = {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
           'transformers': transformers.__version__, 'peft': peft.__version__,
           'train_token_lengths': {'min': min(train_set.lengths), 'max': max(train_set.lengths)},
           'validation_token_lengths': {'min': min(val_set.lengths), 'max': max(val_set.lengths)}}
    (a.output_dir / 'environment.json').write_text(json.dumps(env, indent=2) + '\n')
    training_args = TrainingArguments(output_dir=str(a.output_dir), num_train_epochs=a.epochs,
                                     per_device_train_batch_size=a.micro_batch_size,
                                     per_device_eval_batch_size=1,
                                     gradient_accumulation_steps=a.gradient_accumulation,
                                     learning_rate=a.learning_rate, warmup_ratio=0.03, weight_decay=0.0,
                                     lr_scheduler_type='cosine', bf16=True, tf32=True,
                                     gradient_checkpointing=True,
                                     gradient_checkpointing_kwargs={'use_reentrant': False},
                                     eval_strategy='epoch', save_strategy='epoch', save_total_limit=2,
                                     load_best_model_at_end=True, metric_for_best_model='eval_loss',
                                     greater_is_better=False, report_to=[], seed=a.seed,
                                     logging_steps=a.logging_steps, remove_unused_columns=False, label_names=['labels'])
    trainer = Trainer(model=model, args=training_args, train_dataset=train_set, eval_dataset=val_set,
                      data_collator=collate, callbacks=[LocalLog()])
    result = trainer.train()
    trainer.save_model(str(a.output_dir / 'adapter'))
    tokenizer.save_pretrained(a.output_dir / 'adapter')
    metrics = {**result.metrics, **trainer.evaluate()}
    (a.output_dir / 'final_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    provenance = {'adapter_method': METHOD, 'adapter_checkpoint': 'adapter',
                  'base_model': a.model.name, 'source_version': SOURCE_VERSION,
                  'training_config_sha256': digest(a.output_dir / 'training_config.json'),
                  'split_manifest_sha256': digest(a.split_manifest), 'split_audit': split,
                  'training_complete': True, 'selection': 'minimum validation loss; no evaluation success feedback'}
    (a.output_dir / 'adapter' / 'btgenbot_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'status': 'TRAINING_COMPLETE', 'metrics': metrics}, indent=2))


if __name__ == '__main__':
    main()
