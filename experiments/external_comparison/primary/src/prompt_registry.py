"""Lossless schematic domain view for bounded model prompts.

Runtime grounds are generated from these schemas and role/region bindings. The
full ground-action cross product is intentionally not duplicated in each prompt.
"""
import copy

def for_prompt(registry):
 return {k:copy.deepcopy(v) for k,v in registry.items() if k not in {'ground_actions','target','gold_tree','reference_plan','test_solution','intent_examples'}}
