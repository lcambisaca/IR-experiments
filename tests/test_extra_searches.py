"""Matan's extra-search strategies: reply parsing and merging, without models."""
import torch

from agentir.extra_searches import parse_variants, round_outcome


def test_parse_variants_reads_each_strategy_format():
    assert parse_variants("split", 'Sure:\n{"queries": ["a cat", "a street", "a cat", " "]}') == ["a cat", "a street"]
    assert parse_variants("focus", '{"subjects": ["cat"], "queries": ["a cat on a street"]}') == ["a cat on a street"]
    details = '{"colors": "a red bus", "setting": "a bus in a city", "action": "", "count": "two buses", "background": "a bus, hills behind"}'
    assert parse_variants("details", details) == ["a red bus", "a bus in a city", "two buses", "a bus, hills behind"]
    assert parse_variants("split", "not json") == []
    assert parse_variants("split", '{"queries": [broken') == []


def test_round_outcome_merges_and_controls():
    images, target = 100, 7
    scores = torch.linspace(1, 0, images).repeat(3, 1)  # Every search ranks photo 0 first ... photo 99 last.
    scores[0, target] = -1.0                             # The original search buries the target (last place).
    scores[1, target] = 2.0                              # One variant ranks it first.
    outcome = round_outcome(scores, target)
    assert outcome["original"] == images
    assert outcome["any_top10"] and not outcome["control_same_count"]
    assert outcome["max"] == 1                           # Best single score wins under max.
    assert outcome["mean"] is not None and outcome["mean"] > 1
    assert outcome["pool_size"] == 11 and outcome["k"] == 3


def test_round_outcome_target_outside_every_top10():
    scores = torch.linspace(1, 0, 50).repeat(2, 1)
    outcome = round_outcome(scores, 40)
    assert outcome["original"] == 41 and outcome["max"] is None and outcome["mean"] is None
    assert not outcome["any_top10"] and not outcome["control_same_count"]
