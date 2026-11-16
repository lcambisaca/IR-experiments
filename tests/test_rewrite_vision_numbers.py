"""The numbers reported in the paper come out of the committed results unchanged."""
import rewrite_vision_numbers


def test_paper_numbers_reproduce_from_committed_results(capsys):
    rewrite_vision_numbers.rewrite_numbers()
    rewrite_vision_numbers.vision_numbers()
    out = capsys.readouterr().out
    for line in [
        "caption alone R@10: 74.5",
        "one paraphrase rewrite (avg) R@10: 73.7",
        "one decompose rewrite (avg) R@10: 35.8",
        "oracle over all 21 queries R@10: 86.5",
        "oracle over all 21 vs caption: +12.0 [+7.5, +16.5]",
        "all 21 fused (mean embedding, caption x3) vs caption: +0.5 [-2.0, +3.0]",
        "paraphrase z-max vs caption, 200 targets: +3.5 [+1.0, +6.0]",
        "paraphrase z-max vs caption, all 2064 targets: -0.3 [-1.3, +0.6]",
        "rule 1 (vision order) R@10 vs BLIP: -1.9 [-13.7, +6.5]",
        "rule 2 (vision filter) R@1 vs BLIP: +4.0 [-8.1, +14.2]",
        "rule 2 (vision filter) R@10 vs BLIP: +4.0 [+2.0, +6.5]",
        "rule 1, BLIP rank 11-30: top 10 0 -> 11 of 27",
        "rule 2, BLIP rank 11-30: top 10 0 -> 8 of 27",
        "shortlists where 10+ of the 30 photos scored above 0.96: 12 of 47",
        "prompt check, 6 targets: target 0.93 vs others 0.41",
    ]:
        assert line in out, line
