import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nli_common as nc  # noqa: E402

T = "Premise: {premise}\nHypothesis: {hypothesis}"


def test_prompt_strips_like_python_and_substitutes_once():
    assert nc.prompt(T, "  a {hypothesis} b \x1c", " h ") == "Premise: a {hypothesis} b\nHypothesis: h"


def test_noul_hypotheses_follow_the_local_openjev_adapter():
    q = {"type": "noul", "instructions": " Is it allowed? ", "criteria": {"true": "All hold."}}
    assert nc.decide_hypotheses(q) == [
        ("yes", 'The answer to "Is it allowed?" is yes: All hold.'),
        ("no", 'The answer to "Is it allowed?" is no: No'),
    ]


def test_choice_and_score_hypotheses():
    c = {"type": "choice", "instructions": "Route?", "criteria": {"billing": "", "tech": "A fault."}}
    assert [h for _, h in nc.decide_hypotheses(c)] == [
        'The answer to "Route?" is billing: billing', 'The answer to "Route?" is tech: A fault.']
    s = {"type": "score", "instructions": "Rate", "criteria": ["bad", "good"]}
    assert nc.decide_hypotheses(s) == [("0", 'The answer to "Rate" is 0: bad'), ("1", 'The answer to "Rate" is 1: good')]


def test_char_windows_match_openjevs_shape():
    p = "x" * 100
    assert nc.char_windows(p, 200) == [p]
    w = nc.char_windows(p, 24)          # overlap 24 // 12 = 2, step 22
    assert [len(x) for x in w] == [24, 24, 24, 24, 12]
    assert w[1] == p[22:46]


def test_fit_windows_shrinks_until_every_prompt_fits():
    toks = lambda s: len(s) // 4 + 1
    wins = nc.fit_windows(T, "y" * 1000, ["short hyp"], toks, 64)
    assert all(toks(nc.prompt(T, w, "short hyp")) <= 64 for w in wins)
    assert len(wins) > 1
    assert nc.fit_windows(T, "tiny", ["h"], toks, 64) == ["tiny"]


def test_normalise_is_max_over_windows_then_sum_to_one():
    p = nc.normalise_entailment([[0.2, 0.6], [0.4, 0.1]])  # [window][option]
    assert p == [0.4 / 1.0, 0.6 / 1.0]


def test_golden_round_trips(tmp_path):
    out = tmp_path / "g.json"
    nc.write_golden(str(out))
    g = json.load(open(out))
    assert {"prompt", "hypotheses", "windows", "fit", "normalise"} <= set(g)
