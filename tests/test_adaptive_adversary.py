"""Adaptive red-team adversaries: attack the *defenses*, not the naive baseline.

C-axis: a defense that only beats a fixed attacker is weak. These check the defenses
at an adaptive attacker's optimum. One holds (the depth gate is Goodhart-resistant to
*more breadth*); one surfaces the expected limit (it trusts the depth *signal*, so a
forger evades) — a finding left failing.
"""
import numpy as np
import pytest

from chord import ChordConfig, UserKnobs
from chord.loop import Chord
from chord.feed.value import FactorContext, blended_value
from chord.types import Exposure, ExposureSource, Post, Reaction


def _broad_world(bait_depth):
    """Two clusters mildly approve both a genuine (high-quality) post and a bait post;
    the bait's depth *signal* is the adaptive knob."""
    posts = {
        "G": Post("G", "ag", features={"depth": 0.85}),   # genuine quality
        "B": Post("B", "ab", features={"depth": bait_depth}),  # bait: quality low, signal = knob
        "L": Post("L", "al", features={"depth": 0.5}),
    }
    rx, exps = [], []
    for u in range(12):
        left = u < 6
        for pid in ("G", "B"):
            rx.append(Reaction(f"u{u}", pid, 0.55))         # both broadly, mildly liked
            exps.append(Exposure(f"u{u}", pid, source=ExposureSource.ORGANIC, propensity=0.5))
        rx.append(Reaction(f"u{u}", "L", 1.0 if left else -1.0))
        exps.append(Exposure(f"u{u}", "L", source=ExposureSource.ORGANIC, propensity=0.5))
    return posts, rx, exps


def _value(chord, st, pid, posts, cfg):
    ctx = FactorContext(
        user_id="u0", post=posts[pid],
        b_lcb=st.bridging.b_lcb.get(pid, 0.0),
        result=st.result, divisiveness=st.divisiveness,
        knobs=UserKnobs(M=1.0),
        extras={"depth": posts[pid].features["depth"],
                "depth_reward": cfg.depth_reward, "depth_gate": cfg.depth_gate},
    )
    return blended_value(ctx)


def test_depth_gate_resists_more_breadth():
    """Adaptive bait can't beat the gate by buying more breadth (approval): with an
    honest low depth signal, a broadly-liked bait stays below genuine quality."""
    cfg = ChordConfig(d=2, n_clusters=2, mf_iters=40, depth_reward=0.5, depth_gate=0.5)
    posts, rx, exps = _broad_world(bait_depth=0.1)      # honest: shallow bait signals shallow
    chord = Chord(cfg, seed=0)
    st = chord.fit_window(rx, posts, exps)
    assert _value(chord, st, "B", posts, cfg) < _value(chord, st, "G", posts, cfg), (
        "honest-signal bait should rank below genuine quality"
    )


def test_forged_depth_signal_evades_the_gate():
    """FOUND LIMIT (left failing): the depth defense trusts the depth *signal*, so a
    baiter that forges a high depth score makes shallow content match genuine quality.
    This is the documented §10 caveat — the signal's own integrity is load-bearing."""
    cfg = ChordConfig(d=2, n_clusters=2, mf_iters=40, depth_reward=0.5, depth_gate=0.5)
    posts, rx, exps = _broad_world(bait_depth=0.95)     # forged: shallow bait claims depth
    chord = Chord(cfg, seed=0)
    st = chord.fit_window(rx, posts, exps)
    vB = _value(chord, st, "B", posts, cfg)
    vG = _value(chord, st, "G", posts, cfg)
    print(f"\n[adaptive] forged-depth bait value={vB:.3f} vs genuine value={vG:.3f}")
    # We WANT the gate to still hold bait below genuine even when the signal is forged;
    # it cannot (it trusts the signal), so this fails — documenting the dependence.
    assert vB < vG - 0.05, (
        f"forged-depth bait ({vB:.3f}) matched/beat genuine quality ({vG:.3f}); the "
        f"depth defense trusts the signal and is evadable by forging it."
    )
