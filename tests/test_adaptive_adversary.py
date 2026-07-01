"""Adaptive red-team adversaries: attack the *defenses*, not the naive baseline.

C-axis: a defense that only beats a fixed attacker is weak. Since F4, depth is an
**earned** latent estimated from a vouch channel (§10), so an author can neither forge
it nor buy it with more approval breadth.
"""
import numpy as np
import pytest

from chord import ChordConfig, UserKnobs
from chord.loop import Chord
from chord.feed.value import FactorContext, blended_value
from chord.types import Exposure, ExposureSource, Post, Reaction, ReactionKind


def _world(b_approval=0.55, b_vouch=False, b_feature=0.5):
    """G earns cross-cluster vouches; B gets ``b_approval`` broad approval, optional
    vouches, and a possibly-forged depth ``b_feature``. L is partisan filler (clusters)."""
    posts = {
        "G": Post("G", "ag", features={"depth": 0.5}),
        "B": Post("B", "ab", features={"depth": b_feature}),
        "L": Post("L", "al", features={"depth": 0.5}),
    }
    rx, exps = [], []
    for u in range(12):
        left = u < 6
        rx.append(Reaction(f"u{u}", "G", 0.55))
        rx.append(Reaction(f"u{u}", "B", b_approval))
        rx.append(Reaction(f"u{u}", "L", 1.0 if left else -1.0))
        for pid in ("G", "B", "L"):
            exps.append(Exposure(f"u{u}", pid, source=ExposureSource.ORGANIC, propensity=0.5))
        rx.append(Reaction(f"u{u}", "G", 1.0, kind=ReactionKind.VOUCH))     # genuine merit
        exps.append(Exposure(f"u{u}", "G", source=ExposureSource.ORGANIC, propensity=0.5))
        if b_vouch:
            rx.append(Reaction(f"u{u}", "B", 1.0, kind=ReactionKind.VOUCH))
            exps.append(Exposure(f"u{u}", "B", source=ExposureSource.ORGANIC, propensity=0.5))
    return posts, rx, exps


def _value(pid, st, posts, cfg):
    ctx = FactorContext(
        user_id="u0", post=posts[pid],
        b_lcb=st.bridging.b_lcb.get(pid, 0.0),
        result=st.result, divisiveness=st.divisiveness,
        knobs=UserKnobs(M=1.0),
        extras={"depth": st.depth.get(pid, 0.5),         # the EARNED estimate, not the feature
                "depth_reward": cfg.depth_reward, "depth_gate": cfg.depth_gate},
    )
    return blended_value(ctx)


def test_depth_is_earned_not_forged():
    """F4: depth is estimated from the vouch channel, so forging the author feature does
    nothing — genuinely-vouched content earns depth; the bait's forged feature is ignored
    and it stays at the neutral prior, gated below the vouched post."""
    cfg = ChordConfig(d=2, n_clusters=2, mf_iters=40, depth_reward=0.5, depth_gate=0.5)
    posts, rx, exps = _world(b_approval=0.55, b_vouch=False, b_feature=0.95)   # forged
    st = Chord(cfg, seed=0).fit_window(rx, posts, exps)
    dG, dB = st.depth.get("G", 0.5), st.depth.get("B", 0.5)
    print(f"\n[adaptive] earned depth: vouched G={dG:.3f}  forged-feature bait B={dB:.3f}")
    assert dB < 0.6, f"forged depth feature leaked into the estimate (B depth {dB:.3f})"
    assert dG > dB + 0.15, f"cross-cluster vouches should earn depth ({dG:.3f} vs {dB:.3f})"
    assert _value("B", st, posts, cfg) < _value("G", st, posts, cfg)


def test_earned_depth_resists_more_breadth():
    """The bait can't beat the gate by buying more *approval* either: with more breadth
    but no vouches (low earned depth), it stays below the genuinely-vouched post."""
    cfg = ChordConfig(d=2, n_clusters=2, mf_iters=40, depth_reward=0.5, depth_gate=0.6)
    posts, rx, exps = _world(b_approval=0.95, b_vouch=False)   # MORE approval, no vouches
    st = Chord(cfg, seed=0).fit_window(rx, posts, exps)
    print(f"\n[adaptive] breadth bait: b_lcb B={st.bridging.b_lcb['B']:.2f}>G={st.bridging.b_lcb['G']:.2f}"
          f"  depth B={st.depth.get('B',0.5):.2f} G={st.depth.get('G',0.5):.2f}"
          f"  value B={_value('B',st,posts,cfg):.3f} G={_value('G',st,posts,cfg):.3f}")
    assert st.bridging.b_lcb["B"] > st.bridging.b_lcb["G"]      # bait bought more approval
    assert _value("G", st, posts, cfg) > _value("B", st, posts, cfg)   # yet the deep post wins
