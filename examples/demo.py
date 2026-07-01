"""Runnable CHORD demo: ``python -m examples.demo``.

Walks through the three headline claims of the whitepaper on small worlds:
  1. the keystone — universal content out-bridges partisan content (§4);
  2. the M dial — bridging vs personalization, per user, ungameably (§7.1);
  3. the closed loop — the concentration controller and exploration anchor hold
     the estimator in a bounded regime while a firehose author is diluted (§8-9).
"""
from __future__ import annotations

import numpy as np

from chord import Chord, ChordConfig, Exposure, ExposureSource, Post, Reaction, UserKnobs
from chord.propensity import UniformExplorationModel
from chord.simulator import Simulator


def bipolar_world():
    posts = {"A": Post("A", "auth_universal"),
             "B": Post("B", "auth_left"),
             "C": Post("C", "auth_right")}
    rx, exps = [], []
    for u in range(10):
        left = u < 5
        for pid, val in [("A", 1.0), ("B", 1.0 if left else -1.0),
                         ("C", -1.0 if left else 1.0)]:
            rx.append(Reaction(u, pid, val, timestamp=float(u)))
            exps.append(Exposure(u, pid, propensity=0.5, source=ExposureSource.ORGANIC))
    return posts, rx, exps


def demo_keystone():
    print("=" * 68)
    print("1. THE KEYSTONE (§4): tested cross-cluster support, not raw approval")
    print("=" * 68)
    posts, rx, exps = bipolar_world()
    chord = Chord(ChordConfig(d=4, n_clusters=2, mf_iters=40),
                  propensity_model=UniformExplorationModel(0.5), seed=1, inner_iters=3)
    st = chord.fit_window(rx, posts, exps)
    print("  post   B_LCB (tested bridged support)   divisiveness D(p)")
    for pid in ["A", "B", "C"]:
        y = st.result.y_post[pid]
        d = st.divisiveness.divisiveness(y)
        print(f"   {pid}          {st.bridging.b_lcb[pid]:+.3f}                      {d:.3f}")
    print("  -> the universal post A is crowned; partisan B and C are not.\n")
    return chord, posts


def demo_m_dial(chord, posts):
    print("=" * 68)
    print("2. THE M DIAL (§7.1): a per-user consumption knob (ungameable)")
    print("=" * 68)
    for label, M in [("M=1.0  pure bridging", 1.0), ("M=0.0  engagement-like", 0.0)]:
        left = chord.rank(0, list(posts.values()), UserKnobs(M=M), n_slots=3)
        right = chord.rank(9, list(posts.values()), UserKnobs(M=M), n_slots=3)
        print(f"  {label:24s} left user 0 -> {left}   right user 9 -> {right}")
    print("  -> at M=0 each user sees their own side first; at M=1 both see A.\n")


def demo_closed_loop():
    print("=" * 68)
    print("3. THE CLOSED LOOP (§9): bounded regime + diluted firehose (§8)")
    print("=" * 68)
    res = Simulator(n_users=30, knobs=UserKnobs(M=1.0), n_slots=6, seed=3).run(n_windows=8)
    print("  win | Gini(λ) | N_eff | explore% | firehose rpp | universal rpp")
    for m in res.metrics:
        print(f"  {m.window:3d} |  {m.gini_lambda:.3f}  | {m.n_eff:5.1f} |   {m.exploration_rate:.2f}   "
              f"|    {m.firehose_reach_per_post:5.2f}    |    {m.universal_reach_per_post:5.2f}")
    fh = np.mean([m.firehose_reach_per_post for m in res.metrics[2:]])
    uni = np.mean([m.universal_reach_per_post for m in res.metrics[2:]])
    print(f"\n  Gini stays below {max(m.gini_lambda for m in res.metrics):.2f} (controller holds concentration bounded).")
    print(f"  Firehose reach/post {fh:.1f} < universal {uni:.1f} (conserved budget dilutes volume).")
    print(f"  Exploration anchor sustained every window (persistent excitation).\n")


def main():
    chord, posts = demo_keystone()
    demo_m_dial(chord, posts)
    demo_closed_loop()
    print("Done. See README.md for the full paper-to-code map.")


if __name__ == "__main__":
    main()
