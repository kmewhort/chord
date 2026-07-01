"""Performativity: does the incentive shape the content ecosystem? (§9.2, App C.4)

Authors are strategic — they hill-climb (a (1+1)-ES in content-direction space) on
the reach the ranker gives them. This tests the performative claim: the *reward
gradient the ranker exposes* changes what authors produce. Under engagement, reach
is highest for in-group content, so partisan authors grow **more** extreme over
time; under CHORD, bridged value is not served by extremity, so they do **not**.

Measured on the partisan authors' extremity (|style · main-axis|), averaged over
seeds so it is the systematic drift, not one noisy trajectory.
"""
import numpy as np
import pytest

from chord.simulator import Simulator

SEEDS = (1, 2, 3, 4, 5)
PERFORMATIVITY = 0.4
WINDOWS = 12


def _extremity_trajectory(ranker):
    traj = []
    for seed in SEEDS:
        sim = Simulator(n_users=36, n_slots=6, seed=seed, adaptive_authors=False,
                        performativity=PERFORMATIVITY)
        r = sim.run(ranker, n_windows=WINDOWS)
        traj.append(r.series("partisan_extremity"))
    return np.array(traj).mean(0)


@pytest.fixture(scope="module")
def trajectories():
    t = {"chord": _extremity_trajectory("chord"),
         "engagement": _extremity_trajectory("engagement")}
    print("\n[sim perf] partisan extremity |style·axis0| (window0 -> last):")
    for name, tr in t.items():
        print(f"[sim perf]   {name:<11} {tr[0]:.3f} -> {tr[-1]:.3f}  (Δ {tr[-1]-tr[0]:+.3f})")
    return t


def test_engagement_incentive_drives_authors_to_polarize(trajectories):
    # Under engagement, strategic partisan authors become MORE extreme over time.
    eng = trajectories["engagement"]
    assert eng[-1] > eng[0] + 0.01, (
        f"expected engagement to entrench partisan extremity ({eng[0]:.3f} -> {eng[-1]:.3f})"
    )


def test_chord_incentive_does_not_reward_extremity(trajectories):
    # Under CHORD the same strategic authors do NOT drift more extreme — the reward
    # gradient does not point at the poles.
    chord, eng = trajectories["chord"], trajectories["engagement"]
    chord_drift = chord[-1] - chord[0]
    eng_drift = eng[-1] - eng[0]
    assert chord_drift < eng_drift - 0.05, (
        f"CHORD drift ({chord_drift:+.3f}) should be well below engagement's "
        f"({eng_drift:+.3f}) — the incentive is not shaping content away from extremity"
    )
    assert chord[-1] < eng[-1], (
        f"CHORD's ecosystem ({chord[-1]:.3f}) ended at least as polarized as "
        f"engagement's ({eng[-1]:.3f})"
    )
