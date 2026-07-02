"""Landed Tier 2 mechanisms (E1/E3/E4/E6/E11/E12) — all gated, default off.

Each test drives the mechanism through its core entry point and asserts the property the
experiment demonstrated. See EXPERIMENTS.md for the motivating results.
"""
import numpy as np

from chord.config import ChordConfig
from chord.loop import Chord
from chord.monitor import (
    ConcentrationController,
    empirical_lipschitz,
    gini,
    saturation_depth_prior,
)
from chord.types import Exposure, ExposureSource, Post, Reaction


def _lam(weights):
    w = np.asarray(weights, float)
    w = w / w.sum()
    return {i: float(w[i]) for i in range(len(w))}


def _healthy(rng, n=24):
    return _lam(np.abs(rng.normal(1.0, 0.15, n)))          # near-uniform → low Gini


def _concentrated(rng, n=24):
    w = np.abs(rng.normal(1.0, 0.15, n))
    w[:4] *= 9.0                                            # a few raters dominate → high Gini
    return _lam(w)


# ---- E12: CUSUM controller ----

def test_e12_cusum_fires_on_concentration_drift_not_healthy_noise():
    rng = np.random.default_rng(0)
    cfg = ChordConfig(controller_cusum=True, controller_cusum_warmup=5)

    healthy = ConcentrationController(cfg)
    for _ in range(16):
        healthy.step(_healthy(rng))
    assert not healthy.state.cusum_alarm, "CUSUM should stay silent on healthy noise"

    attacked = ConcentrationController(cfg)
    for _ in range(7):                                      # healthy baseline
        attacked.step(_healthy(rng))
    assert not attacked.state.cusum_alarm
    for _ in range(7):                                      # concentration attack begins
        attacked.step(_concentrated(rng))
    assert attacked.state.cusum_alarm, "CUSUM should fire on the concentration drift"

    # and the level ceiling alone would have missed it (Gini never near 0.6)
    ginis = [h["gini"] for h in attacked.state.history]
    assert max(ginis) < cfg.gini_ceiling


def test_e12_off_by_default_is_level_only():
    # default config: no cusum attribute effect — a healthy run never alarms and the
    # controller is the plain level guard.
    rng = np.random.default_rng(1)
    c = ConcentrationController(ChordConfig())
    for _ in range(12):
        c.step(_concentrated(rng))
    assert not c.state.cusum_alarm      # cusum disabled → flag never set


# ---- E4: residual-whiteness crowning gate ----

def _hidden_axis_world():
    rng = np.random.default_rng(0)
    nu = 48
    a0 = rng.normal(0, 1, nu)
    a1 = rng.normal(0, 1, nu)                 # the HIDDEN axis (beyond fitted d=1)
    posts, rx, exps = {}, [], []
    for j in range(16):                        # posts driven by axis 0 → d=1 learns it
        pol = rng.choice([-1.0, 1.0])
        posts[f"p{j}"] = Post(f"p{j}", "auth")
        for u in range(nu):
            rx.append(Reaction(f"u{u}", f"p{j}", float(np.tanh(pol * a0[u] + rng.normal(0, 0.3)))))
    posts["BRIDGE"] = Post("BRIDGE", "ab")     # genuine bridge: broad approval, no axis
    posts["HIDDEN"] = Post("HIDDEN", "ah")     # appears to bridge (positive mean) but hides
    for u in range(nu):                        # a divide along axis 1 in the residuals
        rx.append(Reaction(f"u{u}", "BRIDGE", float(np.clip(0.6 + rng.normal(0, 0.2), -1, 1))))
        rx.append(Reaction(f"u{u}", "HIDDEN",
                           float(np.clip(0.55 + 0.85 * a1[u] + rng.normal(0, 0.2), -1, 1))))
    for r in rx:
        exps.append(Exposure(r.user_id, r.post_id, source=ExposureSource.ORGANIC, propensity=0.5))
    return posts, rx, exps


def test_e4_whiteness_gate_flags_hidden_divide_passes_genuine_bridge():
    posts, rx, exps = _hidden_axis_world()
    cfg = ChordConfig(d=1, n_clusters=2, mf_iters=60, whiteness_gate=True)
    st = Chord(cfg, seed=0).fit_window(rx, posts, exps)
    wI, wp = st.residual_whiteness.get("HIDDEN", (0.0, 1.0))
    bI, bp = st.residual_whiteness.get("BRIDGE", (0.0, 1.0))
    print(f"\n[E4] HIDDEN Moran's I={wI:+.3f} p={wp:.3f}; BRIDGE I={bI:+.3f} p={bp:.3f}; "
          f"HIDDEN B_LCB={st.bridging.b_lcb['HIDDEN']:.3f}")
    # the hidden-axis divide is flagged (significantly non-white residuals) and demoted...
    assert wp < 0.05 and wI > 0.0, "hidden-axis divide should be flagged"
    assert st.bridging.b_lcb["HIDDEN"] < st.bridging.b_lcb["BRIDGE"], "flagged post demoted below the genuine bridge"
    # ...while the genuine bridge passes (white residuals)
    assert bp > 0.05, "genuine bridge should pass (white residuals)"


def test_e4_gate_off_by_default_computes_nothing():
    posts, rx, exps = _hidden_axis_world()
    st = Chord(ChordConfig(d=1, n_clusters=2, mf_iters=40), seed=0).fit_window(rx, posts, exps)
    assert st.residual_whiteness == {}


# ---- E3: amplification collar ----

def test_e3_collar_throttles_reach_beyond_tested_audience():
    # OVER-reached post: 40 exposures, only 3 reactions (tested) -> reach >> κ·tested.
    # WELL-tested post: 12 exposures, 12 reactions -> reach <= κ·tested.
    rng = np.random.default_rng(0)
    posts = {"OVER": Post("OVER", "ao"), "WELL": Post("WELL", "aw"), "F": Post("F", "af")}
    rx, exps = [], []
    for u in range(20):                                   # cluster structure via a partisan post
        rx.append(Reaction(f"u{u}", "F", 1.0 if u < 10 else -1.0))
        exps.append(Exposure(f"u{u}", "F", source=ExposureSource.ORGANIC, propensity=0.5))
    for u in range(40):                                   # OVER: 40 exposures
        exps.append(Exposure(f"v{u}", "OVER", source=ExposureSource.ORGANIC, propensity=0.5))
    for u in range(3):                                    # ...but only 3 tested it
        rx.append(Reaction(f"v{u}", "OVER", 0.8))
    for u in range(12):                                   # WELL: 12 exposures, all 12 tested
        rx.append(Reaction(f"w{u}", "WELL", 0.8))
        exps.append(Exposure(f"w{u}", "WELL", source=ExposureSource.ORGANIC, propensity=0.5))
    cfg = ChordConfig(d=2, n_clusters=2, mf_iters=30, amplification_collar=True, collar_kappa=4.0)
    st = Chord(cfg, seed=0).fit_window(rx, posts, exps)
    print(f"\n[E3] collar: OVER={st.collar.get('OVER'):.2f}  WELL={st.collar.get('WELL'):.2f}")
    assert st.collar["OVER"] < 0.5, "over-reached post should be throttled"
    assert st.collar["WELL"] == 1.0, "well-tested post should not be throttled"


# ---- E6: off-policy recycling verification ----

def test_e6_offpolicy_verify_denies_the_recycling_farmer():
    # genuine under-served user prefers ε content; farmer acts dissatisfied but doesn't.
    rng = np.random.default_rng(0)
    posts = {f"p{j}": Post(f"p{j}", f"a{j%3}") for j in range(10)}
    rx, exps = [], []
    # background raters for structure
    for u in range(16):
        for j in range(10):
            v = 0.6 if (j % 2 == 0) == (u < 8) else -0.6
            rx.append(Reaction(f"u{u}", f"p{j}", v))
            exps.append(Exposure(f"u{u}", f"p{j}", source=ExposureSource.ORGANIC, propensity=0.5))
    # genuine under-served G: low value on organic feed, HIGH on ε items
    for j in range(4):
        rx.append(Reaction("G", f"p{j}", -0.5))
        exps.append(Exposure("G", f"p{j}", source=ExposureSource.ORGANIC, propensity=0.5))
    for j in range(4, 8):
        rx.append(Reaction("G", f"p{j}", 0.8))
        exps.append(Exposure("G", f"p{j}", source=ExposureSource.EXPLORATION, propensity=0.1))
    # farmer F: low value on organic (acts dissatisfied), also low on ε (no real preference)
    for j in range(4):
        rx.append(Reaction("F", f"p{j}", -0.5))
        exps.append(Exposure("F", f"p{j}", source=ExposureSource.ORGANIC, propensity=0.5))
    for j in range(4, 8):
        rx.append(Reaction("F", f"p{j}", -0.5))
        exps.append(Exposure("F", f"p{j}", source=ExposureSource.EXPLORATION, propensity=0.1))

    def lam_eff(verify):
        cfg = ChordConfig(d=2, n_clusters=2, mf_iters=30, recycling_offpolicy_verify=verify)
        return Chord(cfg, seed=0).fit_window(rx, posts, exps).rater_lambda_eff
    off, on = lam_eff(False), lam_eff(True)
    print(f"\n[E6] farmer λ_eff off={off['F']:.4f} on={on['F']:.4f}; genuine off={off['G']:.4f} on={on['G']:.4f}")
    # with verification the farmer's boost is withdrawn, the genuine user's kept
    assert on["F"] < off["F"], "off-policy verify should withdraw the farmer's recycling boost"
    assert on["G"] >= off["G"] * 0.95, "the genuine under-served user should keep their boost"


# ---- E1: measured performativity (empirical Lipschitz) ----

def test_e1_empirical_lipschitz_recovers_known_sensitivity():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(0, 1, 40))                    # input (ranking-change proxy)
    y = 2.0 * x + rng.normal(0, 1e-3, 40)                  # output = 2·input (Lipschitz 2)
    assert abs(empirical_lipschitz(y, x) - 2.0) < 0.1
    # a flatter map has a smaller Lipschitz — the signal the controller would hold down
    y2 = 0.4 * x + rng.normal(0, 1e-3, 40)
    assert empirical_lipschitz(y2, x) < empirical_lipschitz(y, x)


# ---- E11: saturation-trajectory depth prior ----

def test_e11_saturation_trajectory_separates_bait_from_depth():
    rng = np.random.default_rng(0)
    def traj(halflife):
        return [np.exp(-t / halflife) + rng.normal(0, 0.02) for t in range(12)]
    bait = saturation_depth_prior(traj(1.5))              # fast saturation
    depth = saturation_depth_prior(traj(7.0))             # slow burn
    print(f"\n[E11] depth prior: bait={bait:.3f}  depth={depth:.3f}")
    assert depth > bait + 0.3, "slow-burn depth should earn a higher prior than fast-saturating bait"
    assert 0.0 <= bait <= 1.0 and 0.0 <= depth <= 1.0
