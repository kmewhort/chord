"""Landed Tier 2 mechanisms (E1/E3/E4/E6/E11/E12) — all gated, default off.

Each test drives the mechanism through its core entry point and asserts the property the
experiment demonstrated. See EXPERIMENTS.md for the motivating results.
"""
import numpy as np

from chord.config import ChordConfig
from chord.loop import Chord
from chord.monitor import ConcentrationController, gini
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
