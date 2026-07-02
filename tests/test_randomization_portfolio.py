"""E-meta: the randomization portfolio (§13) — allocating the scarce ε budget by value.

The ε-slice serves many masters (identification, cold-start, stability + E2 calibration,
E5/E12 audit, E1 probe, E3 re-certification). A floored bandit that water-fills the budget
toward the currently-highest-value demand beats a uniform split under a fixed budget — and,
crucially, keeps a floor on every arm so ε stays a floored invariant.
"""
import numpy as np

from chord.economy import RandomizationPortfolio


ARMS = ["audition", "calibration", "audit", "probe"]


def test_allocation_is_a_distribution_with_a_floor():
    p = RandomizationPortfolio(ARMS, floor_frac=0.4)
    p.observe("audit", 10.0)                         # one arm looks very valuable
    a = p.allocate()
    assert abs(sum(a.values()) - 1.0) < 1e-9         # a distribution
    floor = 0.4 / len(ARMS)
    assert all(v >= floor - 1e-9 for v in a.values())  # every arm keeps its floor (ε invariant)
    assert a["audit"] == max(a.values())             # value-weighted toward the valuable arm


def test_pure_floor_is_the_uniform_split():
    p = RandomizationPortfolio(ARMS, floor_frac=1.0)
    p.observe("audit", 100.0)                        # learning can't tilt a pure-floor policy
    a = p.allocate()
    assert all(abs(v - 0.25) < 1e-9 for v in a.values())


def _value(arm, t):
    if arm == "audition":   return 2.5 * np.exp(-t / 6.0) + 0.2   # cold-start, decays
    if arm == "calibration": return 0.8                          # steady
    if arm == "audit":      return 2.2 if 12 <= t < 20 else 0.05  # spikes during an attack
    return 0.3                                                    # probe: low


def test_portfolio_beats_uniform_under_shifting_needs():
    rng = np.random.default_rng(0)
    kappa = 0.25
    captured = lambda share, v: v * (1.0 - np.exp(-share / kappa))   # diminishing returns

    def run(port, learn):
        total, audit_attack = 0.0, 0.0
        for t in range(30):
            alloc = port.allocate()
            for arm in ARMS:
                v = _value(arm, t)
                total += captured(alloc[arm], v)
                if arm == "audit" and 12 <= t < 20:
                    audit_attack += captured(alloc[arm], v)
                if learn:
                    port.observe(arm, v + rng.normal(0, 0.15))
        return total, audit_attack

    uni, uni_aud = run(RandomizationPortfolio(ARMS, floor_frac=1.0), learn=False)
    port, port_aud = run(RandomizationPortfolio(ARMS, floor_frac=0.4, lr=0.4, temperature=0.6),
                         learn=True)
    print(f"\n[E-meta] captured value: uniform={uni:.2f} portfolio={port:.2f} "
          f"(+{100*(port-uni)/uni:.0f}%); attack-window audit uniform={uni_aud:.2f} port={port_aud:.2f}")
    assert port > uni                                # value-weighting beats uniform
    assert port_aud > uni_aud                        # and redirects to audit during the attack
