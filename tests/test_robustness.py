"""Robustness sweeps: do the headline claims survive config/seed variation?

Most claims elsewhere are single-config point estimates. These sweep a grid of
(d, n_clusters, aggregator) × seeds and require the claim to hold across (nearly) all
of it — turning "it worked once" into "it's robust." (The welfare claim proves robust;
see test_dynamics / the property tests for the claims that are init-fragile.)
"""
import itertools

import numpy as np
import pytest

from chord.config import ChordConfig
from chord.simulator import Simulator

GRID = list(itertools.product((2, 3), (2, 3), ("nash", "min")))  # d, n_clusters, agg
SEEDS = (1, 2, 3)


def _welfare(ranker, d, ncl, agg, seed):
    cfg = ChordConfig(d=d, n_clusters=ncl, mf_iters=25, budget_B0=2.0, budget_max=6.0,
                      bridging_aggregator=agg)
    sim = Simulator(config=cfg, n_users=36, n_slots=6, seed=seed, adaptive_authors=False)
    r = sim.run(ranker, n_windows=7)
    return r.tail("true_value", 4), r.tail("divisiveness", 4)


@pytest.fixture(scope="module")
def sweep():
    tv_wins, div_wins, tv_margins = [], [], []
    for d, ncl, agg in GRID:
        ct, et, cd, ed = [], [], [], []
        for s in SEEDS:
            c = _welfare("chord", d, ncl, agg, s)
            e = _welfare("engagement", d, ncl, agg, s)
            ct.append(c[0]); et.append(e[0]); cd.append(c[1]); ed.append(e[1])
        cm, em = np.nanmean(ct), np.nanmean(et)
        tv_wins.append(cm > em)
        tv_margins.append(cm - em)
        div_wins.append(np.nanmean(cd) < np.nanmean(ed))
    out = dict(n=len(GRID), tv=sum(tv_wins), div=sum(div_wins),
               min_margin=float(np.min(tv_margins)))
    print(f"\n[robustness] across {out['n']} configs: CHORD beats engagement on "
          f"true_value {out['tv']}/{out['n']}, divisiveness {out['div']}/{out['n']}; "
          f"min true_value margin={out['min_margin']:+.3f}")
    return out


def test_chord_beats_engagement_across_configs(sweep):
    # The welfare win must be robust, not cherry-picked: hold in (almost) every config.
    assert sweep["tv"] >= sweep["n"] - 1, (
        f"CHORD>engagement on true_value held in only {sweep['tv']}/{sweep['n']} configs"
    )
    assert sweep["div"] >= sweep["n"] - 1, (
        f"CHORD<engagement on divisiveness held in only {sweep['div']}/{sweep['n']} configs"
    )


def test_true_value_margin_is_positive_everywhere(sweep):
    # Even the worst config keeps a positive margin (no near-ties masking a loss).
    assert sweep["min_margin"] > 0.0, (
        f"some config had a non-positive true_value margin ({sweep['min_margin']:+.3f})"
    )
