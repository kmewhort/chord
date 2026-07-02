"""Budget recursion refinements (§8, Fable review) — all gated, default off.

The replenishment is a floored, capped replicator update. These test the four issues:
#1 memoryless cadence penalty (carry term γ), #2 the ηΦ̄ bifurcation (diagnostic),
#4 procyclical aggregate issuance (share-based pool). #3 (streaming credit) is a design
direction documented in §8/§13, not a code change here.
"""
import dataclasses

from chord.config import ChordConfig
from chord.economy import AuthorBudgetLedger


def test_default_recursion_is_unchanged():
    # budget_memory=0, share_based=False → original B_{t+1}=clip(B_0+η·ΣΦE)
    cfg = ChordConfig(budget_B0=2.0, budget_eta=0.5, budget_max=100.0)
    L = AuthorBudgetLedger(cfg)
    L.replenish({"p": 0.6}, {"p": 4.0}, {"p": "a"})
    assert L.budget("a") == 2.0 + 0.5 * 0.6 * 4.0


def _build_then_quiet(memory):
    cfg = ChordConfig(budget_B0=2.0, budget_eta=0.5, budget_max=10.0, budget_memory=memory)
    L = AuthorBudgetLedger(cfg)
    for _ in range(5):                                   # earn steadily
        L.replenish({"p": 0.6}, {"p": L.budget("A")}, {"p": "A"})
    built = L.budget("A")
    L.replenish({}, {}, {})                              # one quiet window (posted nothing)
    return built, L.budget("A")


def test_1_memory_carry_survives_a_quiet_window():
    built0, quiet0 = _build_then_quiet(0.0)
    assert quiet0 == 2.0                                 # memoryless: reset to the floor
    built, quiet = _build_then_quiet(0.7)
    # with memory the earned standing above the floor is (mostly) carried, not reset
    assert quiet > 2.0 + 0.5 * (built - 2.0)


def test_4_share_based_issuance_is_not_procyclical():
    cfg = ChordConfig(budget_B0=2.0, budget_max=1000.0, budget_share_based=True,
                      budget_aggregate_factor=2.0)

    def total(phi, reach):
        L = AuthorBudgetLedger(cfg)
        L.replenish({f"p{i}": phi for i in range(10)},
                    {f"p{i}": reach for i in range(10)},
                    {f"p{i}": f"a{i}" for i in range(10)})
        return sum(L.budgets.values())

    normal = total(0.3, 1.0)
    news_day = total(0.8, 3.0)                           # engagement up across the board
    assert abs(normal - news_day) < 1e-6                 # aggregate issuance is conserved
    assert abs(normal - 2.0 * 10 * 2.0) < 1e-6           # == factor · n · B_0 (fixed pool)


def test_4_share_based_still_allocates_by_relative_strength():
    cfg = ChordConfig(budget_B0=2.0, budget_max=1000.0, budget_share_based=True,
                      budget_aggregate_factor=2.0)
    L = AuthorBudgetLedger(cfg)
    L.replenish({"hi": 0.9, "lo": 0.1}, {"hi": 5.0, "lo": 5.0},
                {"hi": "A", "lo": "B"})
    assert L.budget("A") > L.budget("B")                 # more relative strength → more budget


def test_3_streaming_credit_flows_as_evidence_arrives():
    cfg = ChordConfig(budget_B0=2.0, budget_eta=1.0, budget_max=1e6,
                      budget_streaming_credit=True)
    L = AuthorBudgetLedger(cfg)
    increments, prev = [], 0.0
    for conf in (0.3, 0.6, 1.0):                          # posterior tightens over windows
        L.replenish({"p": 0.6}, {"p": 10.0}, {"p": "A"}, {"p": conf})
        increments.append(L._credited["p"] - prev)
        prev = L._credited["p"]
    # credit flows across windows (not one lump at saturation), rising as evidence accrues
    assert all(i >= 0 for i in increments) and increments[-1] > increments[0]
    # and totals exactly to the batch credit for the fully-confident post: Φ·1·E_cumulative
    assert abs(L._credited["p"] - 0.6 * 1.0 * 30.0) < 1e-6
    # a shallow-but-bursty post (never gains confidence) is not over-credited at a boundary
    L2 = AuthorBudgetLedger(cfg)
    L2.replenish({"q": 0.9}, {"q": 100.0}, {"q": "B"}, {"q": 0.05})   # huge burst, no evidence
    assert L2._credited["q"] == 0.9 * 0.05 * 100.0        # credited only at its low confidence


def test_2_replicator_gain_diagnostic():
    cfg = ChordConfig(budget_eta=0.5, budget_memory=0.2)
    L = AuthorBudgetLedger(cfg)
    gain = L.replicator_gain({"p": 0.6}, {"p": 10.0}, {"p": "a"})
    assert abs(gain - (0.2 + 0.5 * 0.6)) < 1e-9          # γ + η·Φ̄
    # a supercritical gain (≥1) is the runaway/phase-transition regime
    hot = dataclasses.replace(cfg, budget_eta=2.0)
    assert AuthorBudgetLedger(hot).replicator_gain({"p": 0.6}, {"p": 1.0}, {"p": "a"}) > 1.0
