"""Retroactive validation suite (whitepaper Appendix C).

This is a *second*, opt-in test suite that sits alongside ``tests/``. Where
``tests/`` proves the code is internally correct on synthetic worlds, ``validate/``
checks that CHORD's static components reproduce the whitepaper's claims on **real
public datasets** — the layer-to-dataset mapping of Appendix C.1:

    | CHORD layer                     | Dataset(s)              | Module                        |
    |---------------------------------|-------------------------|-------------------------------|
    | Keystone / clusters (§4)        | Polis, Community Notes  | test_polis, test_community_notes |
    | Rater weighting / trust (§5)    | SNAP signed nets, CN    | test_signed_nets, test_community_notes |
    | Propensity / MNAR (§6)          | Coat, MovieLens         | test_coat, test_movielens     |

The dynamic feedback loop (§9) is deliberately *not* here — a fixed archive cannot
respond to the ranker's own allocations, so it is exercised by the simulator in
``tests/test_simulator.py`` instead (Appendix C.4).

The datasets are not part of the default suite: ``pyproject.toml`` pins
``testpaths = ["tests"]`` so ``pytest -q`` stays offline and fast. Run the
validation suite explicitly::

    pip install -e '.[validate]'
    python -m validate.fetch all      # download into validate/data (Git LFS)
    pytest validate/                  # tests self-skip if data is absent
"""
