"""Prototype experiments testing research-proposed fixes on real data.

These are *not* claims about the shipped algorithm — they evaluate candidate
improvements (from the §5 Sybil-ring and §4 keystone research) against the same
real-data benchmarks the findings were measured on, so we can see whether a
proposed tweak actually moves the numbers before promoting it into ``chord/`` and
the whitepaper. Each experiment reimplements the candidate locally (the core is
untouched) and prints a comparison table.
"""
