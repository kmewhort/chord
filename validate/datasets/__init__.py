"""Adapters mapping each public dataset onto CHORD's core types.

Every module here exposes a ``load(...)`` (or dataset-specific loader) that turns
raw files under ``validate/data/<name>/`` into the lingua franca of :mod:`chord.types`
(:class:`~chord.types.Reaction`, :class:`~chord.types.Post`,
:class:`~chord.types.Exposure`) or into dense rating matrices where the dataset is
naturally matrix-shaped (Coat, MovieLens). Adapters never download; use
:mod:`validate.fetch` for that.
"""
