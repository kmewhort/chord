"""Content and author agents that adapt to the incentive (Appendix C.4).

To test whether the conserved budget actually suppresses firehosing and whether
"bridging-bait" emerges, author-agents generate posts with a latent loading
``y_p`` and choose *how much* to post based on the reach they realize. Two
archetypes matter:

* a **universal** author whose posts sit near the origin of opinion space (broad
  appeal, low divisiveness) — should thrive under CHORD;
* a **partisan** author whose posts load heavily on one pole — should be capped;
* a **firehose** author who posts a high volume of mediocre content — should be
  diluted by the conserved budget (§8).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..types import Post


@dataclass
class AuthorAgent:
    """A content-producing agent that adapts posting volume to realized reach."""

    id: int
    # Mean loading direction of this author's content in opinion space.
    style: np.ndarray
    # Spread of individual posts around the style.
    spread: float = 0.3
    # How strongly the author responds to realized reach when setting volume.
    adaptivity: float = 1.0
    base_volume: int = 1
    max_volume: int = 8
    # running estimate of realized reach-per-post (updated by the engine)
    realized_reach: float = 1.0

    def generate(self, window: int, rng: np.random.Generator) -> List[Post]:
        """Produce this window's posts; volume adapts to realized reach.

        Engagement logic makes volume rational (each post an independent lottery
        ticket). A CHORD-aware author instead sees reach *per post* fall when they
        firehose, because the conserved budget spreads thin — so ``adaptivity``
        pushes volume up only while marginal reach stays high.
        """
        volume = self.base_volume
        if self.adaptivity > 0:
            # more reach -> post a bit more; the budget mechanism is what makes
            # this self-limiting rather than runaway.
            volume = int(round(self.base_volume + self.adaptivity * self.realized_reach))
        volume = max(1, min(self.max_volume, volume))

        posts: List[Post] = []
        for k in range(volume):
            loading = self.style + rng.normal(0, self.spread, size=self.style.shape)
            pid = f"a{self.id}_w{window}_{k}"
            posts.append(
                Post(id=pid, author_id=self.id, created_at=float(window),
                     features={"norm": float(np.linalg.norm(loading))})
            )
            # stash the true loading on the post for the response model
            _TRUE_LOADING[pid] = loading
        return posts


# Side table of ground-truth post loadings (the simulator's hidden truth).
_TRUE_LOADING: dict = {}


def true_loading(post_id) -> Optional[np.ndarray]:
    return _TRUE_LOADING.get(post_id)


def reset_truth() -> None:
    _TRUE_LOADING.clear()


def make_authors(d: int = 2, seed: int = 0) -> List[AuthorAgent]:
    """A standard cast: universal, two partisans, and a firehose (Appendix C.4)."""
    rng = np.random.default_rng(seed)
    authors = [
        AuthorAgent(id=1000, style=np.zeros(d), spread=0.15, adaptivity=0.5,
                    base_volume=1, max_volume=4),                      # universal
        AuthorAgent(id=1001, style=_pole(d, +1, 2.0), spread=0.2, adaptivity=0.5,
                    base_volume=1, max_volume=4),                      # partisan +
        AuthorAgent(id=1002, style=_pole(d, -1, 2.0), spread=0.2, adaptivity=0.5,
                    base_volume=1, max_volume=4),                      # partisan -
        AuthorAgent(id=1003, style=np.zeros(d), spread=1.5, adaptivity=3.0,
                    base_volume=3, max_volume=8),                      # firehose
    ]
    return authors


def _pole(d: int, sign: int, mag: float) -> np.ndarray:
    v = np.zeros(d)
    v[0] = sign * mag
    return v
