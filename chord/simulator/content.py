"""Content and author agents that adapt to the incentive (Appendix C.4).

Author-agents generate posts with a latent loading ``y_p`` plus two *hidden*
attributes that make the world non-circular (so the estimator cannot trivially
recover its own assumptions):

* ``toxicity`` ∈ [0,1] — an affective-polarization channel roughly orthogonal to
  genuine value: toxic posts draw *more* reactions (engagement pull) and sharpen
  the in-group/out-group divide, but are not what a healthy feed should amplify.
* ``quality`` ∈ [0,1] — genuine value, which barely moves reactions (that is the
  whole problem: engagement ≠ value). A post's *true* worth is quality × how
  broadly it is received.

Archetypes (Appendix C.4, extended):

* **universal** — near-origin loading, high quality, low toxicity (should thrive);
* **partisan** ± — heavy one-pole loading (should be capped by B_LCB / divisiveness);
* **firehose** — high volume of mediocre content (should be diluted by the budget, §8);
* **toxic partisan** — one-pole loading + high toxicity (the engagement magnet);
* **bridging-bait** — near-origin loading (broad mild appeal) but *low* quality —
  shallow universal content that games B_LCB (the §10 Goodhart concern).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..types import Post


@dataclass
class PostTruth:
    """The simulator's hidden ground truth for one post."""
    loading: np.ndarray     # true y_p in R^{d_true}
    toxicity: float         # affective-polarization channel in [0,1]
    quality: float          # genuine value in [0,1]


@dataclass
class AuthorAgent:
    """A content-producing agent that adapts posting volume to realized reach."""

    id: int
    # Mean loading direction of this author's content in opinion space (R^{d_true}).
    style: np.ndarray
    spread: float = 0.3
    # How strongly the author responds to realized reach when setting volume.
    adaptivity: float = 1.0
    base_volume: int = 1
    max_volume: int = 8
    # hidden-attribute propensities
    toxicity: float = 0.1
    quality: float = 0.7
    # performativity: how fast the author moves its *content direction* toward
    # whatever the ranker rewarded (0 = fixed style; the §9.2 stability knob). It
    # sets the (1+1)-ES trial step, so bigger values chase harder — and can
    # destabilize the loop, which is the point of the phase-transition sweep.
    performativity: float = 0.0
    # running estimate of realized reach-per-post (updated by the engine)
    realized_reach: float = 1.0
    label: str = ""
    # (1+1)-ES state: the committed style, the current trial, and its reach-to-beat
    _committed: Optional[np.ndarray] = None
    _best_reach: float = -1.0

    def generate(self, window: int, rng: np.random.Generator) -> List[Post]:
        """Produce this window's posts; volume adapts to realized reach.

        A CHORD-aware author sees reach *per post* fall when they firehose (the
        conserved budget spreads thin), so ``adaptivity`` pushes volume up only
        while marginal reach stays high — self-limiting rather than runaway.
        """
        volume = self.base_volume
        if self.adaptivity > 0:
            volume = int(round(self.base_volume + self.adaptivity * self.realized_reach))
        volume = max(1, min(self.max_volume, volume))

        # (1+1)-ES trial: perturb the committed style before producing this window's
        # posts; adapt_style() below keeps it only if reach improved.
        if self.performativity > 0.0:
            if self._committed is None:
                self._committed = self.style.copy()
            step = self.performativity * 1.2
            self.style = self._committed + rng.normal(0, step, size=self._committed.shape)

        posts: List[Post] = []
        for k in range(volume):
            loading = self.style + rng.normal(0, self.spread, size=self.style.shape)
            tox = float(np.clip(self.toxicity + rng.normal(0, 0.1), 0.0, 1.0))
            qual = float(np.clip(self.quality + rng.normal(0, 0.15), 0.0, 1.0))
            pid = f"a{self.id}_w{window}_{k}"
            # `depth` is a *noisy* observable proxy of quality (CHORD's θ_depth
            # defense against shallow bait sees this, never the true quality).
            depth = float(np.clip(qual + rng.normal(0, 0.2), 0.0, 1.0))
            posts.append(
                Post(id=pid, author_id=self.id, created_at=float(window),
                     features={"norm": float(np.linalg.norm(loading)), "depth": depth})
            )
            _POST_TRUTH[pid] = PostTruth(loading=loading, toxicity=tox, quality=qual)
        return posts

    def adapt_style(self, realized_reach: float) -> None:
        """(1+1)-ES accept/reject: keep this window's trial style iff reach improved.

        The author hill-climbs realized reach in *content-direction* space. Under a
        bridging ranker, reach is highest for broadly-received (near-origin) content,
        so authors migrate toward bridging; under engagement, reach is highest for
        in-group content, so they polarize. ``performativity`` sets the trial step:
        larger steps chase harder and can overshoot into oscillation (§9.2).
        """
        if self.performativity <= 0.0 or self._committed is None:
            return
        if realized_reach >= self._best_reach:
            self._committed = self.style.copy()   # accept the trial
            self._best_reach = realized_reach
        else:
            self.style = self._committed.copy()   # reject; snap back to committed


# Side table of ground-truth post attributes (the simulator's hidden truth).
_POST_TRUTH: Dict[str, PostTruth] = {}


def true_post(post_id) -> Optional[PostTruth]:
    return _POST_TRUTH.get(post_id)


def true_loading(post_id) -> Optional[np.ndarray]:
    t = _POST_TRUTH.get(post_id)
    return None if t is None else t.loading


def reset_truth() -> None:
    _POST_TRUTH.clear()


def make_authors(d: int = 2, seed: int = 0, d_true: Optional[int] = None,
                 include_toxic_and_bait: bool = True,
                 performativity: float = 0.0) -> List[AuthorAgent]:
    """The standard cast (Appendix C.4). ``d_true`` matches the population's true dim.

    ``performativity`` (the §9.2 knob) is applied to every author's style-chasing
    rate; 0 keeps content styles fixed.
    """
    d_true = d_true if d_true is not None else d
    authors = [
        AuthorAgent(id=1000, style=np.zeros(d_true), spread=0.15, adaptivity=0.5,
                    base_volume=1, max_volume=4, toxicity=0.05, quality=0.85,
                    label="universal"),
        AuthorAgent(id=1001, style=_pole(d_true, +1, 2.0), spread=0.2, adaptivity=0.5,
                    base_volume=1, max_volume=4, toxicity=0.2, quality=0.6,
                    label="partisan+"),
        AuthorAgent(id=1002, style=_pole(d_true, -1, 2.0), spread=0.2, adaptivity=0.5,
                    base_volume=1, max_volume=4, toxicity=0.2, quality=0.6,
                    label="partisan-"),
        AuthorAgent(id=1003, style=np.zeros(d_true), spread=1.5, adaptivity=3.0,
                    base_volume=3, max_volume=8, toxicity=0.3, quality=0.35,
                    label="firehose"),
    ]
    if include_toxic_and_bait:
        authors += [
            AuthorAgent(id=1004, style=_pole(d_true, +1, 1.6), spread=0.25, adaptivity=0.5,
                        base_volume=1, max_volume=5, toxicity=0.9, quality=0.4,
                        label="toxic_partisan"),
            AuthorAgent(id=1005, style=np.zeros(d_true), spread=0.15, adaptivity=0.5,
                        base_volume=1, max_volume=5, toxicity=0.05, quality=0.1,
                        label="bridging_bait"),
        ]
    if performativity:
        for a in authors:
            a.performativity = performativity
    return authors


def _pole(d: int, sign: int, mag: float) -> np.ndarray:
    v = np.zeros(d)
    v[0] = sign * mag
    return v
