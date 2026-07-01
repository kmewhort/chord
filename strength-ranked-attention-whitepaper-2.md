# CHORD

### Cross-cluster Harmonized Optimization of Reception and Dissonance

*A bridging, attention-economy feed-ranking algorithm for federated social networks —
working draft*

---

## Abstract

We describe **CHORD** (Cross-cluster Harmonized Optimization of Reception and Dissonance), a
feed-ranking algorithm that inverts the incentive of engagement-based
ranking. Where engagement ranking rewards content that maximizes reactions — and
therefore structurally favors divisive, high-variance "split-decision" content — CHORD
ranks content by a **strength score** that rewards broad support net of
divisiveness. The scoring core is built on the bridging-based ranking literature — the
matrix-factorization approach deployed in Community Notes and Polis — and takes from the
Ethelo collective-decision engine its objective: preferring broadly-supported, low-dissonance
outcomes over higher-average but polarizing ones. Around that core we add
three attention-economy mechanisms: a
**quality-weighted rater model** so that a discriminating reaction counts more than a
reflexive one; a **conserved, strength-replenished author visibility budget** so that
high-volume posting dilutes rather than multiplies reach; and a **commons-funded
exploration pool** that gives unproven content from newcomers a fair audition. CHORD is
structured as a valuation-and-allocation layer between retrieval and presentation,
factored into ports so it can run standalone on a single Mastodon instance or dock into a
richer host. We ground the estimator's stability in performative-prediction and
two-timescale stochastic-approximation theory, correct a rater-weighting pathology
identified in deployed bridging systems, and are explicit about what remains genuinely
open — chiefly the propensity model on which identifiability rests.

---

## 1. Motivation

Recommender systems on social media are best understood not as "content distribution" but
as **attention allocation**: from a firehose of candidates they fill a scarce number of
attention slots, and in doing so they set the incentives of the attention economy
[Ovadya & Thorburn 2023]. Most such systems optimize measurable engagement — clicks,
reactions, replies, dwell — which is often a good proxy for what people want but also
rewards content that is sensational, outrageous, or divisive. Outrage and ratio-bait are
high-*variance* reception events: some love them, some hate them, and that split is what
drives the reactions engagement ranking chases.

Two independent lines of work converge on the opposite objective. The **bridging-based
ranking** literature, motivated by depolarization, ranks content by whether it earns
support *across* a population's divides; it has a governance-tool lineage (Polis) and one
deployed instance at scale (X's Community Notes). The **Ethelo** collective-decision engine,
motivated by fair group decisions, scores an outcome by average support penalized for
**dissonance** (the variance of support across participants) and will rank a
lower-average-but-unified outcome above a higher-average-but-polarizing one. These are the
same objective reached from two directions, and it is precisely a ranker that penalizes
what engagement rewards.

This paper takes that objective as its starting point and builds the ranker around it. The
scoring keystone comes from the bridging/collective-response literature (§4); the machinery
that makes it identifiable and stable comes from counterfactual learning-to-rank,
performative prediction, and two-timescale stochastic approximation (§6, §9); the
attention-economy mechanisms draw on provider-side fairness, bandit exploration, and the
discoverer literature (§8). Ethelo contributes the objective and the influence-recycling
idea of §8. Section 14 gives the full attribution.

This paper assembles the strength objective, the attention-economy mechanisms it implies,
and the estimator that makes it identifiable and stable, targeted at the fediverse — where
there is no incumbent algorithm, appetite for transparent and swappable ranking, and a
governance structure (per-instance, per-user) that makes the necessary value choices
legitimate rather than imposed.

## 2. Design principles

1. **Bridging, not consensus.** Reward support that holds *across* a population's divides.
   The goal is constructive conflict (conflict *transformation*), not homogeneity or the
   elimination of disagreement.
2. **Price signal quality, not activity.** A discriminating, cross-cutting reaction is
   worth more than a reflexive one; an indiscriminate scroller's reaction is near-zero
   information.
3. **Earn visibility.** The right to be seen at volume is earned through realized broad
   support, not bought with raw posting.
4. **Circulate influence.** Prevent influence — over rankings or over the model itself —
   from calcifying into a permanent taste aristocracy.
5. **Fair audition for the unproven.** A newcomer's first exposure is a commons cost, not
   an author cost, and is decision-theoretically correct, not charity.
6. **Consumption is free; authority is earned.** A user may tune what *they* see without
   limit; no user may tune how much the system trusts *them*.
7. **Portable and federated.** Users own their preference profile; ranking is transparent,
   opt-in, and swappable; instances and users set the value parameters.

## 3. System overview

CHORD is a **valuation-and-allocation layer** sitting between candidate retrieval
(upstream, not owned) and presentation (downstream, not owned). It does three things:
values attention (which raters count, §5), values content (strength, §4), and allocates
scarce visibility (budget + constrained selection, §7–8). Everything else enters through
**ports** with crude built-in default adapters (so the core runs standalone) and optional
rich adapters (so a capable host upgrades it):

| Port | Supplies | Default adapter | Rich adapter |
|---|---|---|---|
| Identity | stable handle + forge-cost | account age/heuristics | verified-human / ZK-pseudonymous |
| Preference | influent function + history | local store | portable data pod (Solid/ATProto) |
| Partition | opinion clusters | built-in matrix factorization (§4) | Polis / external bridging service |
| Signal | attention-event stream | native reactions | richer telemetry |
| Candidate | posts to re-rank | home/instance timeline | dedicated retrieval |
| Config | M, ρ, θ, ε, constraints | instance defaults | governance UI |

The load-bearing observation for federation: identity, preference, signal, and candidate
all have *some* native fediverse answer; **opinion-clustering (Partition) does not**, which
is why the core must ship its own factorization as the default rather than depending on an
external service.

## 4. The relation model (keystone)

### 4.1 Factor reception; do not score it separately

The keystone comes from the bridging/collective-response literature: the opinion-embedding
factorization below is the Community Notes / Polis approach. Build a **weighted, biased
matrix factorization** of the signed reaction matrix
$r_{up}$ (e.g. boost $+1$, favorite $+0.5$, exposed-no-reaction $-c$, mute $-1$):

$$
\hat r_{up} \;=\; \mu \;+\; b_u \;+\; b_{a(p)} \;+\; b_p \;+\; \langle x_u,\, y_p\rangle
$$

- $x_u\in\mathbb R^d$ — user **opinion embedding** (position in divide-space)
- $y_p\in\mathbb R^d$ — post **opinion loading** (which pole it appeals to)
- $b_u$ — rater leniency; $b_{a(p)}$ — **author** baseline; $b_p$ — post intercept

The author term $b_{a(p)}$ is essential and was easy to omit: without it, a famous
account's blanket elevation from reach leaks into $b_p$. With partial pooling
($b_p\sim\mathcal N(0,\tau_p^2)$, $b_a\sim\mathcal N(0,\tau_a^2)$), $b_p$ measures a post's
*marginal* breadth above its author's baseline.

**Bridged support and divisiveness fall out as orthogonal quantities.** The partisan appeal
lives entirely in $\langle x_u,y_p\rangle$; the reception *not* explained by opinion
alignment is the intercept $b_p$. So $b_p$ is a first proxy for bridged support (this is
what Community Notes ranks on), and divisiveness is the population spread of the alignment
term:

$$
D(p) \;=\; y_p^\top A\, y_p
$$

With whitened embeddings and $A=I$ this is just $\lVert y_p\rVert^2$ — universal posts sit
near the origin of opinion space; partisan posts sit far out. But $A=I$ is too glib: it
penalizes a "fishing vs. libraries" split as much as a political fault line. We take $A
\succeq 0$ **weighted toward the axes that correlate with an affective-polarization signal**
(regress the signal on embedding dimensions). This is where the instance's "which divides
do we care to bridge" decision physically lives (the ρ knob, §7).

### 4.2 Do not trust the scalar intercept — reconstruct per cluster

The scalar $b_p$ is a linear proxy for diverse approval and diverges from the real target
when clusters sit asymmetrically about the origin (it happily rewards a post one cluster
loves and another merely tolerates). Using the Partition port's clusters $c$, reconstruct
each cluster's predicted reception

$$
\hat r_{cp} \;=\; \mu + \bar b_c + b_{a(p)} + b_p + \langle \bar x_c,\, y_p\rangle
$$

and define bridged support as a **tested-breadth lower confidence bound**:

$$
\boxed{\;B_{\mathrm{LCB}}(p) \;=\; \min_{c}\Big[\, \hat r_{cp} \;-\; \beta\,\frac{\sigma}{\sqrt{n_{cp}+1}} \,\Big]\;}
$$

where $n_{cp}$ is the (propensity-corrected) number of cluster-$c$ users who were actually
exposed to $p$. This resolves the central failure mode in one stroke: if a cluster that
would disagree has not yet been exposed ($n_{cp}\approx 0$), its penalty term is large and
$B_{\mathrm{LCB}}$ stays low — **a post is not credited as bridging until it has survived
contact with the people who would dislike it.** The $\min_c$ form is Ethelo's Rawlsian
strength and Polis's group-aware consensus. Keep the scalar $b_p$ as a cheap pre-filter;
rank on $B_{\mathrm{LCB}}$.

Note the deliberate asymmetry in how uncertainty is used: the exploration pool (§8) samples
*high*-uncertainty posts optimistically to decide what to **audition**; $B_{\mathrm{LCB}}$
uses uncertainty pessimistically to decide what to **crown**. Optimism explores; pessimism
rewards. This is what prevents imperfect estimates from manufacturing false bridging.

## 5. Rater weighting: quality-tracking, not variance

Each observation in the fit is weighted by a per-rater influence $\lambda_u$, so that
discriminating cross-cutting raters dominate the estimate and indiscriminate scrollers
barely move it ("price selectivity, not activity"). **The naive choice is actively
harmful.** Inverse-variance weighting rewards *predictability*: extremists are predictable,
so they get low residual variance and high weight, while thoughtful evaluators get inflated
residuals and are downweighted — empirically, this amplifies ideologically extreme raters
and makes the system *more* vulnerable to partisan attack [Quality-Sensitive MF for
Community Notes, 2026].

The correction is a **quality-tracking / peer-prediction weight**, estimated jointly inside
the factorization: give more influence to raters whose *ideology-adjusted* ratings are
consistent with the note-quality estimate learned from all ratings. This is also the modern
form of the recursive cross-divide credibility used by YourView ("you are credible if
trusted by credible people who disagree with you"), computable via a damped, teleporting
eigentrust iteration on the learned geometry:

$$
\lambda \;\leftarrow\; \tfrac{1-\delta}{n}\mathbf 1 \;+\; \delta\, T^\top \lambda,
\qquad
T_{vu} \;=\; \frac{\sum_{p:\,a(p)=u}\, [r_{vp}]_+\,\cdot\,\mathrm{dist}(x_v, x_u)}
                  {\sum_{u'}\sum_{p:\,a(p)=u'}\, [r_{vp}]_+\,\cdot\,\mathrm{dist}(x_v, x_{u'})}
$$

$T$ must be **row-stochastic** — normalized over each *rater's outgoing* trust
($\sum_u T_{vu}=1$), not over each author's incoming — so a rater distributes one fixed unit
of trust among the authors it approves (classic EigenTrust). The choice is load-bearing for
Sybil starvation: under *column* normalization a Sybil author boosted by a single colluding
puppet would inherit that puppet's entire weight (its lone incoming edge normalizes to $1$),
whereas under row normalization an honestly-approved author accrues from *many* independent
cross-divide raters while a one-puppet Sybil receives only that puppet's fraction. The teleport
floor ($\delta<1$) makes this a contraction with a unique fixed point, floors
every rater's weight (no one is zeroed), and starves Sybils (fresh accounts have no
incoming cross-divide trust). **Whichever estimator is used, weight by agreement with the
bridged-quality signal after ideology is projected out — never by residual variance.**

Two earned quantities feed off the same geometry:

- **Scout precision** $q_{\text{scout}}(u)$ — reward being *early* on posts that
  *eventually* score high strength. Self-correcting, because it is graded against future
  outcomes, not current consensus:
  $$
  q_{\text{scout}}(u) \;=\; \frac{\sum_{p\in P_u^{+}} e^{-\alpha\,\mathrm{rank}_t(u,p)}\,\Phi_\infty(p)}{\sum_{p\in P_u^{+}} e^{-\alpha\,\mathrm{rank}_t(u,p)}}
  $$
- **Consumption vs. authority.** A user's *consumption* appetite along any factor is a free
  knob; their *authority* (λ, scout precision) is earned. Declaring scout-inclination can
  route more auditions to a user, but their verdict carries scout-weight only if their
  record earned it.

## 6. Identifiability and the propensity model

### 6.1 The problem

Reactions are observed only where a user was *shown* a post, and every historical policy
shows people mostly in-group content. So a divisive post can look bridging simply because
the outgroup that would hate it never saw it — the negatives are **missing, not negative**,
and the missingness depends on the very alignment we are estimating (MNAR). Uncorrected,
this inflates $b_p$ and shrinks $\lVert y_p\rVert$: in-group popularity masquerades as
bridging.

### 6.2 The correction

This is the counterfactual learning-to-rank problem. Weight each observation by inverse
propensity $1/\pi_{up}$ (the probability the pair was exposed), which yields an unbiased
objective **provided propensities are accurate and non-zero for every relevant pair**
[Joachims, Swaminathan & Schnabel 2017; Schnabel et al. 2016]. The non-zero-everywhere
(positivity) requirement is exactly why the exploration pool is not optional — it
guarantees $\pi\ge\epsilon>0$ for all items, and its known, alignment-*independent*
exposure is the **unconfounded anchor** that makes organic propensities estimable and
checkable. Use the self-normalized estimator to control IPW variance, and treat silent
disagreement (exposed-but-no-reaction) as a weak negative so haters scrolling past stop
reading as "missing."

$$
\mathcal L(\Theta) \;=\; \frac{\sum_{(u,p)\in E} \omega_{up}\,(r_{up}-\hat r_{up})^2}{\sum_{(u,p)\in E}\omega_{up}} \;+\; \Omega(\Theta),
\qquad
\omega_{up} \;=\; \lambda_u \cdot \min\!\Big(\tfrac{1}{\hat\pi_{up}},\,W_{\max}\Big) \cdot s_{up}
$$

Tie the clip to the exploration floor: with $\pi\ge\epsilon$ guaranteed for audited items,
setting $W_{\max}=1/\epsilon$ gives a natural ceiling on inverse weights and hence on
gradient variance. This is a variance-for-bias trade, not a free structural guarantee — a
genuinely in-group-misaligned pair (an outgroup member who would dislike a post and was
never shown it) can have true per-pair propensity *below* $\epsilon$, so clipping there
biases exactly the deep-MNAR pairs the correction exists to recover. Adopt the cap, but
size it knowing it slightly under-corrects the hardest cases.

**Weight scale vs. regularization.** The data term is self-normalized (divided by
$\sum\omega$) and hence invariant to a global rescaling of $\omega$, but $\Omega(\Theta)$ is
*not*. Because $\lambda_u$ arrives as a normalized distribution ($\sum_u\lambda_u=1$ from the
§5 eigentrust fixed point), the raw $\omega_{up}$ are $O(1/|E|)$, so a fixed $\Omega$ silently
dominates the fit and collapses the embeddings toward the origin. Rescale $\omega$ to unit
mean over $E$ before each solve (equivalently, read $\Omega$'s strength relative to
mean-one weights); by the self-normalization above this leaves every estimate unchanged
while restoring a well-conditioned regularized problem.

### 6.3 The propensity model — an open menu, not a commitment

Because misspecified propensities silently reintroduce the MNAR bias the whole edifice
corrects, we deliberately keep this pluggable and prefer estimators that fail gracefully:

- **(a) Doubly-robust (default posture).** Pair *any* propensity estimate with a reception
  imputation model; the estimator is consistent if *either* the propensity *or* the
  imputation is right [Dudík et al. 2014; Saito 2020]. This is the safest default and
  should wrap whatever else is chosen.
- **(b) Exposure/position click models** (cascade, examination) — classic CLTR propensity
  from the logging policy's presentation.
- **(c) Policy-derived propensity** — if the logging ranker's scores are known, derive
  $\hat\pi$ from its (softmax) selection probabilities directly.
- **(d) Intervention harvesting / randomization** — use the exploration pool's known
  $\pi\approx\epsilon$ as ground-truth anchor to calibrate or validate (b)/(c).
- **(e) Propensity-light fallbacks** — affine corrections or propensity-independent bias
  recovery for regimes where propensities are unreliable [Vardasbi et al. 2020].

The throughline across all options: **the exploration pool is the randomized, unconfounded
mass that anchors identifiability**, so its rate $\epsilon$ can never be driven to zero.

## 7. Value model and feed assembly

### 7.1 Personalized value

$$
V(u,p) \;=\; \underbrace{B_{\mathrm{LCB}}(p)}_{\text{tested bridged support}} \;+\; \underbrace{(1-M)\,\langle x_u, y_p\rangle}_{\text{personalization}} \;-\; \underbrace{M\,\rho\, y_p^\top A\, y_p}_{\text{divisiveness penalty}}
$$

$M\in[0,1]$ is the master dial. $M=0$: give me what my side likes, divisiveness included
(engagement-like). $M=1$: pure bridging — broad tested support only, partisan lean ignored,
divisiveness penalized. This is a **consumption** choice and therefore ungameable — it only
changes the chooser's own feed.

Here $A$ is the *fixed* ($\rho=1$) divide-weighting of §4.1 and $D(p)=y_p^\top A\,y_p$ is
likewise defined at $\rho=1$ (Appendix A); the $\rho$ knob enters the value **exactly once**,
as the coefficient on the penalty term. The §12 shorthand "$\rho$ scales $A$" is realized
*here* — do not additionally fold $\rho$ into $A$ inside $D$, or the knob is applied twice.

### 7.2 Feed as constrained selection

Per viewer, choose $N$ slots to maximize value under constraints. This follows Ethelo's
framing of a decision as picking the combination that maximizes strength subject to
constraints; the feed realizes it as submodular top-$k$ re-ranking with a greedy $1-1/e$
guarantee:

$$
\max_{S:\,|S|=N}\; \sum_{p\in S}\Big[(1-\epsilon)\,\textstyle\sum_f \theta_f V_f(u,p) \;+\; \epsilon\,\tilde\Phi(p)\Big]\cdot \mathrm{posdisc}(p)
$$

subject to: per-author cap and budget ($\sum_{p\in S,\,a(p)=a}E(p)\le B(a)$, §8);
**diverse-*approval* coverage** (submodular — diminishing returns for re-covering an
already-covered region of opinion space); exploration floor $\ge\epsilon N$. Greedy gives
the $1-1/e$ guarantee; run it over the top few hundred candidates, not the corpus. Meet the
hard floor by rounding **up**, $\lceil\epsilon N\rceil$: on the small feeds typical of a
fediverse instance a floor of $\lfloor\epsilon N\rfloor$ rounds a sub-unit reservation to
zero and silently forfeits the positivity guarantee $\pi\ge\epsilon>0$ that §6.2
identifiability rests on. (The *randomized* identifiability anchor of §6.2 — the slice whose
unbiasedness needs the exposure rate to equal $\epsilon$ **in expectation**, not merely to be
nonzero — is instead realized by *stochastic* rounding of $\epsilon N$: reserve
$\lfloor\epsilon N\rfloor$ and one more with probability $\epsilon N-\lfloor\epsilon N\rfloor$.)

> **Diverse approval, not diverse exposure.** The intuitive "show people the outgroup" move
> has poor validity as a bridging measure — evidence suggests indiscriminate outgroup
> exposure can *increase* affective polarization [Ovadya & Thorburn 2023, §Evaluation]. The
> constraint rewards content that *earns* cross-cluster support, never forced exposure.

### 7.3 The factor vector

$V_f$ are per-factor value functions (trend, scout, depth, locality, recency…), each with
its own per-rater precision, its own consumption weight $\theta_f$ (a free user knob on the
simplex), and its own slider. Trend and scout are latency-coupled: scout-strength is a
leading indicator of trend-strength, so users cannot fully opt out of the scout factor
without starving future discovery (a commons argument for the exploration pool).

## 8. Attention-economy mechanisms

**Author visibility budget (conserved + earned).** Per author, per window, exposure is
capped and replenished by realized strength:

$$
\sum_{p:\,a(p)=a,\,p\in W} E(p) \le B(a),
\qquad
B_{t+1}(a) = \mathrm{clip}\!\Big(B_0 + \eta\!\!\sum_{p:\,a(p)=a,\,p\in W_t}\!\! [\Phi(p)]_+\,E(p),\ \ 0,\ B_{\max}\Big)
$$

Two guards keep the replenishment well-behaved: the rectifier $[\Phi(p)]_+$ makes a
net-divisive post ($\Phi<0$) simply *fail to replenish* rather than draining the author's
floor $B_0$ (which would double-punish and could drive $B(a)$ negative), and the clip to
$[0,B_{\max}]$ keeps every budget bounded — a precondition the §9 bounded-regime argument
relies on. Firehose posting spreads a fixed budget thin; quality regenerates it. This inverts the
engagement logic (where each post is an independent virality lottery ticket, so volume is
rational) into one where posting more *dilutes you* unless it earns. The budget binds to
the **identity** port, not the raw account, so it cannot be sharded across sockpuppets.

**Exploration pool (cold-start, base-rate-calibrated).** New posts have no $b_p$. Audition
them via Thompson sampling — but **not** with a flat optimistic prior, which
systematically over-explores weak items because the real base rate of "winners" is far
below 50% [Dynamic Prior Thompson Sampling 2025]. Initialize the prior to the empirical
newcomer strength rate. A commons-funded fraction $\epsilon$ of every feed's slots is
reserved for high-uncertainty posts, routed preferentially to **high-$q_{\text{scout}}$
raters** (they resolve uncertainty in the fewest impressions). The audition closes on
**evaluation saturation** ($\hat\sigma_p^2$ below threshold), not wall-clock, so slow-burn
long-form is not buried. Grants are bounded **per verified identity** (a periodic audition
for humans; nothing for Sybil farms).

**Influence recycling (anti-ossification).** Damp the consistently-satisfied, boost the
under-served, so the system listens hardest to whoever it serves worst:

$$
\lambda^{\mathrm{eff}}_u \;=\; \lambda_u\,\big(1 + \zeta\,(\overline S - \bar S(u))\big)
$$

with $\bar S(u)$ the user's model-estimated realized value over what they were shown (hard
to fake by "acting dissatisfied"). This is the governor that keeps §5's rater-weighting
from calcifying into a taste aristocracy — the one mechanism most ranking systems lack.

## 9. Estimation and dynamics

### 9.1 The loop

Per window: **(1)** fit the doubly-robust, IPW-corrected, λ-weighted MF (block-convex ALS,
closed-form per block) → embeddings, biases; **(2)** whiten, recompute $A$ and $D$; **(3)**
update quality-tracking λ on the new geometry; **iterate 1–3**; **(4)** update
$q_{\text{scout}}$; **(5)** update author budgets; **(6)** update Thompson posteriors;
**(7)** apply recycling → $\lambda^{\mathrm{eff}}$. Per request: retrieve candidates → score
$V(u,p)$ with the user's knobs → greedy constrained select → serve → log.

### 9.2 Why it is stable — and the exact conditions

Two coupled instabilities, each with a governing theory:

- **The λ↔x coupling** (credibility depends on geometry; geometry is fit weighted by
  credibility) is structurally an **actor-critic**, hence **two-timescale stochastic
  approximation** [Borkar 1997]: the fast "critic" is the embedding fit tracking a
  quasi-static λ; the slow "actor" is the λ update. Almost-sure convergence holds under
  timescale separation (fast step / slow step $\to 0$) plus a stability condition
  [Lakshminarayanan & Bhatnagar 2017], with concentration bounds [Borkar & Pattathil 2018]
  and finite-time rates [Doan 2023]. The theorem needs each timescale's limiting ODE to
  have a unique stable equilibrium; the fast one (embedding given λ) has this **only if
  strongly convex**, which the exploration anchor provides. So the anchor is the
  *precondition* that makes the convergence result apply — not a separate trick.
- **The outer allocation→data loop** (rankings change the reactions we then train on) is
  **performative prediction** [Perdomo et al. 2020]. Repeated retraining converges linearly
  to a **performatively stable** point iff performative effects are weak and
  well-conditioned (roughly, distributional sensitivity below the strong-convexity /
  smoothness ratio). Above that threshold it can oscillate or diverge. This gives a concrete
  safety target: keep the sensitivity of the reaction distribution to a re-rank small
  relative to loss curvature — mechanically bought by the exploration anchor and by slow
  knob changes. The λ-memory case is the *stateful* variant [Brown, Hod & Kalemaj].

### 9.3 Stability as a monitored runtime property

Because global convergence is **not** guaranteed in the nonconvex regime, the right target
is a **bounded stationary regime**, held by: persistent excitation ($\epsilon\ge
\epsilon_{\min}$, so the system never stops sampling regions it stopped showing); slow knob
changes; SNIPW + clipping to bound gradient variance; under-relaxation and two-timescale
separation. Run a **controller on the estimator's own concentration**: track effective
rater count $(\sum\lambda)^2/\sum\lambda^2$ (or $\mathrm{Gini}(\lambda)$); if it collapses,
automatically raise the teleport floor $\delta$ and the damping. The exploration pool is
therefore load-bearing four times over — provider fairness, cold-start, causal
identification, and estimator stability — and its rate is a floored system invariant, not a
user preference.

**Separating endogenous from exogenous shift.** The reaction distribution moves for two
reasons: the ranker's own allocations (endogenous, a feedback loop to *damp*) and real-world
drift such as breaking news (exogenous, a signal to *track*). Damping both chases the loop
*and* lags real events; tracking both amplifies the loop. The exploration floor already
supplies the instrument to tell them apart: the non-personalized slice (chronological or
randomly-ranked exposures at rate $\ge\epsilon_{\min}$) is not driven by the personalized
ranker, so distributional drift measured *there* is an estimate of exogenous background
shift, while drift in the personalized stream *beyond* that baseline is attributable to the
loop. Feed the exogenous estimate to the controller as an offset — raise damping and knob
inertia only in response to the residual endogenous component — so the system holds steady
against its own feedback without treating a genuine news event as instability to suppress.
This reuses machinery already present (the exploration cohort) rather than adding a
subsystem.

## 10. Adversarial robustness

- **Sybil / sockpuppets** — the visibility budget binds to identity, not accounts;
  trust propagation gives fresh accounts ≈ zero weight. Sharding across accounts gains
  nothing.
- **Brigading** — two independent defenses: a brigade of fresh accounts has no cross-divide
  trust path, and a brigade *creates* a split distribution that the divisiveness term and
  the $B_{\mathrm{LCB}}$ min-over-clusters penalize. Gaming lowers the score.
- **Bridging-bait (Goodhart)** — shallow universal content (cat memes) can score high
  bridged support; the depth factor $\theta_{\text{depth}}$ resists this but does not
  eliminate it.
- **High-precision collusive clique** — genuinely the hardest residual: colluding experts
  read like independently-delighted experts, and peer-prediction weighting has equilibria
  where raters agree with *each other* rather than with truth. Only timing/provenance
  separates collusion from consensus. Named, not solved.

## 11. Federated deployment

Every port ships a crude default so the core runs on a single instance, and a rich-adapter
slot so a capable host upgrades it. On Bluesky's AT Protocol, pluggable "feed generators"
make the whole layer first-class; on Mastodon/ActivityPub there is no equivalent standard,
so deployment is per-instance or client-side. A maximal host (e.g. a Polity-class stack)
fills every port — verified/ZK identity, portable data pods, Polis clustering, a feed
substrate, governance UI — and thereby serves as an existence proof that all six ports can
be filled; but the design depends on *none* of it, and treats such a host as one adapter
set among many. Two natural levels: an instance runs a strength-ranked community timeline
(the collective-surface use the bridging/collective-response lineage fits most naturally);
individuals get personalized ranking on top, with
$M$, $\rho$, $\theta$, $\epsilon$ as their own dials.

## 12. The knob panel

| Knob | Meaning | Range | Who sets it |
|---|---|---|---|
| $M$ | bridging vs. personalization | $[0,1]$ | user (free) |
| $\rho$ | collective-identity / which divides to bridge | $[0,1]$, scales $A$ | instance default + user override |
| $\theta_f$ | factor mix (trend/scout/depth/locality…) | simplex | user (free) |
| $\epsilon$ | exploration appetite (floored) | $[\epsilon_{\min},\epsilon_{\max}]$ | user (free), floor system-set |
| $\lambda_u,\ q_{\text{scout}},\ B(a)$ | rater/author authority | — | **earned, never user-set** |

The last row is the consumption-vs-authority wall, enforced structurally.

## 13. Limitations and open problems

Honest residuals, several of which no fix fully removes:

1. **Conditional convergence.** All guarantees assume *weak* performativity and, for the
   inner loop, a locally unique equilibrium the anchor only provides *locally*. Under strong
   performative effects the system can oscillate; we monitor rather than prove.
2. **Unobserved confounding in the propensity model** (§6) is the softest load-bearing
   wall — and the residual is *bias*, not variance. The variance failure (weights exploding
   as $\pi\to0$) is handled by SNIPW, doubly-robust estimation, the $1/\epsilon$ clip, and
   the $B_{\mathrm{LCB}}$ pessimism. What none of these touch is a hidden variable that drives
   *both* exposure and reaction but is absent from the model: no calibration, clipping, or
   doubly-robust wrapping removes confounding bias, because the estimand itself is
   misidentified. The honest instrument here is not another variance damper but **sensitivity
   analysis** — quantifying how much unobserved confounding would be required to overturn a
   bridging verdict (e.g. Rosenbaum-style bounds or a confounding-strength parameter on
   $\hat\pi$) — reported as a robustness interval on $B_{\mathrm{LCB}}$ rather than a point
   claim. The exploration pool's randomized slice bounds this in the limit (randomized
   exposure is unconfounded by construction), but only for the fraction of traffic it covers.
   Named, bounded, not closed.
3. **Bridging is non-monotone in audience.** $B_{\mathrm{LCB}}$ certifies bridging over the
   *exposed* set; a post bridging at 10K may divide at 10M. The confidence must widen as the
   target population outruns the tested set — this ties back to saturation windowing and
   never fully closes.
4. **Low-dimensional opinion space** may under-represent true plurality; divisiveness along
   an unmodeled axis leaks into $b_p$. $d$ is a real bias-variance knob (too small leaks
   divisiveness into support; too large smears the divide you care about).
5. **Peer-prediction collusion** and the high-precision clique (§10) — a structural
   equilibrium problem, not a bug.
6. **Recycling is mildly farmable** in principle (acting under-served), mitigated by
   model-estimated satisfaction but not eliminated.
7. **Distinguishing harmful from benign divides** ($A$'s weighting) is a normative,
   instance-level choice with no purely technical answer.

*Directions considered and declined.* Several defensive add-ons were evaluated and left out
deliberately, on the principle that each addition should strengthen an existing mechanism or
sharpen a limitation rather than open a new subsystem: safe-policy (KL-to-chronological)
regularization (redundant with the three variance controls above, and pulls toward a
recency baseline the design rejects); stress-triggered $L_2$ boosting (raises convexity but
flattens the very opinion geometry $D(p)$ depends on); and timing-entropy or
clique-percolation collusion *detectors* (evadable by jitter, computationally heavy, and —
critically on the fediverse — false-positive on legitimate tight-knit niche communities,
which are the platform's atomic unit). The structural defenses of §10 (attacks made
unprofitable) are preferred over evadable detectors. Adversarial-challenge auditing of
suspicious bridged support is retained only as an anti-amateur measure; it does not defeat
the §10 high-precision clique, whose sock-puppets already occupy the extreme coordinates a
challenge would route to.

## 14. Relationship to prior work

Ordered roughly by how much of the working machinery each source contributes.

- **Bridging systems** [Ovadya & Thorburn 2023] — the conceptual foundation: the
  attention-allocation framing, the diverse-approval motif, and the validity/reliability
  cautions (notably that diverse *exposure* can backfire). Also the federated-deployment
  open problem this work takes up.
- **Collective response — Community Notes / Polis / YourView** — the **keystone**. The
  opinion-embedding matrix factorization that separates ideology from cross-cutting support
  (Community Notes / Polis), group-aware consensus and the clustering that $B_{\mathrm{LCB}}$
  reconstructs over (Polis), and recursive cross-divide credibility (YourView). The
  single most important mechanism in the paper — the scoring keystone of §4 — originates
  here.
- **Community Notes / QSMF** — the deployed bridging MF at scale, its documented failure
  list (latency, surge instability, hyperactive-minority influence, manipulation) that
  validates our problem set, and the inverse-variance pathology plus the quality-tracking
  correction adopted in §5.
- **Counterfactual LTR** [Joachims–Swaminathan–Schnabel 2017; Schnabel et al. 2016;
  Steck 2010; Swaminathan–Joachims 2015; Dudík et al. 2014] — the MNAR/propensity core;
  everything in §6 that makes the keystone identifiable.
- **Performative prediction** [Perdomo et al. 2020; Brown–Hod–Kalemaj] — the outer-loop
  stability theory (§9).
- **Two-timescale stochastic approximation** [Borkar 1997; Lakshminarayanan–Bhatnagar 2017;
  Borkar–Pattathil 2018; Doan 2023] — the λ↔x convergence theory (§9).
- **Provider-side fairness / popularity bias** [Abdollahpouri et al.; Patro et al.] — the
  Matthew-effect / long-tail framing behind the budget and exploration mechanisms (§8), and
  the "re-ranker, not retrieval" architecture.
- **Multi-armed bandits** [Chapelle–Li; Li et al.; Dynamic Prior TS 2025] — the exploration
  pool and its base-rate-calibrated prior (§8).
- **Discoverers / "Hit-Savvy"** — empirical basis for scout precision (§5).
- **Ethelo** — the **objective** (strength = support net of dissonance), the
  **influence-recycling** mechanism (§8), and the framings of Rawlsian aggregation and
  constrained-outcome selection.

## Appendix A. Notation

$u$ user, $p$ post, $a(p)$ author, $c$ opinion cluster · $r_{up}$ signed reaction ·
$x_u,y_p\in\mathbb R^d$ opinion embedding / loading · $\mu,b_u,b_{a(p)},b_p$ biases ·
$\lambda_u$ rater influence (quality-tracking) · $q_{\text{scout}}(u)$ scout precision ·
$D(p)=y_p^\top A y_p$ divisiveness · $A\succeq0$ divide-weighting · $B_{\mathrm{LCB}}(p)$
tested bridged support · $\pi_{up}$ exposure propensity · $\epsilon$ exploration rate ·
$V(u,p)$ value · $\Phi(p)$ realized strength · $B(a)$ author budget · $\eta$ replenishment ·
$\zeta$ recycling · knobs $M,\rho,\theta_f,\epsilon$.

## Appendix B. One-line summary of the architecture

*Rank by tested cross-cluster support net of weighted divisiveness; weight raters by
quality-tracking not variance; price authors by a strength-replenished conserved budget;
audition the unproven from a floored commons pool that doubles as the identifiability
anchor; correct exposure MNAR with doubly-robust propensity weighting; stabilize the coupled
estimator as two-timescale stochastic approximation held in a monitored bounded regime; and
expose M/ρ/θ/ε as knobs while keeping all authority earned.*

## Appendix C. Retroactive evaluation

No single public dataset exercises the whole system, and the feedback loop (§9) cannot be
tested on any fixed dataset at all — a static archive cannot respond to the ranker's own
allocations. Retroactive evaluation therefore validates the *static* components
(scoring, rater weighting, ranking quality, debiasing) against public data, and defers the
*dynamic* components (performative stability, exploration) to a simulator. The critical
constraint is exposure: almost no organic social dataset logs what users were *shown*, so
the propensity layer (§6) can only be validated on the rare datasets that carry a
randomly-exposed holdout, or via a semi-synthetic harness where exposure is simulated so
ground truth is known.

### C.1 Layer-to-dataset mapping

| Layer | What it tests | Best datasets | Notes |
|---|---|---|---|
| Keystone (§4): $B_{\mathrm{LCB}}$, per-cluster reception, divisiveness | bridged support, opinion embeddings | **Community Notes** (rater×note ratings, public daily TSVs) | Same shape as the model; open-source CN baseline **and** QSMF baseline to benchmark against. Start here. |
| Rater weighting (§5): quality-tracking $\lambda$ | manipulation resistance, extremist-amplification check | **Community Notes** | Reproduce the QSMF inverse-variance pathology and the correction directly. |
| Real deliberation / clusters | per-cluster reconstruction on genuine divides | **Polis** open conversation data | Small; participant×comment agree/disagree/pass with validated clusters. |
| Scale + real divides + authors + time | author budget, scout precision, ranking realism | **Reddit** (Academic Torrents dumps) | Votes not per-user; build $x_u$ from community co-participation (Waller–Anderson). |
| Propensity / MNAR (§6) | IPS / doubly-robust recovery | **Yahoo!R3, Coat, KuaiRec/KuaiRand** | The only datasets with a randomly-exposed (MAR) holdout = the unconfounded anchor. KuaiRand adds timestamps/features. |
| Exploration / bandit (§8) | Thompson audition, base-rate prior | **Open Bandit Dataset**, KuaiRand | Logged uniform-random **and** Thompson-sampling policies. |
| Trust / credibility propagation (§5) | eigentrust $\lambda$ on signed votes | **Wikipedia RfA**, Epinions/Slashdot signed nets (SNAP) | Rare per-user *signed* votes. |
| Temporal / author / scout only | budgets, early-signal backtest, ranking sandbox | **Hacker News**, StackExchange | No per-user votes → cannot do the keystone; use aggregate score as $\Phi_\infty$ and early *commenters* as a scout proxy. |

### C.2 Suggested sequence

1. **Community Notes** — implement $B_{\mathrm{LCB}}$, per-cluster reconstruction, and the
   quality-tracking $\lambda$; benchmark against the open-source CN algorithm and QSMF.
   This proves the keystone and rater weighting against real baselines on real divides.
2. **Yahoo!R3 / Coat / KuaiRand + semi-synthetic (C.3)** — validate the propensity menu and
   the doubly-robust estimator against a known unbiased holdout.
3. **Reddit** — test author budget, scout precision, and end-to-end ranking on polarized
   topics at scale with real timestamps and authors.
4. **Simulator (C.4)** — the only way to exercise the closed loop.

### C.3 Semi-synthetic propensity harness

Because the propensity model is deliberately an open menu (§6.3), the cleanest test gives
you control of ground truth:

1. Take a near-complete or MAR ratings matrix $R$ (e.g. the Yahoo!R3 / Coat / KuaiRand
   random-exposure block, or a dense MovieLens slice).
2. Define a synthetic logging policy $\pi^{\text{log}}_{up}$ that exposes items in
   proportion to predicted in-group alignment — i.e. deliberately induce MNAR of the exact
   kind §6.1 warns about (in-group over-exposure).
3. Sample an observed set $E\sim\pi^{\text{log}}$; hide the rest.
4. Fit the model on $E$ under each propensity option (position/cascade, policy-derived,
   intervention-harvested from a held-in random $\epsilon$-slice, doubly-robust wrapping).
5. Score against the *hidden* full matrix as ground truth. Report: (a) does uncorrected
   fitting inflate $b_p$ / shrink $\lVert y_p\rVert$ (the predicted pathology); (b) does each
   propensity option recover the unbiased ranking; (c) how gracefully each degrades as
   $\pi^{\text{log}}$ is misspecified relative to the fitted $\hat\pi$; (d) the value of the
   random $\epsilon$-anchor — sweep its size toward zero and watch identifiability fail.

Metrics: unbiased NDCG@k / AUC against the MAR holdout (the standard for Yahoo!R3/Coat), plus
the pathology diagnostics on $b_p$ and $\lVert y_p\rVert$, plus an extremist-amplification
curve for $\lambda$ (reputation vs. ideological extremity, per QSMF Figure 1).

### C.4 What requires a simulator (not retroactive)

The feedback loop is untestable on fixed data. To exercise §9 you need an agent-based
simulator with: a synthetic population placed in opinion space; a response model
$P(\text{react}\mid x_u, y_p)$; content generation by author-agents that adapt to the
incentive (to test whether the conserved budget actually suppresses firehosing and whether
"bridging-bait" emerges); and the full closed loop (rank → simulated reactions → retrain).
Targets to measure: convergence to a performatively stable point vs. oscillation as you vary
the sensitivity knobs; the effective-rater-count / Gini controller (§9.3) holding
concentration bounded; and whether exploration at rate $\epsilon$ sustains the identifiability
anchor over time. This is where the performative-prediction and two-timescale results become
empirical rather than assumed.

## Appendix D. Reference architecture for a fediverse deployment

A concrete sketch. The design is two planes — an offline **learning plane** (batch,
per-window) and an online **serving plane** (per-request) — connected by an append-only
event log and a set of model/ledger stores. Everything maps to the ports of §3; example
technologies are illustrative, not prescriptive.

### D.1 Deployment topology

ActivityPub/Mastodon has no native pluggable-feed-generator hook, so the ranker runs as an
**instance-side re-ranking service** (a sidecar the instance's timeline endpoint calls) or
as a **client-side ranker** consuming the user's home timeline via the Mastodon API. On
Bluesky's AT Protocol the same service registers as a first-class **feed generator**, which
is the cleaner integration and requires no instance cooperation. A crucial scale point: most
instances are small (10²–10⁴ users), the ranker is a re-ranker over a few hundred
candidates, and the MF trains on one instance's bounded reaction history — so this runs on
modest hardware, which is precisely why it fits the fediverse rather than requiring
hyperscale infrastructure.

```
                          ┌──────────────────────── LEARNING PLANE (batch, per window) ─────────────────────────┐
                          │                                                                                      │
  reactions / exposures   │   ┌───────────────┐   ┌──────────────────┐   ┌───────────────────┐   ┌────────────┐ │
  ───────────────────────────▶│  Event log    │──▶│ Relation-model   │──▶│ Rater/credibility │──▶│ Propensity │ │
        (Signal port)     │   │ (append-only) │   │ trainer (wMF+DR) │   │ (eigentrust + QT λ)│   │ estimator  │ │
                          │   └───────────────┘   └───────┬──────────┘   └─────────┬─────────┘   └─────┬──────┘ │
                          │           ▲                   │  embeddings,           │ λ_eff            │ π̂       │
                          │           │                   ▼  b_p, clusters, A      ▼                  ▼        │
                          │           │            ┌──────────────────────────────────────────────────────┐   │
                          │           │            │  Model & ledger store  (embeddings, budgets B(a),     │   │
                          │           │            │  Thompson posteriors, scout q, config M/ρ/θ/ε)        │   │
                          │           │            └───────────────┬──────────────────────────────────────┘   │
                          │           │                            │   ▲  concentration monitor (Gini, N_eff)  │
                          │           │                            │   └──── controller: adjust δ, η, ε_min ────┤
                          └───────────┼────────────────────────────┼──────────────────────────────────────────┘
                                      │                            │  reads
        ┌─────────────────────────────┼───────────────────────────┼───────────────── SERVING PLANE (per request) ─┐
        │                             │ logs events                ▼                                               │
  user  │   ┌────────────┐   ┌────────┴───────┐   ┌───────────────────────┐   ┌───────────────────────────────┐   │
  ──────────▶│ Candidate  │──▶│  Scorer        │──▶│ Feed assembler        │──▶│ Serve (Mastodon timeline API  │──────▶ feed
 request │   │ source     │   │ V(u,p): B_LCB, │   │ constrained submodular│   │ / ATProto feed generator)     │   │
        │   │ (Candidate │   │ D, factor blend│   │ select (budgets, ε,   │   └───────────────────────────────┘   │
        │   │  port)     │   │ + user knobs   │   │ diverse-approval)     │                                       │
        │   └────────────┘   └────────────────┘   └───────────────────────┘                                       │
        └──────────────────────────────────────────────────────────────────────────────────────────────────────┘
     Identity port (verified/ZK) and Preference port (portable pod: Solid / ATProto) sit alongside, read by both planes.
```

### D.2 Components

**Serving plane (low-latency, per request):**

- **Candidate source** (Candidate port) — instance/home timeline, follow graph, or
  federated firehose. Retrieval stays upstream; the ranker only re-ranks.
- **Scorer** — loads $x_u$, and per-candidate $b_p, y_p$, cluster receptions, and $A$ from
  the model store; computes $B_{\mathrm{LCB}}$, $D(p)$, the factor blend, and $V(u,p)$ with
  the requesting user's knobs $M,\rho,\theta,\epsilon$.
- **Feed assembler** — greedy submodular constrained selection (author caps + budget
  $B(a)$, diverse-*approval* coverage, exploration floor $\epsilon N$). Optionally the
  constrained step is delegated to the AGPL Ethelo engine (Dockerized, JSON in/out) as a
  MINLP solver; a greedy $1-1/e$ selector is the lightweight default.
- **Serve + log** — returns the ordered feed via the Mastodon timeline API or an ATProto
  feed generator, and emits the exposure event (which slots were shown) plus subsequent
  reactions back into the event log. Logging **exposure**, not just reactions, is what makes
  the propensity layer possible — most systems fail to log it.

**Learning plane (batch, per window):**

- **Event log** — append-only stream of exposures and reactions (e.g. a Postgres table or a
  Kafka/Redis stream). The single source of truth.
- **Relation-model trainer** — weighted, doubly-robust MF (ALS); emits $x_u, y_p, b_u,
  b_{a(p)}, b_p$, the whitened divisiveness metric $A$, and opinion clusters (the default
  Partition-port adapter; a rich adapter can call an external Polis/bridging service).
- **Rater/credibility service** — damped eigentrust on the learned geometry + quality-
  tracking $\lambda$; applies influence recycling to emit $\lambda^{\text{eff}}$.
- **Propensity estimator** — pluggable (§6.3 menu), wrapped doubly-robust; calibrated
  against the exploration pool's known $\epsilon$-exposures.
- **Ledgers** — per-author budgets $B(a)$ (strength-replenished), Thompson posteriors for
  auditions, scout precision $q_{\text{scout}}$.
- **Concentration monitor / controller** — tracks effective rater count
  $(\sum\lambda)^2/\sum\lambda^2$ and $\mathrm{Gini}(\lambda)$; if concentration climbs,
  raises the teleport floor $\delta$, damping $\eta$, and $\epsilon_{\min}$. This is the §9.3
  runtime guard that keeps the coupled estimator in its bounded regime.

The two planes run at different cadences — this **is** the two-timescale separation of §9:
the serving plane and the fast MF refit are the fast timescale; the $\lambda$/credibility
update is the slow timescale.

### D.3 Cross-cutting ports

- **Identity** — verified-human / ZK-pseudonymous handle; the author budget and Sybil
  resistance bind here, not to raw accounts. Default adapter: account-age/heuristic
  forge-cost.
- **Preference** — the user's opinion embedding, factor knobs, and scout/credibility
  history live in a **portable data pod** (Solid or ATProto record) the user controls, so
  moving instances carries the profile. Default adapter: local store.
- **Config** — instance defaults for $M,\rho,\theta,\epsilon$ and the constraint set (a
  governance surface), with per-user overrides for the consumption knobs; earned quantities
  ($\lambda, q_{\text{scout}}, B(a)$) are never user-writable.

### D.4 Federation notes

Clustering and the relation model are computed **per instance** on that instance's own
reaction history — no central authority, and the default Partition adapter needs no external
service. Cross-instance reputation flows through the trust graph (a graded alternative to
binary defederation). Because the influent function is portable, a user's profile is not
captured by any one instance. A maximal host that fills every port richly (verified/ZK
identity, portable pods, external Polis clustering, a feed substrate, a governance UI) is an
existence proof that all adapters can be provided, but the architecture runs on a single
vanilla Mastodon instance with only the default adapters.

## Appendix E. Implementation-derived corrections

Building a reference implementation surfaced five places where the math above was
underspecified or, taken literally, subtly wrong. Each correction is folded into the relevant
section; they are collected here for the record because each is a place a careful
re-implementer would otherwise rediscover the hard way.

1. **EigenTrust normalization (§5).** The trust matrix $T$ must be row-stochastic — normalized
   over each *rater's outgoing* trust ($\sum_u T_{vu}=1$) — not column-stochastic. Column
   normalization lets a Sybil author boosted by a single dedicated puppet inherit that puppet's
   entire weight (its lone incoming edge normalizes to $1$), defeating the very Sybil-starvation
   the mechanism exists for. The paper's "$\propto$" hid this; it is now written explicitly.
2. **Weight scale vs. regularization (§6.2).** Because $\lambda$ is a normalized distribution,
   the observation weights $\omega_{up}$ are $O(1/|E|)$. The self-normalized data term is
   invariant to that scale but the regularizer $\Omega(\Theta)$ is not, so $\Omega$ dominates
   and collapses the embeddings toward the origin. Rescale $\omega$ to unit mean before each
   solve; self-normalization leaves every estimate unchanged.
3. **Budget replenishment (§8).** Rectify the strength term, $[\Phi(p)]_+$, so a net-divisive
   post fails to replenish rather than draining the author's floor $B_0$, and clip
   $B_{t+1}\in[0,B_{\max}]$ so budgets stay bounded — a precondition the §9 bounded-regime
   argument relies on. The original unrectified, unclipped sum can drive $B(a)$ negative.
4. **Exploration-floor positivity (§7.2).** Meet the hard feed floor $\ge\epsilon N$ by rounding
   *up* ($\lceil\epsilon N\rceil$), and realize the *randomized* identifiability anchor of §6.2
   by *stochastic* rounding of $\epsilon N$ (correct in expectation). Either way, a plain
   $\lfloor\epsilon N\rfloor$ rounds a sub-unit reservation to zero on the small feeds typical
   of a fediverse instance, silently forfeiting the $\pi\ge\epsilon>0$ positivity that §6.2
   identifiability — and the §9.3 persistent-excitation invariant — rest on.
5. **Single application of $\rho$ (§7.1).** $D(p)$ and $A$ are defined at $\rho=1$; the $\rho$
   knob enters only as the §7.1 penalty coefficient. Reading §12's "$\rho$ scales $A$" as a
   rescaling of $A$ *inside* $D$ *and* keeping the §7.1 coefficient applies the knob twice.

None of these alters the paper's objective or its qualitative claims; each was validated by a
regression test that reproduces the corresponding claim on synthetic data (the keystone
ordering, avoidance of the inverse-variance pathology, IPW recovery under MNAR with the anchor
sweep, firehose dilution and Sybil binding, and the bounded-regime controller).
