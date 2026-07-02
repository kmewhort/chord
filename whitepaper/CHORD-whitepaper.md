---
title: "CHORD"
subtitle: "Cross-cluster Harmonized Optimization of Reception and Dissonance"
author: "A bridging, attention-economy feed-ranking algorithm for federated social networks"
date: "Working draft"
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
open — chiefly the propensity model on which identifiability rests. A reference
implementation is checked against real public datasets (Appendix C) and exercised in a
counterfactual agent-based simulator (Appendix C.4); this find-it-then-fix-it loop drove
several corrections that are folded into the sections below — an exposure-weighted
shrinkage keystone, an outgoing-diversity trust weight and a cluster-spread-gated
collusion defense, an anti-bait depth gate, and a bridging-vs-personalization operating
point that is interior rather than at the pure-bridging extreme.

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

Two roles for this factorization must be kept separate. It supplies the *personalized* value
$V(u,p)$ (§7) — where the embedding $x_u,y_p$ is genuinely needed and a non-unique local optimum
is harmless — and, historically, the per-cluster reception for $B_{\mathrm{LCB}}$. Because the
bilinear fit is non-convex and order-dependent (§4.2), the **authority** signal
$B_{\mathrm{LCB}}$ no longer reads reception off the embedding; the opinion **clusters** it needs
come instead from a deterministic spectral split of the (per-post-centred) reaction matrix — the
Community Notes / Polis PCA-on-votes construction, canonicalized so it depends only on the data,
not on input order or an RNG. ($D(p)=y_p^\top A y_p$ still rides the personalization embedding
$y_p$; it enters only the fuzzy $V(u,p)$, not the reproducible authority score.)

### 4.2 Do not trust the scalar intercept — reconstruct per cluster

The scalar $b_p$ is a linear proxy for diverse approval and diverges from the real target
when clusters sit asymmetrically about the origin (it happily rewards a post one cluster
loves and another merely tolerates). So we compute reception **per opinion cluster,
empirically**. Using deterministic opinion clusters $c$ (a canonical spectral split of the
reaction matrix — *not* $k$-means on the learned embedding), take each cluster's
**empirical reception** — the IPW-weighted mean of the signed reactions cluster-$c$ members
actually gave the post —

$$
r^{\mathrm{emp}}_{cp} \;=\; \frac{\sum_{u\in c}\, \omega_{up}\, r_{up}}{\sum_{u\in c}\, \omega_{up}},
\qquad n_{cp} \;=\; \sum_{u\in c}\, \omega_{up}
$$

(sum over cluster-$c$ users who reacted to $p$; $\omega_{up}$ is the §6 propensity weight), and
define bridged support by **shrinking each cluster's reception toward the global prior by how
much evidence that cluster gave, then aggregating across clusters**:

$$
\boxed{\;B_{\mathrm{LCB}}(p) \;=\; \operatorname*{agg}_{c}\; \tilde r_{cp},
\qquad
\tilde r_{cp} \;=\; \mu \;+\; \frac{n_{cp}}{n_{cp}+n_0}\,\big(r^{\mathrm{emp}}_{cp}-\mu\big)\;}
$$

where $\mu$ is the global mean reception. This is an empirical-Bayes (James–Stein /
DerSimonian–Laird) shrinkage: a cluster with little evidence ($n_{cp}\to0$) regresses to the
prior — its apparent dissent is sampling noise, not a tested divide — while a **well-observed**
disagreeing cluster keeps its low reception and pulls the score down. **A post is not credited
as bridging above the prior until it has survived contact with the people who would dislike
it.** The aggregator is a knob: the default $\operatorname{agg}=\text{nash}$ (geometric mean of
per-cluster agree-probabilities) is Polis's group-informed consensus — one opposed group blocks
high consensus, without the brittleness of a hard $\min$; $\operatorname{agg}=\min$ recovers
Ethelo's Rawlsian worst-cluster, and an Atkinson equally-distributed-equivalent interpolates
between them. Keep the scalar $b_p$ as a cheap pre-filter; rank on $B_{\mathrm{LCB}}$.

**Reception is empirical, not the bilinear reconstruction $\langle\bar x_c, y_p\rangle$ used
in an earlier version — for reproducibility.** Routing per-cluster reception through the
learned embedding imported that embedding's non-identifiability into the one number that is
supposed to be robust: the biased MF is bilinear and non-convex, so fitting it at fixed low
rank by ALS from a random init lands in an *order-dependent* local optimum. Relabelling ids and
shuffling the reaction order — the *same data* — then changed the bridged-support ranking
(only $\sim0.73$ Spearman-stable). The empirical cluster mean depends on the embedding only
through the **discrete, deterministic** cluster label, so $B_{\mathrm{LCB}}$ becomes reproducible
($\sim0.96$; the residual is the $\lambda$-weighting in $\omega_{up}$, kept for Sybil
resistance) — and on real data it is *more* faithful, not less (Community Notes helpful/not
AUC $0.86\to0.9996$; Polis cluster ARI $0.06\to0.61$). The full MF (§4.1) is retained for the
personalized value $V(u,p)$, where a local optimum is harmless; the *authority* signal
$B_{\mathrm{LCB}}$ no longer depends on it. (The $\tfrac12(\lVert X\rVert^2+\lVert Y\rVert^2)$
regularizer is the variational nuclear norm, so the completion $L=XY^\top$ *is* convex with a
unique optimum — the instability was the fixed-low-rank ALS, and over-ranking or a convex
Soft-Impute solver would reach it; empirical per-cluster means sidestep it entirely.)

Note the deliberate asymmetry in how uncertainty is used: the exploration pool (§8) samples
*high*-uncertainty posts optimistically to decide what to **audition**; $B_{\mathrm{LCB}}$
shrinks *low*-exposure clusters to the mean and credits only tested breadth to decide what to
**crown**. Optimism explores; conservatism rewards. This is what prevents imperfect estimates
from manufacturing false bridging.

**This form was arrived at empirically, against a deployed baseline (Appendix C.5),** through
*two* corrections. First, an early version used a subtractive penalty $B_{\mathrm{LCB}}=
\min_c[\hat r_{cp}-\beta\sigma/\sqrt{n_{cp}+1}]$. Benchmarked against X's Community Notes — the
one deployed bridging system whose helpful/not decisions we can treat as ground truth — that
version was *beaten by both* the scalar $b_p$ and a naive mean of signed helpfulness, because
with a rating-count $n_{cp}$ the $\min$-minus-penalty demotes notes *sampled* by fewer clusters
(noise) rather than notes *divisive* across clusters (risk) — it subtracted noise, not risk.
Replacing it with the exposure-weighted empirical-Bayes shrinkage and the nash aggregator
reached $b_p$ parity on CN and tracked genuine multi-group support far better on Polis. Second,
the *reception itself* was moved from the bilinear reconstruction to the empirical cluster mean
above (for the reproducibility reason just given): this both removes the order-dependence and
*improves* faithfulness — on a class-balanced Community Notes sample the empirical
$B_{\mathrm{LCB}}$ scores AUC $0.9996$ against `CURRENTLY_RATED_HELPFUL` (vs $0.86$ for the
reconstruction, $0.96$ for $b_p$). The one standing requirement is that $n_{cp}$ be a real
propensity-corrected exposure/evidence weight (§6). Two honest caveats remain: against CN's own
intercept-thresholded label a naive mean is near-unbeatable *by construction*, so the decisive
evidence for the aggregator is Polis, not CN; and with the global-mean prior $\mu$ an *untested
one-sided* post regresses to neutral rather than being predicted-low, so $B_{\mathrm{LCB}}$ is
deliberately lenient on unexposed content — it is the author budget (§8), not $B_{\mathrm{LCB}}$,
that bounds a high-volume firehose's total reach.

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
\lambda \;\leftarrow\; \tfrac{1-\delta}{n}\mathbf 1 \;+\; \delta\, T^\top (w \odot \lambda),
\qquad
w_v \;=\; \frac{H(T_{v\cdot})}{\log \deg^{+}(v)},
\qquad
T_{vu} \;=\; \frac{\sum_{p:\,a(p)=u}\, [r_{vp}]_+\,\cdot\,\mathrm{dist}(x_v, x_u)}
                  {\sum_{u'}\sum_{p:\,a(p)=u'}\, [r_{vp}]_+\,\cdot\,\mathrm{dist}(x_v, x_{u'})}
$$

where $w_v\in[0,1]$ is the **outgoing-diversity** weight — the normalized Shannon entropy of
rater $v$'s trust row — introduced below to defeat collusion rings; $w_v=1$ recovers plain
eigentrust.

$T$ must be **row-stochastic** — normalized over each *rater's outgoing* trust
($\sum_u T_{vu}=1$), not over each author's incoming — so a rater distributes one fixed unit
of trust among the authors it approves (classic EigenTrust). The choice is load-bearing for
Sybil starvation: under *column* normalization a Sybil author boosted by a single colluding
puppet would inherit that puppet's entire weight (its lone incoming edge normalizes to $1$),
whereas under row normalization an honestly-approved author accrues from *many* independent
cross-divide raters while a one-puppet Sybil receives only that puppet's fraction. The teleport
floor ($\delta<1$) makes this a contraction with a unique fixed point and floors
every rater's own weight (no one is zeroed). But the floor is also an attack surface:

**Row-stochasticity alone is not enough — a *ring* beats it, so we add outgoing-diversity
weighting (Appendix C.5).** Row normalization defeats the *one*-puppet attack but not a
coordinated ring, as validation on real signed votes (Wikipedia RfA) made concrete. The
teleport floor gives *every* account — puppets included — a baseline weight of $(1-\delta)/n$.
A ring of $K$ fresh accounts that each cast exactly one approval, all of it aimed at one target
author, would hand that target $\delta\!\sum_i\lambda_{\text{puppet}_i}\approx \delta
K(1-\delta)/n$ of transported mass: each puppet, being row-stochastic, forwards its *entire*
unit to the target, while honest boosters split their unit across the many authors they
genuinely approve. The puppets themselves stay starved, but the *target* harvests their pooled
baseline mass — the original ranker let the target climb from the $75$th percentile of
real-editor influence at $K=5$ to the $100$th at $K=100$. This is not a CHORD-specific bug: no
symmetric reputation function is Sybil-proof (Cheng & Friedman 2005), and the uniform teleport
$\tfrac{1-\delta}{n}\mathbf 1$ is exactly that symmetric part.

The fix, now the shipped default, is the transmit weight $w_v$ above: weight each rater's
*outgoing* trust by the normalized entropy of its trust row, so a **single-target rater —
every ring puppet, out-degree $1$ — transmits $w_v=0$**, forwarding none of its floor mass,
while an honest rater who spreads approval across many authors keeps $w_v\approx1$. On RfA this
collapses the ring target from the $100$th percentile back to the $0$th at every $K$, at *no*
cost to honest ranking (the influence order of real editors is unchanged, Spearman $1.0$),
because the attack's signature — concentrating one's entire outgoing unit on a single
beneficiary — is precisely what the entropy weight penalizes; an attacker who dilutes a puppet
across several authors to raise its entropy thereby splits its mass *away* from the target. A
small floor on $w_v$ keeps a genuine low-activity newcomer from being muted. This is a first
line, not a completeness proof: the theoretically complete defense is an *asymmetric* teleport
(a pre-trusted seed set à la TrustRank / EigenTrust's pre-trusted peers), which CHORD supports
as an optional seeded restart bound to the identity port (§11); combined with the identity
forge-cost it closes the residual. **Whichever estimator is used, weight by agreement with the
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
changes the chooser's own feed. Note $M=1$ is *not* the welfare-optimal setting: in the
closed-loop simulator (Appendix C.4) $M=1$ is **Pareto-dominated by an interior $M\approx0.7$**,
which delivers more genuine (quality $\times$ bridged) value *and* less divisiveness *and* no
loss of satisfaction — keeping a little personalization surfaces content that is both liked
and actually good, whereas the pure-bridging corner over-corrects. The default sits near this
knee, not at the extreme.

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
B_{t+1}(a) = \mathrm{clip}\!\Big(B_0 + \gamma\,(B_t(a)-B_0) + \eta\!\!\sum_{p:\,a(p)=a,\,p\in W_t}\!\! [\Phi(p)]_+\,E(p),\ \ 0,\ B_{\max}\Big)
$$

Two guards keep the replenishment well-behaved: the rectifier $[\Phi(p)]_+$ makes a
net-divisive post ($\Phi<0$) simply *fail to replenish* rather than draining the author's
floor $B_0$ (which would double-punish and could drive $B(a)$ negative), and the clip to
$[0,B_{\max}]$ keeps every budget bounded — a precondition the §9 bounded-regime argument
relies on. Firehose posting spreads a fixed budget thin; quality regenerates it. This inverts the
engagement logic (where each post is an independent virality lottery ticket, so volume is
rational) into one where posting more *dilutes you* unless it earns. The budget binds to
the **identity** port, not the raw account, so it cannot be sharded across sockpuppets.

Formally this is a **floored, capped replicator equation**: reach reproduces in proportion
to earned cross-cluster fitness, bounded below by the commons floor $B_0$ and above by
$B_{\max}$. It *enforces* an amortized fairness constraint of the Biega et al. (2018)
"equity of attention" kind — cumulative exposure tracking cumulative merit — but with the
merit being *tested bridging strength* and the refill *endogenous* to it, which is the
novel combination. Reading it as a replicator equation also surfaces four properties of its
time-dependence that the naive form gets wrong (each a gated refinement, off by default):

- **Memory ($\gamma$, #1).** The $\gamma\,(B_t-B_0)$ carry term is the author-side
  anti-ossification half-life mirroring rater recycling. With $\gamma=0$ (the original rule)
  a single quiet window resets any author to the floor, conflating "posted nothing this
  week" with "earned nothing" — the wrong forgetting schedule for irregular-cadence
  hobbyists. $\gamma>0$ carries earned standing across gaps.
- **Bifurcation at $\gamma+\eta\bar\Phi=1$ (#2).** For a fully-spending author, earnings
  $\approx\bar\Phi\,B_t$, so the recursion is linear with gain $\gamma+\eta\bar\Phi$ and
  fixed point $B^*=B_0(1-\gamma)/(1-\gamma-\eta\bar\Phi)$ — graded below the critical
  strength, runaway to $B_{\max}$ above it. So $\eta$ is a **phase-transition parameter**,
  not a smooth gain; set it to keep the gain (a shipped diagnostic) comfortably below $1$.
- **Streaming credit (#3, direction).** $\Phi(p)$ is not known at window close for
  slow-burn content — auditions close on *evaluation saturation*, not wall-clock. Batching
  credit at window boundaries therefore either credits noisy provisional $\Phi$ or lags by
  the saturation time, disadvantaging exactly the long-form the depth mechanism protects.
  The clean form is **incremental (leaky-bucket) credit** as the Thompson posterior on
  $\Phi(p)$ tightens — credit flowing at the rate evidence arrives, which also removes
  window-boundary gaming. Noted as the intended refinement; the batch rule ships today.
- **System-wide conservation (#4).** Per-author issuance $\eta\sum[\Phi]_+E$ is
  *procyclical*: a high-engagement window (breaking news lifts everyone's $\Phi$) inflates
  total issuance, and once aggregate budget exceeds slot supply the constraint silently
  stops binding. The share-based option issues a **fixed aggregate pool** per window,
  distributed by *relative* realized strength — monetary policy for attention rather than a
  fixed money-printing rate — so "conserved" holds system-wide, not just per author.

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

There is in fact a **third timescale**: the author-budget recursion (§8). Its refill rate
$\eta$ is itself a **performativity gain** — a larger $\eta$ makes the visibility an author
receives, and hence the data distribution, more sensitive to the ranker's own allocations,
pushing *against* the Perdomo condition. So the budget update belongs on the slow timescale
with $\lambda$ (budgets should change no faster than the actor), and $\eta$ (together with
the memory $\gamma$, since the replicator gain is $\gamma+\eta\bar\Phi$) belongs *inside*
the measured performativity ratio of §9.3, not tuned independently of it. The exploration
anchor helps here too: randomized exposure is allocation the author did not earn, which
directly damps the budget→data feedback.

### 9.3 Stability as a monitored runtime property

Because global convergence is **not** guaranteed in the nonconvex regime, the right target
is a **bounded stationary regime**, held by: persistent excitation ($\epsilon\ge
\epsilon_{\min}$, so the system never stops sampling regions it stopped showing); slow knob
changes; SNIPW + clipping to bound gradient variance; under-relaxation and two-timescale
separation. Run a **controller on the estimator's own concentration**: track effective
rater count $(\sum\lambda)^2/\sum\lambda^2$ (or $\mathrm{Gini}(\lambda)$); if it collapses,
automatically raise the teleport floor $\delta$ and $\epsilon_{\min}$. The controller's
response now genuinely feeds back into the estimator — the next window's eigentrust and
exploration floor read the controller's adjusted $\delta,\epsilon_{\min}$, not the static
config — and relaxes to the configured values when concentration is healthy. In practice the
other defenses hold $\mathrm{Gini}(\lambda)$ so far below the ceiling that it seldom fires (a
noted open tuning question, §13#12), so it is best read as a bounded safety net rather than an
always-active regulator. The exploration pool is therefore load-bearing four times over —
provider fairness, cold-start, causal identification, and estimator stability — and its rate is
a floored system invariant, not a user preference.

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

- **Sybil / sockpuppets** — the visibility budget binds to identity, not accounts, and trust
  propagation gives fresh accounts ≈ zero weight *as raters*. A naive teleport-floor eigentrust
  had a residual hole — a ring of fresh accounts all approving one target let that *target*
  harvest the puppets' floor mass — which the **outgoing-diversity transmit weight** now closes:
  a single-target puppet forwards zero trust, so the ring collapses to the floor at any size
  (§5, Appendix C.5, validated on RfA). Sharding across accounts therefore gains nothing, for
  puppets *or* their beneficiary; the optional seeded-teleport asymmetry plus the identity
  port's forge-cost (§11) close the theoretical remainder. **Residual (found in the simulator,
  App C.4):** a *distributed* ring that camouflages puppets across clusters — each rating
  genuine content to embed in a real cluster, then boosting one target — manufactures fake
  cross-cluster support that defeats the $B_{\mathrm{LCB}}$ min *and* is invisible to the
  out-diversity weight (the puppets are not single-target *raters*). A plain co-approval discount
  only partially contains it, but a **cluster-spread-gated loyalty discount** does — it keys on
  the one act camouflage can't hide (the same accounts approving *every* target post over time)
  and gates by the bloc's opinion-cluster spread, driving the ring's amplification below a
  legitimate author's; the exploration-anchor cap and spectral spike-removal are the principled
  hardenings still open (§13.10).
- **Brigading** — two independent defenses: a brigade of fresh accounts has no cross-divide
  trust path, and a brigade *creates* a split distribution that the divisiveness term and
  the $B_{\mathrm{LCB}}$ min-over-clusters penalize. Gaming lowers the score.
- **Bridging-bait (Goodhart)** — shallow universal content (cat memes) can score high
  bridged support. The depth handling resists it structurally: an **estimated** depth $q_p$
  both *rewards* genuine depth and multiplicatively **gates** a shallow post's positive bridged
  support toward a floor, so breadth alone cannot buy a crown. Crucially $q_p$ is *earned*, not
  an author-set feature — it is the empirical-Bayes-shrunk, λ-weighted, per-cluster mean of a
  separate opinion-independent *vouch* channel (§13#11), so a baiter cannot forge it and must
  instead obtain genuine cross-cluster vouching (whereupon the collusion defenses apply). In the
  simulator, turning the earned gate on raises delivered true value and reduces a bait author's
  reach; the effect is smaller than with an oracle depth feature — the honest cost of estimating
  the signal rather than trusting the author to report it.
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
   performative effects the system can oscillate; we monitor rather than prove. The
   "weak-performativity" assumption is now *measurable* rather than asserted (E1): the
   empirical Lipschitz of the performative map — reaction-distribution shift per unit
   ranking perturbation — is estimable from the exploration slice (`empirical_lipschitz`),
   scales with the performativity rate in the simulator, and can be fed to the §9.3
   controller to hold the sensitivity/curvature ratio below the Perdomo threshold.
2. **Unobserved confounding in the propensity model** (§6) is the softest load-bearing
   wall — and the residual is *bias*, not variance. The variance failure (weights exploding
   as $\pi\to0$) is handled by SNIPW, doubly-robust estimation, the $1/\epsilon$ clip, and
   the $B_{\mathrm{LCB}}$ pessimism. What none of these touch is a hidden variable that drives
   *both* exposure and reaction but is absent from the model: no calibration, clipping, or
   doubly-robust wrapping removes confounding bias, because the estimand itself is
   misidentified. But CHORD has an asset most systems lack: for every post that receives
   ε-exposure it observes *both* confounded organic and unconfounded randomized reception on
   the same content, so the gap is a direct estimate of the total bias — including the
   unobserved part. So rather than only *bound* the confounding (Rosenbaum) we can **calibrate**
   it (E2): fit a per-cluster bias model $r_{\text{exp}}\approx a_c+b_c\,r_{\text{org}}$ on the
   paired ε-slice observations (accumulated across windows) and predict unconfounded reception
   everywhere — proximal-causal in spirit, the ε-slice playing the negative-control exposure.
   On Coat's dense random-exposure block this beats IPW at de-biasing held-out items ($|$err$|$
   $0.19$ vs $0.24$) and *transports* to items with no ε-coverage. The honest caveat is the
   converse of its strength: it **spends ε**, and at the bare exploration *floor* the paired
   sample is too thin — in the closed-loop simulator the effect is within seed noise — so the
   calibrator is gated (`bias_calibration`) and its benefit scales with how much randomized
   traffic it is *allocated* (the portfolio point, #13). Where ε coverage is absent entirely,
   report an **E-value** per crowning decision (the minimum confounding strength that would
   overturn the verdict — one line on numbers already computed) and gate crowning on it. So
   this moves from "named, bounded" toward "calibrated where ε reaches, E-value-gated where it
   does not" — with residual bias only where the bias model itself fails to transport.
3. **Bridging is non-monotone in audience.** $B_{\mathrm{LCB}}$ certifies bridging over the
   *exposed* set; a post bridging at 10K may divide at 10M. Never fully closed, but now
   structurally bounded by an **amplification collar** (E3, `amplification_collar`): a post's
   realized strength — hence its budget and reach — is throttled when reach outruns its
   tested audience, $E(p)>\kappa\,n_{\text{tested}}$, so amplification proceeds in rungs that
   re-certify $B_{\mathrm{LCB}}$ on a larger tested set before each expansion. In a
   rare-opposition test $B_{\mathrm{LCB}}$ falls monotonically as the tested audience grows,
   so the collar halts a lucky small-sample post before it over-amplifies.
4. **Low-dimensional opinion space** may under-represent true plurality; divisiveness along
   an unmodeled axis leaks into $b_p$. $d$ is a real bias-variance knob. This is now a
   *testable null* rather than a silent failure (E4, `whiteness_gate`): crowning is gated on
   a **residual-whiteness test** — Moran's I of a post's rank-$d$ residuals against the
   co-reaction graph, permutation-tested. A post dividing along a hidden axis has
   significantly autocorrelated residuals (flagged $p{<}0.01$) and is demoted; a genuine
   bridge is white ($p{\approx}0.5$). Rising rejection rates are a principled "$d$ too low"
   diagnostic.
5. **Peer-prediction collusion** and the high-precision clique (§10) — a structural
   equilibrium problem, not a bug.
6. **Recycling is mildly farmable** in principle (acting under-served) — now much less so
   (E6, `recycling_offpolicy_verify`): the λ-boost is credited only when apparent
   under-service is *corroborated off-policy* — the user realizes more value on ε-slice items
   than on their personalized feed. A farmer who acts dissatisfied but does not actually
   prefer the exploration content shows no gap and gets no boost; a genuinely under-served
   user keeps it. The signal verifies itself against randomized ground truth.
7. **Distinguishing harmful from benign divides** ($A$'s weighting) is a normative,
   instance-level choice with no purely technical answer.
8. **Sybil rings — mostly closed, with a residual** (§5, validated in C.5). A naive
   teleport-floor eigentrust let a single-target ring's *beneficiary* harvest the puppets'
   pooled floor mass (top-percentile influence as the ring grows). The shipped outgoing-diversity
   transmit weight closes this — a single-target puppet forwards nothing, and on RfA the ring
   collapses to the floor at every size with honest ranking preserved. The residual is
   theoretical: Cheng & Friedman prove no *symmetric* reputation function is Sybil-proof, so full
   resistance needs the *asymmetric* seeded teleport (supported, bound to the identity port) plus
   the forge-cost — and a sufficiently patient attacker who diversifies each puppet across many
   authors trades ring stealth for diluted impact. Hardened, not proven impossible.
9. **$B_{\mathrm{LCB}}$ now earns its complexity, and is reproducible** (§4.2, validated in C.5).
   Two corrections got it here. The earlier subtractive $\min_c$ penalty, fed raw rating counts,
   was beaten on Community Notes by both $b_p$ and a naive mean (it subtracted noise, not risk);
   exposure-weighted empirical-Bayes shrinkage plus the nash aggregator fixed that. Then a
   *reproducibility* failure surfaced under property testing: because per-cluster reception was
   read off the **non-convex bilinear embedding**, relabelling ids and shuffling reaction order —
   the same data — changed the ranking (only $\sim0.73$ Spearman-stable), and this fragility
   propagated into the tuned simulator claims. Moving reception to the **empirical** IPW-shrunk
   cluster mean, with **deterministic spectral clusters**, made $B_{\mathrm{LCB}}$ reproducible
   ($\sim0.96$) *and more* faithful (CN AUC $0.86\to0.9996$, Polis ARI $0.06\to0.61$). Residuals:
   against CN's intercept-thresholded label a naive mean is near-unbeatable by construction (so
   Polis, not CN, is the aggregator's court of appeal); the shrinkage is only as good as the
   evidence weight $n_{cp}$; and the global-mean prior makes $B_{\mathrm{LCB}}$ lenient on
   *untested* one-sided content (a firehose post regresses to neutral, not below). That last gap
   is now closed by a **hierarchical author×cluster prior** (E9): shrink each cluster's reception
   toward the author's own decayed history *in that cluster* → the cluster mean → $\mu$, so an
   untested post from an author cluster $c$ has consistently disliked regresses to that low prior
   and $B_{\mathrm{LCB}}$ predicts-low *before* the budget bites, while a well-observed post
   overwhelms the prior at the $n_{cp}/(n_{cp}+n_0)$ rate (self-correcting for reformed authors).
   It is deterministic, so reproducibility is untouched; in the simulator, turning it on raises
   delivered true value $\sim\!28\%$ and suppresses firehose reach relative to quality (gated on
   `hierarchical_prior`, since flipping the default would re-tune the $\mu$-calibrated sim suite).
   The full reproducibility fix is the empirical means; an over-ranked or convex (Soft-Impute)
   completion for the *personalization* embedding is a noted, not-yet-shipped further step.
10. **The camouflaged distributed ring — defended, with a residual** (§10, found *and fixed* in
    the simulator, App C.4). A ring that embeds puppets in *every* opinion cluster (by rating
    genuine content) and has them all boost one target fabricates cross-cluster support that
    beats $B_{\mathrm{LCB}}$'s min *and* slips past the out-diversity λ weight (the puppets are
    not single-target raters); a plain co-approval discount is only partial (camouflage dilutes
    it). The attack's mechanism is a **rank-1 common-mode lift of the shared intercepts** $b_p,
    b_a$: the puppets scatter across clusters so their directional pull on $y_p$ cancels, and
    only their shared positive residual survives, lifting every cluster's reception equally. The
    shipped defense keys on the one act camouflage cannot hide — the *same accounts*
    disproportionately supporting the target over time — and gates it by how **opinion-dispersed**
    that loyal bloc is, so a camouflaged ring (scattered across the divide) is penalized while a
    coherent niche fanbase is not; in the simulator this drives the ring's amplification from
    $2.8\times$ a legitimate author's reach back below $1\times$, and on the real Community Notes
    slice it detects a $K{=}40$ ring (manufactured-fraction $\approx0.82$) and reverses its
    inflation. Two design points earn their keep. Dispersion is measured on the **continuous
    opinion-axis coordinate** (the spectral top vector), not a discrete cluster label — on
    weakly-divided data the 2-way split is degenerate ($\approx$ all-one-cluster), which blinds a
    cluster-entropy gate but not the coordinate spread. And loyalty is scored **continuously**
    (support relative to the most-loyal supporter), with *no* hard threshold, so an adaptive ring
    cannot approve *just under* a cutoff to dodge detection while still inflating. The residual is
    sharp and, we argue, **fundamental**: a ring can still evade by spreading so thin that each
    puppet supports $\lesssim$ one of the target's posts — at which point every account is
    *statistically indistinguishable from a genuine casual supporter*, penalizing it would
    penalize the dispersed organic support the system exists to reward, and its inflation is in any
    case bounded to what genuine dispersed support would produce. A second, complementary
    defense is also built — an *exploration-anchor cap* that bounds each cluster's reception by
    the unconfounded ε-exploration reception (§6.2); it *de-confounds* organic reception (raising
    delivered true value) and removes the ring's common-mode lift $K$-independently, but cannot
    *alone* push a ring below parity (a near-origin target's true reception equals genuine broad
    content's) and needs more randomized traffic than the floor provides — so it is paired with
    the loyalty penalty rather than used alone. *Spectral spike-removal* — deflating the
    rank-1 boost block from the residual — was also implemented and yielded an instructive
    **negative** result: per window, by reaction pattern alone, a coordinated ring is
    *indistinguishable from genuine cross-cluster consensus* (both are rank-1 residual blocks,
    and a genuinely-loved author's is the *larger*), so deflation suppresses the very bridging
    content the system exists to reward. The robust discriminator is therefore necessarily
    **temporal** (the loyalty signal — do the same accounts co-occur only on this target over
    time?) or out-of-band (identity/provenance); RPCA is doubly wrong (the boost lands in its
    low-rank part and is absorbed). Timing/provenance and costly identity remain the ultimate
    backstops, and this is the collusion analogue of the high-precision clique (#5).
11. **The anti-bait depth signal was forgeable — fixed; depth is now earned.** An adversarial
    test found the obvious hole in the original design: depth was a per-post *feature* the author
    set, so forging a high score made shallow content (value $0.43$) beat genuine quality ($0.36$)
    — a §12 wall violation (an authority quantity sitting on the author-settable side). The fix
    makes depth an **estimated latent** $q_p$, computed the way $B_{\mathrm{LCB}}$ computes bridged
    support: the empirical-Bayes-shrunk (neutral-prior), λ-weighted, per-cluster mean of a
    *separate merit/vouch channel* — vouches that a post is substantive, opinion-independent so
    genuine depth earns *cross-cluster* support while a broadly-liked shallow bait earns
    anti-vouches. An author cannot set it; it emerges from *others'* dispersed vouching and
    inherits the collusion defenses (a bait's fake vouchers face the same loyalty/out-diversity/
    exploration machinery as a boost ring). In tests, forging the feature now does nothing and
    more approval *breadth* cannot rescue a shallow bait; in the closed loop, turning the
    (earned) gate on still raises delivered true value and reduces bait reach — with a smaller
    effect than the old oracle feature, the honest cost of a signal estimated from noisy vouches
    rather than read off ground truth. The residual is the same coordinated-clique of #5/#10 (a
    patient adversary who accrues honest vouch-weight then farms fake merit votes), for which a
    determinant-mutual-information (DMI) peer-prediction weight on the vouch channel — replacing
    the correlate-with-the-crowd quality heuristic — is the noted next hardening. A second,
    independent depth channel further raises the forgery bar (E11, `saturation_depth_prior`):
    the audition **saturation trajectory** — bait's reception variance collapses fast, slow-burn
    depth decays ~5× slower — is a behavioral prior a forger would have to fake wholesale.
    Three channels (vouches, saturation shape, and — where read behaviour is logged —
    propensity-corrected dwell) make the attack require compromising independent modalities.
12. **The §9.3 concentration controller is wired but dormant.** It now genuinely feeds its response
    back into the estimator (an earlier build computed a response the loop never read), and a
    forced tighten measurably flattens $\lambda$. But in every scenario tried, baseline
    concentration ($\mathrm{Gini}(\lambda)\approx0.06$) sits far below the ceiling that would
    trigger it — the teleport floor, out-diversity transmit weight, and recycling already bound
    concentration well below where the controller engages. The dormancy is addressed by
    reframing the trigger as a **change-point alarm** rather than a fixed level (E12,
    `controller_cusum`): a CUSUM on $\mathrm{Gini}(\lambda)$ drift against a slow rolling
    baseline, with the ceiling *data-derived* ($h\sigma$ of the baseline). It fires on a
    concentration attack (Gini $0.08\to0.24$) that never approaches the $0.6$ level ceiling —
    active where the level guard is dormant — while staying silent on healthy noise. (It
    detects a *shift*, so a system that begins concentrated shows no drift to catch — which is
    change-point detection working as intended; the realistic attack transitions from healthy.)
13. **The randomization portfolio — ε is a scarce resource, not just a floor.** Count the jobs
    the exploration slice now holds: causal identification (§6.2), cold-start and provider
    fairness (§8), estimator-stability excitation (§9.3), *and* every honest answer to the
    problems above — confounding calibration (#2/E2), collusion and ring audit (#5/#10, E5/E12),
    performativity probes (#1/E1), and amplification-collar re-certification (#3/E3). These all
    draw on the *same* floored budget ε, and they compete. So ε warrants an explicit
    **allocation policy**, which is itself a bandit over information value: `RandomizationPortfolio`
    keeps a floor on every demand — the property that makes identification, fairness and stability
    hold, so ε stays a floored invariant *per arm* — then water-fills the remainder toward the
    demand whose current marginal value is highest. Under a fixed ε budget with shifting needs
    (cold-start early, an audit spike when an attack begins), the floored bandit captures ~9% more
    total information value than a uniform split and ~29% more during the attack window,
    approaching the oracle water-fill. This both subsumes several entries above (they are demands
    *on* the portfolio) and sharpens the paper's core argument: the reason ε is a floored system
    invariant is not identifiability alone — it is that *every* honest answer to CHORD's open
    problems turns out to spend randomized exposure, so the commons must be budgeted, not merely
    guaranteed.

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

### 13.11 Optional refinements and their composition trade-offs

Beyond the shipped defaults, the reference implementation carries a set of **individually
validated but optional** refinements (config knobs, default off), each answering one of the
problems above. A deliberate re-validation pass tried to graduate them to defaults and found
a result worth stating plainly: **they do not compose cleanly, and adding a defense is not
free.** Each refinement, switched on, shifts the tuned closed-loop dynamics, and several
interact — sometimes destructively — with the very defenses they sit beside. The headline
guarantees (CHORD beats engagement on welfare; the effective ring is contained) survive, but
specific sub-mechanism guarantees do not survive arbitrary combination. The trade-offs:

| Refinement (knob) | What it buys | The trade-off it introduces |
|---|---|---|
| **Hierarchical prior** (E9, `hierarchical_prior`, §13#9) | $B_{\mathrm{LCB}}$ predicts-low on an *untested* one-sided firehose before the budget bites; +28\% delivered value in isolation | The prior is on **approval history**, so it also *props up* a broadly-approved but shallow **bait** and a distributed-**ring target**, blunting the depth defense and ring containment; worse, rewarding in-cluster approval consistency *raises author extremity above the engagement baseline* — it fights the "no extremists" goal. A **quality-history** prior (on the vouch channel, §13#11) rather than approval-history is the redesign this points to. |
| **Residual-whiteness gate** (E4, `whiteness_gate`, §13#4) | Flags a post that divides along an unmodeled axis (Moran's I, permutation-tested) | Residuals correlate with the cluster structure, so as a *gate* it false-positives on most crowned posts in any clustered population. Excellent as a **diagnostic**, unsafe as a default gate. |
| **ε-slice bias calibration** (E2, `bias_calibration`, §13.2) | De-confounds organic reception with a per-cluster bias model; beats IPW and **subsumes** the exploration-anchor cap | Redundant with (and superior to) the anchor cap — running both double-corrects. It also *shifts the M-frontier*: better de-confounding makes **pure bridging ($M{=}1$) deliver the most value**, moving the interior optimum — a benign shift, but one that invalidates the $M{\approx}0.7$-dominates result calibrated on the un-de-confounded system. |
| **Amplification collar** (E3, `amplification_collar`, §13#3) | Throttles reach that outruns tested audience (staged rungs) | Interacts with the budget's own reach allocation; changes firehose/ring reach dynamics in combination. |
| **Off-policy recycling verify** (E6, `recycling_offpolicy_verify`, §13#6) | Withdraws the λ-boost from a recycling farmer (ε-corroborated) | Changes λ, hence IPW, hence B_LCB — shifts tuned ring/welfare estimates in combination. |
| **CUSUM controller** (E12, `controller_cusum`, §13#12) | An *active* drift alarm where the level ceiling is dormant | *Not inert*: a ring drives Gini drift, so it fires in adversarial scenarios and tightens the controller (§9.3 wiring), changing λ/reach exactly where the ring tests are calibrated. |
| **Budget memory / share / streaming** (`budget_memory`/`budget_share_based`/`budget_streaming_credit`, §8) | Cadence tolerance (γ carry); system-wide conservation; leaky-bucket credit for slow-burn | Memory lets a *persistent firehose accumulate* budget, reversing the §8 dilution; share-based dilutes the welfare gain and (with memory) courts the $\eta\bar\Phi$ bifurcation; each changes the budget-to-reach dynamics the firehose test pins. |

Two structural lessons fall out. First, **several refinements target the same failure mode
from different angles and therefore overlap or conflict** — E2 vs. the anchor cap (both
de-confound), E9 vs. the depth defense (approval vs. quality), budget memory vs. the firehose
dilution (cadence tolerance vs. flood suppression). The design space is *coupled*, not
modular. Second, this is why they ship as **knobs, not defaults**: the tuned guarantees hold
for the validated configuration, and enabling any refinement is a deliberate, per-deployment
choice that requires its own re-validation — which is the honest posture for a mechanism whose
interactions with the rest of the system are real. The one refinement closest to a safe default
is E2 (a strict de-confounding improvement whose only cost is the benign M-frontier shift); the
one most in tension with the design's goals is E9 (until its prior is moved from approval to
quality). Both conclusions came from the simulator's counterfactual, ground-truthed loop —
which is exactly the tool Appendix C.4 exists to provide.

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
| Propensity / MNAR (§6) | IPS / doubly-robust recovery | **Yahoo!R3, Coat, KuaiRec/KuaiRand** | The only datasets with a randomly-exposed (MAR) holdout = the unconfounded anchor. KuaiRand adds timestamps/features. **A true MAR block is essential, not a convenience** — see C.3: on Coat's real random-exposure test set IPW recovers the ranking, but running the same harness on a MovieLens slice (organically MNAR, no MAR block) it does *not*, because the "holdout" inherits the base selection bias. |
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

**The base matrix in step 1 must be genuinely MAR (or near-complete) — this is a correctness
condition, not a convenience, and it is easy to get wrong (Appendix C.5).** The harness scores
step 5 against a "hidden full matrix" taken as ground truth; if that matrix is itself the
product of organic self-selection (as with a MovieLens or MovieLens-style slice, where people
rate what they chose to watch), then the held-out cells carry the *same* unmodelled selection
bias as the training cells. Correcting only the *injected* logging policy $\pi^{\text{log}}$
then adds inverse-propensity variance without removing the pre-existing confound, and IPW can
match or *underperform* the uncorrected fit — not because the correction is wrong but because
the ground truth is contaminated. Empirically this is exactly what happens: on **Coat**, whose
test block is a real uniformly-random exposure, IPW improves the unbiased ranking; run the
identical harness on a dense **MovieLens** slice and it does not. Use the semi-synthetic recipe
only on a dataset with a true random-exposure block (Coat, Yahoo!R3, KuaiRand) or a near-fully-
observed matrix; treat any organically-sparse matrix as a negative control, not a validation.

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

The reference implementation ships this simulator (`chord/simulator/`, see its `SIMULATOR.md`)
built to three commitments that make it a real test rather than a flattering one:
**counterfactual** — every ranker (CHORD, engagement, chronological, random, and a cheating
oracle) runs its own closed loop on the same seeded world, so results read "what if we had
ranked this way"; **non-circular** — the data-generating process is deliberately *not* the
model the estimator fits (a hidden opinion axis beyond $d$; a toxicity channel that drives
engagement while quality — genuine value — barely moves reactions, so engagement $\neq$ value
and bridging-bait can exist); and **ground-truthed** — because the world is synthetic it scores
what no live system can, the true bridged value delivered and the polarization exposed. The
headline counterfactuals (seed-averaged): against an engagement ranker on the same world, CHORD
delivers more true (quality $\times$ bridged) value and less toxicity/divisiveness at a modest
satisfaction cost, and recovers the true opinion geometry better (its exploration anchor keeps
the estimate identifiable in-loop); with strategic author-agents, engagement entrenches partisan
extremity over time while CHORD does not; a sybil ring that boosts a target's posts *gains* reach
under engagement but *loses* it under CHORD as the ring grows (B_LCB's min-over-clusters reads the
ring as an outlier group); and the conserved budget (§8) dilutes a firehose author's per-post
reach. One honest in-loop finding: the §5 out-diversity weight, which fixes the rater-influence
ring, does not by itself stop this *content-boost* ring — the keystone does.

### C.5 Results of the reference validation suite

The reference implementation ships an opt-in suite (`validate/`) that runs the static-component
checks above against real public data — Coat, Polis, MovieLens-100K, Wikipedia RfA (SNAP), and
a dense k-core slice of X Community Notes — with datasets committed via Git LFS. It is built to
*surface* failures rather than confirm the design: claims that hold are assertions; claims that
fail are recorded as documented findings, not tuned away. The first run is summarized here
(indicative numbers, $d=2$, averaged over seeds); several claims held and three did not.

**Held.** (i) On Coat's real MAR holdout, IPW correction improves the unbiased ranking (NDCG@5
$0.714\!\to\!0.718$, AUC $0.657\!\to\!0.663$) — modest but in the predicted direction (§6). (ii)
On Polis, $B_{\mathrm{LCB}}$ tracks genuine cross-group support well (Spearman $\approx 0.71$
against the minimum-across-groups mean vote), and the random-$\epsilon$ anchor demonstrably buys
identifiability (unbiased NDCG falls monotonically as the anchor $\to 0$, §6.2). (iii) The §5
influence distribution is concentrated, not uniform ($N_{\text{eff}}/n\approx 0.4$ on RfA).

**Findings — claims that did not survive real data.** (F1, §5) Teleport-floor eigentrust
starves individual sybils but *not a ring*: a $K$-account ring pointing at one target lifts that
target from the $75$th percentile of real-editor influence at $K=5$ to the $100$th at $K=100$
(the ring-harvesting analysis in §5). (F2, §4.2) On Community Notes, reconstructed
$B_{\mathrm{LCB}}$ (AUC $0.93$ vs. `CURRENTLY_RATED_HELPFUL`) is beaten by both the scalar $b_p$
($0.95$) and a naive mean of signed helpfulness ($0.9994$) — the pessimism penalty subtracts
noise when $n_{cp}$ is not a real exposure count. (F3, §6/C.3) The semi-synthetic IPW harness
recovers the ranking on Coat but fails on an organically-MNAR MovieLens slice, because that
slice has no true MAR block. Two weaker signals: Polis cluster reconstruction is strong on some
conversations (vTaiwan ARI $0.59$) and near-chance on others (football-concussions $0.04$), and
divisiveness $D(p)$ tracks group spread except on very low-conflict conversations.

**Both headline findings were then fixed, and the fixes are the shipped default.** After a
literature survey (trust-metric and welfare/risk/robust-statistics), candidate repairs were
prototyped against the *same* benchmarks and promoted into the core. F1: the outgoing-diversity
transmit weight (§5) collapses the RfA ring from the $100$th percentile back to the $0$th at
every $K$, with honest ranking unchanged (Spearman $1.0$). F2: exposure-weighted empirical-Bayes
shrinkage plus the nash aggregator (§4.2) lift $B_{\mathrm{LCB}}$ to $b_p$ parity on a balanced
Community Notes sample and to Spearman $0.97$ (from $0.80$) against Polis's true cross-group
support. The validation suite's F1/F2 tests, originally written to *document* the failures, now
run green against the fixed code and guard against regression; F3 remains a standing caveat on
the semi-synthetic recipe (use a genuinely-MAR base matrix), not a code defect. This
find-it-then-fix-it loop — real data surfaces a failure, cross-disciplinary math supplies the
repair, the same benchmark confirms it — is exactly what Appendix C is for.

**A deeper validation pass** then stress-tested the whole stack beyond claim-reproduction:
property/metamorphic invariants over randomly-generated worlds, config-grid robustness sweeps
(the welfare win holds $8/8$ configs), the dynamic §9 claims, and *adaptive* adversaries that
optimize against the live defenses. It surfaced and drove fixes for three structural issues the
first pass missed. (i) *Reproducibility*: relabelling ids and shuffling the reaction order — the
same data — changed the $B_{\mathrm{LCB}}$ ranking (only $\sim0.73$ Spearman-stable), because
per-cluster reception was read off the non-convex bilinear embedding; moving to **empirical**
IPW-shrunk cluster means with **deterministic spectral clusters** made it $\sim0.96$-stable and
*more* faithful (CN AUC $0.86\to0.9996$, Polis ARI $0.06\to0.61$; §4.2, §13#9). (ii) *The §9.3
controller was inert* — it computed a response the loop never read; now wired (§13#12). (iii)
*The collusion loyalty gate assumed a balanced split* and broke once the spectral split proved
degenerate on the dense CN core; re-keyed onto the **continuous opinion-axis coordinate** with
**continuous loyalty**, it detects a real Community-Notes ring (manufactured-fraction
$\approx0.82$) and, under an adaptive partial-approval attacker, contains every *effective*
attack — leaving only the thin-ring evasion that is provably indistinguishable from genuine
dispersed support (§13#10). (iv) *The anti-bait depth signal was forgeable* — an author set it
as a feature; it is now an earned latent $q_p$ estimated from a separate opinion-independent
vouch channel (§13#11), so forging the feature or buying more approval breadth no longer works,
at the honest cost of a softer signal than an oracle feature.

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
