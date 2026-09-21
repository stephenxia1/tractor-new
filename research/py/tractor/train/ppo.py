"""PPO over a padded, masked candidate distribution with a privileged critic.

Three things in this file are specific to Tractor rather than to PPO,
and each is a place where the textbook version would be quietly wrong.

*The action space is a padded list of legal moves.* A decision offers
between 1 and 128 candidate multisets, the tensor is always 128 wide,
and the network overwrites the logits of the unused slots with
``MASK_LOGIT``. The chosen action is a slot index, so the policy ratio
is a plain gather -- but the *entropy* is not. Entropy summed over 128
slots is entropy over a distribution that includes padding, and since
14.8% of decisions have exactly one legal move, an entropy bonus that
counts padding pays the network to keep probability mass on slots that
do not exist. ``masked_entropy`` renormalises over the legal slots
before it measures anything, so the bonus is the entropy of the
distribution the sampler actually draws from and nothing else.

*The critic is privileged.* It reads the four hands and the kitty, which
no seat can see, so it is only usable as a baseline -- and it is a
separate tower whose gradient reaches the actor only when asked. The
update calls the net with ``detach_critic=False``: during learning the
critic loss *should* shape the shared history encoder, which is the
whole reason the two towers share one. Collection uses the default,
detached, path.

*There are three auxiliary heads to feed.* They hang off the actor's
own summary vector, they cost one small matmul each, and a head that
appears in no loss is 150,000 parameters of pure waste: computed on
every forward, discarded on every forward, and never trained.

The *belief* head predicts, per card type, a distribution over the five
places a copy the acting seat cannot see might be: the three other
seats, the kitty, the table. Four of those come free from the
privileged tensor and the fifth from the observation's own played
planes, so cross-attention aside the auxiliary signal costs nothing to
produce. ``belief_target`` is the part worth reading twice -- in
particular why the acting seat's own hand is excluded and why the
played slice spans eight planes rather than four.

The *own-value* head is the seat's own estimate of the position, and it
regresses on the same returns the privileged critic does. It is never
used as a baseline -- the critic is -- so it is purely a demand on the
representation: the summary has to carry enough of the position to say
how the hand is going from a view that cannot see three of the hands.

The *outcome* head classifies the level swing the hand ends on, and it
is the one with a label problem. See ``losses`` for why it trains on
``done`` rows only and why that is better than the alternative.

On numerics. The forward runs under ``torch.autocast`` in bfloat16 when
the device supports it, and every loss is reduced in float32 outside the
autocast region: a mean over a few hundred bf16 terms has about three
decimal digits, which is not enough to tell a converged policy loss from
a diverging one. ``MASK_LOGIT`` (-1e9) does not fit in bf16 either, but
the model already promotes to float32 before it masks, so the log-probs
arriving here are float32 whatever the autocast state.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ..nn.features import PLANE_KITTY, PLANE_PLAYED
from ..nn.model import NUM_OUTCOMES, NUM_PRIVILEGED_ROWS, ActorCritic, ObsBatch
from .gae import num_rows
from .rollout import MAX_LEVEL_GAIN

__all__ = [
    "PPOBuffer",
    "PPOConfig",
    "PPOLearner",
    "belief_target",
    "explained_variance",
    "masked_entropy",
    "outcome_class",
]

# Smallest divisor allowed when renormalising a probability vector. The
# candidate distribution is float32 and already normalised, so this only
# ever fires on a degenerate row (no legal slot at all), where it turns a
# NaN into a zero-entropy row.
_EPS = 1e-12

# Bound on the log policy ratio before it is exponentiated. float32
# ``exp`` overflows to +inf at 88.7, and a single infinite ratio makes
# the surrogate mean infinite, the gradient non-finite and the whole
# step a loss. 20 is far outside anything PPO can legitimately produce:
# the clip range is 1 +- 0.2, a log-ratio of +-0.182, so by e^20 the
# ``min`` has long since selected the clipped branch for a positive
# advantage and the pessimistic branch for a negative one. Clamping
# there takes nothing from a legitimate row and turns a run-ending
# infinity into a row that is merely bounded.
_MAX_LOG_RATIO = 20.0

# ``sim.state.hand_result`` returns at most a three-level swing either
# way, and ``rollout`` divides by that to reach [-1, 1], so the seven
# outcome classes are exactly the seven values of the signed swing.
if NUM_OUTCOMES != 2 * int(MAX_LEVEL_GAIN) + 1:
    raise RuntimeError(
        f"NUM_OUTCOMES is {NUM_OUTCOMES} but a level swing of at most "
        f"{MAX_LEVEL_GAIN} has {2 * int(MAX_LEVEL_GAIN) + 1} values; the "
        "outcome head and its label no longer describe the same thing"
    )


class PPOBuffer(Protocol):
    """The slice of the rollout-buffer contract the learner touches."""

    value: np.ndarray  # f32[C], the behaviour critic's estimate
    ret: np.ndarray  # f32[C], filled by ``compute_gae``

    def as_torch(
        self, idx: np.ndarray, device: torch.device | str
    ) -> tuple[ObsBatch, Tensor, dict[str, Tensor]]:
        """Rows ``idx`` as (observations, privileged[B,5,54], scalars).

        The learner reads six of the scalars: ``action`` and ``logp``
        for the ratio, ``advantage`` and ``ret`` for the two objectives,
        and ``reward_level`` with ``done`` for the outcome head's label.
        A stand-in that omits any of them raises ``KeyError`` in
        ``losses`` rather than quietly dropping a term.
        """
        ...


@dataclass
class PPOConfig:
    """Hyperparameters. Defaults are the ones the loop runs with.

    ``target_kl`` is off by default. When set, a minibatch whose approx
    KL exceeds it is *discarded* -- no backward, no step -- and the
    update stops there, so the policy returned has never knowingly left
    the trust region. Stopping after taking the offending step, which is
    the other common implementation, leaves the thing it was meant to
    prevent in the weights.

    ``weight_decay`` is 0, not AdamW's 0.01. A decay the caller did not
    ask for is a regulariser on a policy network that is already
    constrained by the clip, and it is the sort of default that shows up
    months later as an unexplained entropy collapse.

    ``check_actions`` verifies that every stored action indexes a legal
    candidate slot. An illegal index gathers ``MASK_LOGIT``, which makes
    the ratio underflow to zero and the update a silent no-op for that
    row; catching it costs one boolean reduction (and, on CUDA, one
    device sync) per minibatch.

    ``own_value_coef`` matches ``vf_coef``: the actor's own value head
    is regressing the same returns against the same squared error, so
    weighting the two differently would only say that one of the two
    value estimates matters less, which is not a claim anyone here is
    making. ``outcome_coef`` matches ``belief_coef`` at 0.1 -- both are
    representation-shaping auxiliaries rather than objectives, and the
    outcome head sees a fraction of the rows the others do (see
    ``losses``), so a larger weight would let a handful of terminal rows
    per minibatch shout over the policy gradient.

    Setting either coefficient to 0 leaves its head computed and
    untrained, which is the state this class was in before the two
    terms existed; it is a legitimate ablation and nothing else.
    """

    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    belief_coef: float = 0.1
    own_value_coef: float = 0.5
    outcome_coef: float = 0.1
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    epochs: int = 4
    minibatch: int = 256
    normalise_adv: bool = True
    target_kl: float | None = None
    bf16: bool = True

    weight_decay: float = 0.0
    adam_eps: float = 1e-8
    betas: tuple[float, float] = (0.9, 0.999)
    check_actions: bool = True

    # ``compile`` applies torch.compile to the training graph. It is
    # off because it was measured, and it loses badly.
    #
    # In isolation it wins: a fixed batch of 256 replayed twenty times
    # on an A10 runs 82.7 ms eager against 62.9 ms compiled under bf16.
    # In the training loop the same flag took the update span from
    # 17.9 s to 171.0 s, a 9.6x regression, and the step from 28 s to
    # 183 s. The difference is that the benchmark replays one batch
    # while the loop draws a fresh minibatch each iteration and the
    # last of every epoch is ragged -- 24 rows against 256 -- so the
    # backend keeps recompiling for work the benchmark never does.
    #
    # Making it pay would mean padding every minibatch to a fixed
    # width and passing dynamic=False, for a gain the isolated figures
    # cap at about 1.3x. Left off until that is worth doing.
    #
    # Separately, the inference path cannot be compiled at all today:
    # under ``no_grad`` inductor propagates a symbolic size into the
    # ``torch.arange`` in ``_length_mask`` and fails with "Expected a
    # number but got Identity". dynamic=False does not avoid it.
    compile: bool = False


# --- loss pieces ----------------------------------------------------------


def masked_entropy(log_probs: Tensor, mask: Tensor) -> Tensor:
    """Entropy [B] of the candidate distribution over legal slots only.

    ``log_probs`` is [B, N] and ``mask`` is bool[B, N], True where the
    slot holds a real candidate. The distribution is renormalised over
    the legal slots before the entropy is taken, so the answer is the
    entropy of a genuine probability distribution even if the caller
    hands over log-probs that put mass on padding. For the network's own
    output the renormalisation is a no-op -- invalid slots hold
    ``MASK_LOGIT`` and come out at probability exactly zero -- and the
    point of doing it anyway is that the function then cannot be made
    wrong by a change elsewhere that stops the masking being exact.

    Everything is written with ``torch.where`` rather than a
    multiply-by-mask so that no padded term is ever formed: ``0 * -1e9``
    is a harmless -0.0, but ``0 * -inf`` is a NaN, and the difference
    between those two is one refactor of ``MASK_LOGIT``.
    """
    zero = torch.zeros((), dtype=log_probs.dtype, device=log_probs.device)
    probs = torch.where(mask, log_probs.exp(), zero)
    total = probs.sum(dim=-1, keepdim=True).clamp_min(_EPS)
    probs = probs / total
    # log(p / Z) = log p - log Z, on the legal slots; zero elsewhere.
    renormalised = torch.where(mask, log_probs - total.log(), zero)
    return -(probs * renormalised).sum(dim=-1)


def belief_target(obs: ObsBatch, privileged: Tensor) -> tuple[Tensor, Tensor]:
    """Where the cards the acting seat cannot see actually are.

    The head's five slots are named in ``model.py``: ``BELIEF_SEAT_1/2/3``
    are the three other seats in relative order, ``BELIEF_KITTY`` is the
    burial and ``BELIEF_PLAYED`` is the table. Built from two tensors,
    because no single one carries all five::

        slot 0 SEAT_1  = privileged[:, 1]          the seat after me
        slot 1 SEAT_2  = privileged[:, 2]          my partner
        slot 2 SEAT_3  = privileged[:, 3]          the seat before me
        slot 3 KITTY   = privileged[:, 4]
        slot 4 PLAYED  = obs.planes[:, PLANE_PLAYED:PLANE_KITTY].sum(1)

    then normalised per card type::

        total[b, c]     = sum_l counts[b, c, l]
        target[b, c, l] = counts[b, c, l] / total[b, c]  if total > 0
                          0                              if total = 0
        weight[b, c]    = 1.0 if total > 0 else 0.0

    Privileged row 0 -- the acting seat's own hand -- is *dropped*. It is
    known, not believed: it sits in the observation as
    ``planes[PLANE_HAND]``, so including it would give the head a free
    answer for the card types it can already read off its own input, and
    a card type the seat holds every copy of now scores weight 0 and
    leaves the loss entirely, which is correct.

    The counts are clamped at 1.0 rather than at an epsilon because they
    are integers: a nonzero total is at least 1, so the clamp is exact
    where it matters and only guards a division whose result is
    discarded.

    **The played slice spans planes 1..8, not 1..4.** ``PLANE_PLAYED``
    covers *completed* tricks only; the trick in progress lives in the
    four ``PLANE_TRICK`` planes. The two regions are contiguous, so
    ``PLANE_PLAYED:PLANE_KITTY`` is both of them, and it is the same
    span ``features._planes_from_plays`` writes them into.

    Taking only the completed four does not merely lose mass, it
    mislabels, because the loss renormalises whatever is left. A card
    type with one copy in the open trick and one in the partner's hand
    would come out as "certainly with the partner" when the truth is an
    even split. The error is invisible where a type's every unseen copy
    is already PLAYED -- dropping one and renormalising puts it back --
    and bites exactly where a type straddles the table and a hand.

    Measured over 524 real decisions from ``gen.episode.generate``:
    75% of decisions have a non-empty open trick, the completed-only
    variant violates the partition invariant on 56% of them, and about
    one card type per decision is handed an actively wrong distribution,
    with a slot off by as much as a full 2 copies. With the open trick
    included the five slots partition the unseen cards exactly on every
    decision, which is what
    ``test_belief_target_partitions_the_unseen_cards`` asserts::

        sum_l counts[b, c, l] == num_decks - own_hand[b, c]
    """
    played = obs.planes[:, PLANE_PLAYED:PLANE_KITTY].sum(dim=1, keepdim=True)
    hidden = privileged[:, 1:NUM_PRIVILEGED_ROWS]  # seats 1..3 then the kitty
    counts = torch.cat((hidden, played.to(hidden.dtype)), dim=1)  # [B, 5, 54]
    counts = counts.transpose(1, 2)  # [B, 54, 5]

    total = counts.sum(dim=-1, keepdim=True)  # [B, 54, 1]
    known = total > 0
    target = torch.where(known, counts / total.clamp_min(1.0), torch.zeros_like(counts))
    return target, known.squeeze(-1).to(counts.dtype)


def belief_cross_entropy(
    belief_log_probs: Tensor, target: Tensor, weight: Tensor
) -> Tensor:
    """Weighted mean cross-entropy over [B, 54] card types.

    Averaged over the card types that carry weight, not over all 54, so
    the loss does not shrink simply because a late-hand position has
    fewer cards left to place.
    """
    per_card = -(target * belief_log_probs).sum(dim=-1)  # [B, 54]
    return (per_card * weight).sum() / weight.sum().clamp_min(1.0)


def outcome_class(reward_level: Tensor) -> Tensor:
    """Team-signed level reward -> an outcome class index in [0, 7).

    ``rollout`` stores the hand's result as
    ``(attacker_gain - defender_gain) / 3``, signed for the acting
    seat's team, so the signed level swing is the stored value times
    three and lands on an integer in [-3, +3]. Shifting by three gives
    the seven classes ``NUM_OUTCOMES`` names::

        reward_level  -1.0  -2/3  -1/3   0.0  +1/3  +2/3  +1.0
        swing           -3    -2    -1     0    +1    +2    +3
        class            0     1     2     3     4     5     6

    Seat-relative, not attacker-relative: class 6 means *my team* went
    up three, which is the bucket ``hand_result`` calls (0, 3) for an
    attacking seat and the bucket it calls (3, 0) for a defending one.
    That is the right frame for a head reading a seat's own summary,
    which is written from that seat's point of view throughout.

    Rounded rather than truncated because the stored value is a float32
    third, and clamped because a caller is free to hand this function a
    row whose label is absent -- ``losses`` weights those out, but it
    still has to pass a legal class index to ``cross_entropy``.
    """
    swing = torch.round(reward_level.float() * MAX_LEVEL_GAIN)
    shifted = swing + MAX_LEVEL_GAIN
    return shifted.to(torch.int64).clamp_(0, NUM_OUTCOMES - 1)


def explained_variance(predicted: np.ndarray, actual: np.ndarray) -> float:
    """1 - Var(actual - predicted) / Var(actual); NaN when actual is flat.

    The usual critic diagnostic. 0 means the value head is no better than
    predicting the mean return, 1 means it is exact, and negative means
    it is worse than the mean -- which is the number that says a run has
    a credit-assignment bug rather than a slow critic.
    """
    y = np.asarray(actual, dtype=np.float64).reshape(-1)
    p = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if y.size == 0:
        return float("nan")
    var_y = float(y.var())
    if var_y <= 0.0:
        return float("nan")
    return float(1.0 - ((y - p).var() / var_y))


# --- the learner ----------------------------------------------------------


def _bf16_supported(device: torch.device) -> bool:
    """Whether bf16 autocast is worth enabling on this device."""
    if device.type == "cuda":
        try:
            return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        except Exception:
            return False
    # The CPU autocast path builds bf16 everywhere torch does, and is
    # correct if not fast; leaving it on keeps the CPU smoke test
    # exercising the same code path the GPU run takes.
    return device.type == "cpu"


def _make_optimiser(
    net: torch.nn.Module, cfg: PPOConfig, device: torch.device
) -> torch.optim.Optimizer:
    """AdamW, fused when this build and device will take it.

    Probed rather than predicted: whether ``fused`` is accepted depends
    on the torch version, the device type and the parameter dtypes, and
    every one of those has moved between releases.
    """
    params = [p for p in net.parameters() if p.requires_grad]
    kwargs: dict[str, Any] = {
        "lr": cfg.lr,
        "betas": tuple(cfg.betas),
        "eps": cfg.adam_eps,
        "weight_decay": cfg.weight_decay,
    }
    try:
        return torch.optim.AdamW(params, fused=True, **kwargs)
    except (RuntimeError, ValueError, TypeError):
        return torch.optim.AdamW(params, **kwargs)


_LOSS_KEYS = (
    "policy_loss",
    "value_loss",
    "entropy",
    "belief_loss",
    "own_value_loss",
    "outcome_loss_done",
    "total_loss",
    "approx_kl",
    "clip_fraction",
)


class PPOLearner:
    """Runs ``cfg.epochs`` passes of clipped PPO over a filled buffer.

    ::

        learner = PPOLearner(net, PPOConfig(), device, run=run)
        compute_gae(buffer, gamma=1.0, lam=0.95)
        metrics = learner.update(buffer)

    The optimiser is exposed as ``opt`` because the checkpointer saves
    and restores it by that name.
    """

    def __init__(
        self,
        net: ActorCritic,
        cfg: PPOConfig,
        device: torch.device | str,
        run: Any = None,
    ) -> None:
        self.net = net
        self.cfg = cfg
        self.device = torch.device(device)
        self.run = run
        self.opt = _make_optimiser(net, cfg, self.device)
        self.amp = bool(cfg.bf16) and _bf16_supported(self.device)
        # Compiled graph for the update; ``self.net`` stays the
        # eager module so that parameters, state_dict and the
        # rollout all address the same object.
        self.graph = net
        if cfg.compile and self.device.type == "cuda":
            try:
                self.graph = torch.compile(net)
            except Exception as exc:  # pragma: no cover
                warnings.warn(
                    f"ppo: torch.compile unavailable "
                    f"({type(exc).__name__}: {exc}); running eager",
                    stacklevel=2,
                )
        # Optimiser steps taken, across every call to ``update``. Used as
        # the x-axis for the telemetry rows this learner logs.
        self.steps = 0
        self.updates = 0

    # -- one minibatch -----------------------------------------------------

    def losses(
        self, obs: ObsBatch, privileged: Tensor, batch: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Total loss and its parts for one minibatch.

        Returned with the autograd graph intact, so a test can take the
        gradient of a single component; ``update`` detaches when it reads
        the numbers out.

        Six terms: the clipped surrogate and the entropy bonus for the
        policy, the privileged critic's squared error for the baseline,
        and three auxiliaries on the actor's own summary -- belief,
        own value, outcome. Every parameter of ``ActorCritic`` is
        downstream of at least one of them, which is a property
        ``test_every_parameter_gets_a_gradient`` asserts rather than a
        property anyone should trust by inspection.

        **The outcome head is trained on ``done`` rows only**, which is
        why its metric is ``outcome_loss_done`` and not
        ``outcome_loss``. The label is the hand's level swing and
        ``reward_level`` carries it on the row flagged ``done`` and
        nowhere else; every other row of the trajectory holds 0.0, which
        is not a missing value but a real class -- the one meaning "the
        hand ended level". Feeding those rows in would teach the head
        that nineteen decisions in twenty end in a draw, which is false
        and is a worse error than not training on them at all.

        Broadcasting a hand's outcome back over every row of its own
        ``(episode, seat)`` group is the target one would rather have.
        It is not available here: a minibatch is a shuffled sample of
        rows and holds none of the rest of any group, so producing it
        would mean a grouping pass over the whole buffer before each
        update and four more fields (``episode``, ``seat``, ``done``,
        ``reward_level`` in full) on the ``PPOBuffer`` contract. What
        the restriction costs is the rows where the prediction is
        hardest -- mid-hand, where the head would shape the encoder
        most -- and what it buys is that the head is never trained
        against a label that is wrong. That is the right way round, and
        the broadcast is the obvious thing to do if this head is ever
        promoted from auxiliary to objective.
        """
        cfg = self.cfg
        action = batch["action"].to(self.device, torch.int64).reshape(-1, 1)
        old_logp = batch["logp"].to(self.device, torch.float32).reshape(-1)
        advantage = batch["advantage"].to(self.device, torch.float32).reshape(-1)
        returns = batch["ret"].to(self.device, torch.float32).reshape(-1)
        level = batch["reward_level"].to(self.device, torch.float32).reshape(-1)
        done = batch["done"].to(self.device, torch.bool).reshape(-1)

        if cfg.check_actions:
            legal = obs.candidate_mask.gather(1, action)
            if not bool(legal.all()):
                bad = int((~legal).sum())
                raise ValueError(
                    f"{bad} stored action(s) index an illegal candidate slot; "
                    "the rollout and the learner disagree about the move list"
                )

        with torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16, enabled=self.amp
        ):
            out, value = self.graph(obs, privileged, detach_critic=False)

        # Out of autocast from here down: the reductions are float32.
        log_probs = out.log_probs.float()
        value = value.float()

        new_logp = log_probs.gather(1, action).squeeze(1)
        log_ratio = new_logp - old_logp
        # Bounded before the exp; see ``_MAX_LOG_RATIO`` for why the
        # bound cannot touch a legitimate row. The count is reported so
        # that a run where it stops being zero says so on the dashboard
        # instead of only in the gradient norm.
        ratio_clamped = (log_ratio.abs() > _MAX_LOG_RATIO).sum()
        log_ratio = log_ratio.clamp(-_MAX_LOG_RATIO, _MAX_LOG_RATIO)
        ratio = log_ratio.exp()

        if cfg.normalise_adv and advantage.numel() > 1:
            # Per minibatch, which is what makes the scale of the policy
            # gradient independent of how large the returns happen to be
            # this round. Skipped for a single row, where the unbiased
            # standard deviation is NaN rather than small.
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        surrogate = torch.min(
            ratio * advantage,
            ratio.clamp(1.0 - cfg.clip, 1.0 + cfg.clip) * advantage,
        )
        policy_loss = -surrogate.mean()
        value_loss = F.mse_loss(value, returns)
        entropy = masked_entropy(log_probs, obs.candidate_mask).mean()

        target, weight = belief_target(obs, privileged.float())
        belief_loss = belief_cross_entropy(out.belief_log_probs.float(), target, weight)

        # The actor's own value head, on the critic's target. A separate
        # term rather than a second call to ``value_loss`` because the
        # two predictions come from different information sets and only
        # one of them is the baseline the advantages were built with.
        own_value_loss = F.mse_loss(out.value.float(), returns)

        # Unlabelled rows are weighted out rather than indexed out: an
        # empty index would make the mean NaN, while a zero weight
        # leaves the term at exactly 0.0 and -- the part that matters --
        # leaves the head's parameters in the graph, so a minibatch that
        # happens to hold no terminal row still yields a gradient of
        # zeros rather than no gradient at all.
        outcome_weight = done.to(torch.float32)
        outcome_rows = outcome_weight.sum()
        per_row = F.cross_entropy(
            out.outcome_logits.float(), outcome_class(level), reduction="none"
        )
        outcome_loss = (per_row * outcome_weight).sum() / outcome_rows.clamp_min(1.0)

        total = (
            policy_loss
            + cfg.vf_coef * value_loss
            - cfg.ent_coef * entropy
            + cfg.belief_coef * belief_loss
            + cfg.own_value_coef * own_value_loss
            + cfg.outcome_coef * outcome_loss
        )

        with torch.no_grad():
            # Schulman's k3 estimator: unbiased, non-negative, and far
            # less noisy than the mean log-ratio, which is what makes it
            # usable as an early-stopping threshold.
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            clip_fraction = ((ratio - 1.0).abs() > cfg.clip).float().mean()

        parts = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "belief_loss": belief_loss,
            "own_value_loss": own_value_loss,
            "outcome_loss_done": outcome_loss,
            "total_loss": total,
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
            "ratio": ratio,
            "outcome_rows": outcome_rows,
            "ratio_clamped": ratio_clamped,
        }
        return total, parts

    # -- one update --------------------------------------------------------

    def update(self, buffer: PPOBuffer) -> dict[str, float]:
        """``cfg.epochs`` shuffled passes over the buffer. -> metrics.

        Minibatch means are weighted by minibatch size, so a short tail
        batch does not count as much as a full one.
        """
        cfg = self.cfg
        n = num_rows(buffer)
        if n == 0:
            raise ValueError("update() called on a buffer with no rows")

        # Against the *behaviour* critic's predictions, which is the
        # question worth asking: how well did the value function the
        # rollout ran with explain the returns it went on to earn.
        ev = explained_variance(np.asarray(buffer.value)[:n], np.asarray(buffer.ret)[:n])

        size = max(1, int(cfg.minibatch))
        totals = {key: 0.0 for key in _LOSS_KEYS}
        totals["grad_norm"] = 0.0
        rows_seen = 0
        nonfinite = 0
        n_minibatches = 0
        epochs_run = 0
        early_stop = False
        # Row counts rather than row means, and only over the minibatches
        # that actually stepped. Both answer "how many", so averaging
        # them per row would say nothing.
        outcome_rows = 0
        ratio_clamped = 0

        self.net.train()
        for epoch in range(max(0, int(cfg.epochs))):
            order = torch.randperm(n).numpy()
            for start in range(0, n, size):
                idx = order[start : start + size]
                obs, privileged, batch = buffer.as_torch(idx, self.device)
                loss, parts = self.losses(obs, privileged, batch)

                if cfg.target_kl is not None:
                    if float(parts["approx_kl"]) > float(cfg.target_kl):
                        # Discarded entirely: this minibatch contributes
                        # neither a step nor a metric.
                        early_stop = True
                        break

                # A non-finite loss is refused before it can reach
                # the weights. `clip_grad_norm_` is not a guard:
                # with a NaN total norm the clip coefficient is
                # NaN, every gradient is multiplied by it, and
                # `step` writes NaN into the whole network. One bad
                # minibatch would end a run silently, hours before
                # anyone read the metrics.
                if not bool(torch.isfinite(loss)):
                    nonfinite += 1
                    continue

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), cfg.max_grad_norm
                )
                if not bool(torch.isfinite(grad_norm)):
                    # The loss was finite and the gradient is not,
                    # which an overflow in the backward can do.
                    self.opt.zero_grad(set_to_none=True)
                    nonfinite += 1
                    continue

                self.opt.step()
                self.steps += 1

                k = int(idx.shape[0])
                rows_seen += k
                n_minibatches += 1
                for key in _LOSS_KEYS:
                    totals[key] += float(parts[key].detach()) * k
                totals["grad_norm"] += float(grad_norm) * k
                outcome_rows += int(parts["outcome_rows"])
                ratio_clamped += int(parts["ratio_clamped"])

            epochs_run = epoch + 1
            if early_stop:
                break

        self.updates += 1
        denominator = float(max(rows_seen, 1))
        metrics = {key: totals[key] / denominator for key in _LOSS_KEYS}
        metrics["grad_norm"] = totals["grad_norm"] / denominator
        metrics["explained_variance"] = ev
        metrics["lr"] = float(self.opt.param_groups[0]["lr"])
        metrics["n_minibatches"] = float(n_minibatches)
        metrics["epochs_run"] = float(epochs_run)
        metrics["early_stop"] = 1.0 if early_stop else 0.0
        metrics["rows"] = float(n)
        metrics["nonfinite_minibatches"] = float(nonfinite)
        # Counts, so they stay counts: zero clamped rows and zero
        # terminal rows are both true statements about an update that
        # took no step, where a zero *loss* would not be.
        # ``outcome_rows`` is what tells ``outcome_loss_done == 0.0``
        # apart from a head that saw no label and one that got the
        # answer right.
        metrics["outcome_rows"] = float(outcome_rows)
        metrics["ratio_clamped"] = float(ratio_clamped)
        # An update that took no step reports its losses as NaN
        # rather than as zero. Zero is a value the losses can
        # legitimately take, and an entropy of exactly 0.0 on a
        # dashboard reads as collapse rather than as 'not
        # measured'. ``approx_kl`` is the sharpest case: the one
        # thing an update stopped by ``target_kl`` certainly was is
        # *above* the threshold, so reporting 0.0 would invert it.
        if rows_seen == 0:
            for key in _LOSS_KEYS:
                metrics[key] = float("nan")
            metrics["grad_norm"] = float("nan")

        if self.run is not None:
            self.run.log({f"ppo/{k}": v for k, v in metrics.items()}, step=self.steps)
        return metrics
