"""Does the critic generalise, or has it memorised the hands it was fitted to?

``explained_variance`` as the learner reports it is measured on the rows
the update is about to train on, so it answers "did the behaviour critic
explain these returns" and not "will this critic explain the next hand".
Those two numbers can be far apart, and when they are, the value
function is not reducing the variance of the policy gradient -- it is
adding its own.

The instrument here is a holdout. Collect a round, keep a fraction of
the *hands* out of the fit, run the real ``PPOLearner.update`` on the
rest, and then score the fitted critic on both halves. The difference
between the two is the quantity of interest.

Collection and fitting are separate calls because the interesting
comparisons vary the fit, not the data: epochs, the discount, whether
the critic's gradient reaches the shared trunk. Scoring several fits
against one collected round makes those arms paired -- they differ in
the thing under test and not in which cards came up -- and costs one
rollout instead of one per arm.

Three things about the split are load bearing.

*It splits hands, not rows.* The critic reads a privileged view holding
all four hands and the kitty, so a seat held out of a hand whose other
three seats were fitted is not held out at all -- the cards were in the
training set, under a different seat's row. Only a whole hand is
genuinely unseen.

*A hand is one target, near enough.* With ``gamma = 1`` and a terminal
level reward, the returns of a seat's rows within a hand differ only by
the dense trick points still to come, and as the anneal moves weight on
to the level term they converge on each other. So the row count
overstates the sample size by about the number of decisions a seat
makes, and ``targets`` -- distinct ``(episode, seat)`` pairs -- is the
honest denominator. Both are reported.

*The interval is clustered by hand.* Rows inside a hand are dependent,
so a bootstrap that resamples rows would report an interval several
times too narrow. ``ev_held_lo``/``ev_held_hi`` resample hands. This is
the same error that once made a 24-hand measurement read ``+0.42`` when
the answer at 96 hands was ``-0.013``.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
import torch

from ..gen.episode import deal_sizes
from ..nn.model import ActorCritic, ModelConfig
from ..sim.state import NUM_SEATS
from .buffer import RolloutBuffer
from .gae import compute_gae
from .ppo import PPOConfig, PPOLearner, explained_variance
from .rollout import DECKS, collect

# Enough hands that the bootstrap has something to resample, small
# enough to run on a laptop. A caller measuring for real should pass
# more.
DEFAULT_HANDS = 128
DEFAULT_HOLDOUT = 0.25
BOOTSTRAP = 2000


@dataclass
class Round:
    """One collected rollout, with credit already assigned.

    Held as a unit because ``level_weight`` and ``gamma`` are properties
    of the returns stored in the buffer, not of the fit that follows,
    and passing them alongside it is how a sweep ends up labelling its
    rows with the wrong reward mix.
    """

    buffer: RolloutBuffer
    level_weight: float
    gamma: float
    lam: float
    hands: int


@dataclass(frozen=True)
class Gap:
    """One holdout measurement. Positive ``gap`` is overfitting."""

    level_weight: float
    gamma: float
    epochs: int
    detached: bool
    hands_fit: int
    hands_held: int
    rows_fit: int
    rows_held: int
    targets_fit: int  # distinct (episode, seat) pairs; the honest n
    targets_held: int
    ev_behaviour: float  # what the learner logs, for reference
    ev_fit: float
    ev_held: float
    ev_held_lo: float
    ev_held_hi: float
    gap: float  # ev_fit - ev_held
    slope_held: float  # return on prediction; < 1 is over-confident

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


def split_hands(
    episode: np.ndarray, holdout: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Row indices ``(fit, held)``, splitting whole hands.

    Separate from the measurement because this is the part that has to
    be right: a split that leaks one hand's rows across both sides makes
    the holdout score an in-sample score, and the resulting number looks
    like a healthy critic rather than a broken measurement. Returned as
    row indices rather than as a mask so the caller can hand them
    straight to ``update(rows=)``.
    """
    if not 0.0 < holdout < 1.0:
        raise ValueError(f"holdout must be in (0, 1), got {holdout}")
    hands = np.unique(episode)
    if hands.size < 2:
        raise ValueError(f"need at least 2 hands to split, got {hands.size}")
    shuffled = hands.copy()
    rng.shuffle(shuffled)
    # At least one hand on each side whatever the fraction rounds to: a
    # holdout of zero hands is not a small holdout, it is no measurement.
    n_held = min(max(1, int(round(hands.size * holdout))), hands.size - 1)
    held_hands = shuffled[:n_held]
    held_mask = np.isin(episode, held_hands)
    return (
        np.nonzero(~held_mask)[0].astype(np.int64),
        np.nonzero(held_mask)[0].astype(np.int64),
    )


def _predict(
    net: ActorCritic,
    buffer: RolloutBuffer,
    rows: np.ndarray,
    device: torch.device,
    *,
    amp: bool,
    batch: int = 256,
) -> np.ndarray:
    """The critic's value on ``rows``, under the weights it has now."""
    was_training = net.training
    net.eval()
    out = np.empty(int(rows.size), dtype=np.float64)
    try:
        with torch.no_grad():
            for start in range(0, int(rows.size), batch):
                idx = rows[start : start + batch]
                obs, privileged, _ = buffer.as_torch(idx, device)
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=amp
                ):
                    _out, value = net(obs, privileged, detach_critic=True)
                out[start : start + int(idx.size)] = (
                    value.float().reshape(-1).cpu().numpy()
                )
    finally:
        if was_training:
            net.train()
    return out


def _slope(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Least-squares slope of ``actual`` on ``predicted``.

    1.0 is a calibrated critic. Below 1 the predictions are more spread
    out than the returns they predict, which is what memorising the
    training hands looks like from outside: confident, and confident
    about the wrong thing.
    """
    var = float(predicted.var())
    if var <= 0.0:
        return float("nan")
    return float(
        ((predicted - predicted.mean()) * (actual - actual.mean())).mean() / var
    )


def _clustered_ev_interval(
    predicted: np.ndarray,
    actual: np.ndarray,
    episode: np.ndarray,
    rng: np.random.Generator,
    draws: int = BOOTSTRAP,
) -> tuple[float, float]:
    """A 95% interval for the explained variance, resampling *hands*."""
    hands = np.unique(episode)
    if hands.size < 2:
        return float("nan"), float("nan")
    # Row indices per hand, once, so a draw is a concatenate and not a
    # scan of the whole array per hand.
    by_hand = [np.nonzero(episode == h)[0] for h in hands]
    stats = np.empty(draws, dtype=np.float64)
    for d in range(draws):
        pick = rng.integers(0, len(by_hand), size=len(by_hand))
        idx = np.concatenate([by_hand[i] for i in pick])
        stats[d] = explained_variance(predicted[idx], actual[idx])
    finite = stats[np.isfinite(stats)]
    if finite.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return float(lo), float(hi)


def collect_round(
    net: ActorCritic,
    *,
    device: torch.device | str,
    level_weight: float,
    model_cfg: ModelConfig | None = None,
    n_hands: int = DEFAULT_HANDS,
    gamma: float = 1.0,
    lam: float = 0.95,
    seed: int = 0,
    pool: int | None = None,
    temperature: float = 1.0,
    amp: bool = False,
) -> Round:
    """Play ``n_hands`` and assign credit. -> a ``Round`` ready to fit."""
    device = torch.device(device)
    model_cfg = model_cfg or getattr(net, "cfg", None) or ModelConfig()

    # Room for every hand, so the collector never stops short and turns
    # a 128-hand measurement into a 90-hand one without saying so. Every
    # decision sheds at least one card, so a hand cannot exceed one
    # deal's worth of rows per seat -- the same bound ``collect`` checks
    # itself against, taken from the same function rather than guessed.
    per_seat, _kitty = deal_sizes(DECKS)
    buffer = RolloutBuffer(
        capacity=max(1024, n_hands * per_seat * NUM_SEATS), cfg=model_cfg
    )
    stats = collect(
        net,
        np.random.default_rng(seed),
        buffer,
        n_hands=n_hands,
        device=device,
        cfg=model_cfg,
        temperature=temperature,
        level_weight=level_weight,
        pool=pool,
        amp=amp,
    )
    got = int(stats.get("hands", 0))
    if got < n_hands:
        raise RuntimeError(
            f"asked for {n_hands} hands, the buffer held {got}; raise capacity"
        )
    compute_gae(buffer, gamma=gamma, lam=lam)
    return Round(
        buffer=buffer,
        level_weight=float(level_weight),
        gamma=float(gamma),
        lam=float(lam),
        hands=got,
    )


def fit_and_score(
    net: ActorCritic,
    round_: Round,
    *,
    device: torch.device | str,
    ppo_cfg: PPOConfig | None = None,
    holdout: float = DEFAULT_HOLDOUT,
    seed: int = 0,
    restore: bool = True,
) -> Gap:
    """Fit the critic on part of ``round_``; score it on the rest.

    ``net`` is mutated by the fit unless ``restore``, which puts the
    weights back afterwards so a sweep's arms cannot contaminate each
    other. Restoring is the default because the usual caller is a sweep,
    and an arm that silently started from the previous arm's weights
    would make the whole table a sequence rather than a comparison.
    """
    device = torch.device(device)
    ppo_cfg = ppo_cfg or PPOConfig()
    buffer = round_.buffer

    before = copy.deepcopy(net.state_dict()) if restore else None
    rng = np.random.default_rng(seed)

    n = len(buffer)
    episode = np.asarray(buffer.episode)[:n]
    seat = np.asarray(buffer.seat)[:n]
    returns = np.asarray(buffer.ret, dtype=np.float64)[:n]
    behaviour = np.asarray(buffer.value, dtype=np.float64)[:n]

    fit_rows, held_rows = split_hands(episode, holdout, rng)

    learner = PPOLearner(net, ppo_cfg, device)
    learner.update(buffer, rows=fit_rows)

    pred_fit = _predict(net, buffer, fit_rows, device, amp=learner.amp)
    pred_held = _predict(net, buffer, held_rows, device, amp=learner.amp)

    ev_fit = explained_variance(pred_fit, returns[fit_rows])
    ev_held = explained_variance(pred_held, returns[held_rows])
    lo, hi = _clustered_ev_interval(
        pred_held, returns[held_rows], episode[held_rows], rng
    )

    def targets(rows: np.ndarray) -> int:
        pairs = episode[rows].astype(np.int64) * NUM_SEATS + seat[rows].astype(
            np.int64
        )
        return int(np.unique(pairs).size)

    gap = Gap(
        level_weight=round_.level_weight,
        gamma=round_.gamma,
        epochs=int(ppo_cfg.epochs),
        detached=bool(ppo_cfg.detach_critic),
        hands_fit=int(np.unique(episode[fit_rows]).size),
        hands_held=int(np.unique(episode[held_rows]).size),
        rows_fit=int(fit_rows.size),
        rows_held=int(held_rows.size),
        targets_fit=targets(fit_rows),
        targets_held=targets(held_rows),
        ev_behaviour=explained_variance(behaviour, returns),
        ev_fit=float(ev_fit),
        ev_held=float(ev_held),
        ev_held_lo=lo,
        ev_held_hi=hi,
        gap=float(ev_fit - ev_held),
        slope_held=_slope(pred_held, returns[held_rows]),
    )

    if before is not None:
        net.load_state_dict(before)
    return gap


def generalisation_gap(
    net: ActorCritic,
    *,
    device: torch.device | str,
    level_weight: float,
    model_cfg: ModelConfig | None = None,
    ppo_cfg: PPOConfig | None = None,
    n_hands: int = DEFAULT_HANDS,
    holdout: float = DEFAULT_HOLDOUT,
    gamma: float = 1.0,
    lam: float = 0.95,
    seed: int = 0,
    pool: int | None = None,
    temperature: float = 1.0,
    restore: bool = True,
) -> Gap:
    """``collect_round`` then ``fit_and_score``, for a one-off reading."""
    round_ = collect_round(
        net,
        device=device,
        level_weight=level_weight,
        model_cfg=model_cfg,
        n_hands=n_hands,
        gamma=gamma,
        lam=lam,
        seed=seed,
        pool=pool,
        temperature=temperature,
    )
    return fit_and_score(
        net,
        round_,
        device=device,
        ppo_cfg=ppo_cfg,
        holdout=holdout,
        seed=seed,
        restore=restore,
    )
