"""Storage for the decisions one round of self-play produces.

The actor writes about ninety rows per hand and the learner reads them
back in shuffled minibatches, so the store is allocated once at a fixed
capacity and appended into. A list of per-decision dictionaries would
spend more time in the allocator than the network spends on the forward
pass it feeds, and would fragment the arrays the learner wants to slice.

Three choices here are load-bearing.

*The observation fields are float16.* One decision's observation is
27.9 KiB, of which the history block is 9.3 KiB and the padded candidate
block 13.5 KiB. Halving the float width halves the round: 100,000
decisions costs 2.85 GB rather than 5.67. Nothing stored that way loses
anything a network would notice -- every value is either a small exact
integer (card counts, one-hot flags, play summaries) or a feature
already normalised into a range where float16's three-and-a-half figures
are far finer than the quantity itself.

*The reduced fields are float32.* Log-probability, value, reward,
advantage and return are summed, discounted and differenced along a
trajectory. float16 there is not noise, it is bias: a return accumulated
over eighty plays at a 5e-4 relative step loses the early tricks
outright.

*The layout is checked against the model, not copied from it.* The
extents and the ``ObsBatch`` field order are asserted at import, so a
change to ``nn/features.py`` or ``nn/model.py`` stops this module at
load time rather than producing a batch the network silently misreads --
``ObsBatch`` is a NamedTuple, and a positionally built one with two
fields transposed typechecks perfectly.

The rollout casts each observation to float16 *before* the forward pass
it samples from, so the log-probability recorded next to a row is the
log-probability of exactly the tensor stored in that row. See
``rollout.collect``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ..nn.features import (
    CONTEXT_DIM,
    D_CARD,
    HISTORY_DIM,
    MAX_HISTORY,
    NUM_PLANES,
)
from ..nn.model import NUM_PRIVILEGED_ROWS, ModelConfig, ObsBatch
from ..rules.cards import NUM_TYPES

__all__ = [
    "OBS_FIELDS",
    "SCALAR_FIELDS",
    "RolloutBuffer",
    "obs_batch_from_arrays",
]

# The observation half of a row, in the order ``ObsBatch`` declares it.
OBS_FIELDS = (
    "history",
    "history_len",
    "context",
    "unseen",
    "planes",
    "cards",
    "candidates",
    "candidate_mask",
)

# Everything the learner reduces over. ``advantage`` and ``ret`` are
# written by the learner rather than the actor and start at zero.
#
# The reward arrives in two terms that are kept apart as well as summed.
# Training starts on the dense point signal alone and anneals towards
# the level outcome, so the two have to stay separately addressable:
# ``reward`` is what the learner reads, and re-weighting it afterwards
# from a pre-summed column is impossible.
SCALAR_FIELDS = (
    "action",
    "logp",
    "value",
    "reward",
    "reward_points",
    "reward_level",
    "done",
    "seat",
    "episode",
    "advantage",
    "ret",
)

# The float scalars ``add`` is handed, in the order it checks them. Named
# so that the check and the assignment cannot drift apart, and so that
# ``load_npz`` can quote the same list back.
_ADDED_SCALARS = ("logp", "value", "reward", "reward_points", "reward_level")

# Rows per block in ``_validate``. The widest field is 128 x 54 float16,
# so a block of this size costs a 28 MB boolean whatever the buffer's
# capacity, and the check stays a constant-memory pass.
_VALIDATE_ROWS = 4096


def _check_layout() -> None:
    """Fail at import if the feature or model layout has moved."""
    if ObsBatch._fields != OBS_FIELDS:
        raise RuntimeError(
            f"ObsBatch fields {ObsBatch._fields} no longer match the buffer's "
            f"{OBS_FIELDS}; the batch would be built in the wrong order"
        )
    expected = {
        "MAX_HISTORY": (MAX_HISTORY, 128),
        "HISTORY_DIM": (HISTORY_DIM, 37),
        "CONTEXT_DIM": (CONTEXT_DIM, 37),
        "NUM_PLANES": (NUM_PLANES, 10),
        "D_CARD": (D_CARD, 30),
        "NUM_TYPES": (NUM_TYPES, 54),
        "NUM_PRIVILEGED_ROWS": (NUM_PRIVILEGED_ROWS, 5),
    }
    for name, (got, want) in expected.items():
        if got != want:
            raise RuntimeError(
                f"{name} is {got}, not {want}; the buffer's field shapes and "
                f"anything already saved to disk are stale"
            )


_check_layout()


# The two ObsBatch fields that are not float32: everything else is
# promoted after the move to the device.
_OBS_NON_FLOAT = frozenset({"history_len", "candidate_mask"})


def obs_batch_from_arrays(
    history: np.ndarray,
    history_len: np.ndarray,
    context: np.ndarray,
    unseen: np.ndarray,
    planes: np.ndarray,
    cards: np.ndarray,
    candidates: np.ndarray,
    candidate_mask: np.ndarray,
    device: torch.device | str | None = None,
) -> ObsBatch:
    """Batched numpy arrays -> the ``ObsBatch`` the network takes.

    The one place that knows the float16-to-float32 promotion and the
    field order, so the rollout's freshly packed batch and the buffer's
    stored rows cannot drift apart. Arguments are positional in
    ``ObsBatch`` order on purpose: a caller that passes them by keyword
    is protected by the names, one that passes them positionally is
    protected by this function being the only place it happens.

    The promotion happens after the move rather than before it. Widening
    float16 to float32 is exact, so the values are the same either way,
    but done on the host it is the single most expensive thing here --
    about 10 ms for a batch of 256 -- and it doubles what crosses the
    bus. On the device it is a bandwidth-bound kernel over data that is
    already there.
    """
    batch = ObsBatch(
        history=torch.from_numpy(np.ascontiguousarray(history)),
        history_len=torch.from_numpy(
            np.ascontiguousarray(history_len, dtype=np.int64)
        ),
        context=torch.from_numpy(np.ascontiguousarray(context)),
        unseen=torch.from_numpy(np.ascontiguousarray(unseen)),
        planes=torch.from_numpy(np.ascontiguousarray(planes)),
        cards=torch.from_numpy(np.ascontiguousarray(cards)),
        candidates=torch.from_numpy(np.ascontiguousarray(candidates)),
        candidate_mask=torch.from_numpy(
            np.ascontiguousarray(candidate_mask, dtype=bool)
        ),
    )
    if device is not None:
        batch = batch.to(device)
    # `float()` returns the tensor unchanged when it is already float32,
    # so a caller that handed in float32 pays nothing for this.
    return ObsBatch(
        *(
            tensor if name in _OBS_NON_FLOAT else tensor.float()
            for name, tensor in zip(ObsBatch._fields, batch)
        )
    )


class RolloutBuffer:
    """A fixed-capacity, append-only store of self-play decisions.

    ``capacity`` rows are allocated up front and ``add`` fills them in
    order; ``add`` raises rather than overwriting, because a rollout that
    silently wrapped would leave a hand's early tricks credited and its
    late ones missing, and nothing downstream could tell.

    Per row, with the default ``ModelConfig``::

        history        float16[128, 37]   9,472 B
        history_len    int64                  8
        context        float16[37]           74
        unseen         float16[54]          108
        planes         float16[10, 54]    1,080
        cards          float16[54, 30]    3,240
        candidates     float16[128, 54]  13,824
        candidate_mask bool[128]            128
        privileged     float16[5, 54]       540
        action         int64                  8
        logp           float32                4
        value          float32                4
        reward         float32                4
        reward_points  float32                4
        reward_level   float32                4
        done           bool                   1
        seat           int64                  8
        episode        int64                  8
        advantage      float32                4
        ret            float32                4
                                         -------
                                          28,527 B

    ``action`` is a slot index into the candidate axis, not a card or a
    move id: the candidate list is rebuilt per decision and only the slot
    is meaningful next to the stored ``candidates`` and
    ``candidate_mask``.

    **``candidates[action]`` is the play the policy asked for, not
    always the play that happened.** A lead goes through
    ``PlayState._resolve_throw``, which applies the block check: a throw
    some other single hand can beat is cut down to its smallest beatable
    component and the rest returns to the thrower's hand, so the cards
    that reach the table are a subset of the ones the slot names. It is
    the same substitution whichever slot was chosen, and the row's
    observation, log-probability and reward all describe the requested
    play, so the importance ratio is consistent and the substitution is
    simply part of the environment's dynamics -- a policy that keeps
    proposing blocked throws is charged for the outcome it gets. It is
    still not what a reader assumes, and it is not rare: over 300 hands
    at initialisation, 2,137 of 26,536 stored rows -- 8.1% of all rows
    and 32.2% of the rows where the seat led -- were substituted. Any
    analysis that reconstructs the played cards from ``candidates`` and
    ``action`` is wrong on one row in twelve; ``PlayState.step`` returns
    the cards it actually played, and that return value is the only
    record of them.

    ``reward`` is the combined signal the learner reads.
    ``reward_points`` and ``reward_level`` are its two terms kept apart,
    so a run that anneals from the dense point reward to the hand
    outcome can re-weight them; ``rollout.collect`` maintains
    ``reward == reward_points + level_weight * reward_level`` exactly.

    ``episode`` groups the rows of one hand. Rows of different hands
    interleave, because the rollout keeps hundreds of hands in flight, so
    grouping is by this field and never by contiguity.
    """

    def __init__(self, capacity: int, cfg: ModelConfig | None = None) -> None:
        cfg = cfg or ModelConfig()
        capacity = int(capacity)
        if capacity < 1:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if cfg.max_history != MAX_HISTORY:
            # ``features.encode`` always emits MAX_HISTORY rows, so a
            # config that disagrees would store a truncated or a
            # zero-padded history without either side noticing.
            raise ValueError(
                f"cfg.max_history {cfg.max_history} must equal the encoder's "
                f"{MAX_HISTORY}"
            )
        if cfg.max_candidates < 1:
            raise ValueError("cfg.max_candidates must be positive")

        self.cfg = cfg
        self.capacity = capacity
        self._n = 0

        c, h, k = capacity, cfg.max_history, cfg.max_candidates
        self.history = np.zeros((c, h, HISTORY_DIM), dtype=np.float16)
        self.history_len = np.zeros(c, dtype=np.int64)
        self.context = np.zeros((c, CONTEXT_DIM), dtype=np.float16)
        self.unseen = np.zeros((c, NUM_TYPES), dtype=np.float16)
        self.planes = np.zeros((c, NUM_PLANES, NUM_TYPES), dtype=np.float16)
        self.cards = np.zeros((c, NUM_TYPES, D_CARD), dtype=np.float16)
        self.candidates = np.zeros((c, k, NUM_TYPES), dtype=np.float16)
        self.candidate_mask = np.zeros((c, k), dtype=bool)
        self.privileged = np.zeros(
            (c, NUM_PRIVILEGED_ROWS, NUM_TYPES), dtype=np.float16
        )
        self.action = np.zeros(c, dtype=np.int64)
        self.logp = np.zeros(c, dtype=np.float32)
        self.value = np.zeros(c, dtype=np.float32)
        self.reward = np.zeros(c, dtype=np.float32)
        self.reward_points = np.zeros(c, dtype=np.float32)
        self.reward_level = np.zeros(c, dtype=np.float32)
        self.done = np.zeros(c, dtype=bool)
        self.seat = np.zeros(c, dtype=np.int64)
        self.episode = np.zeros(c, dtype=np.int64)
        self.advantage = np.zeros(c, dtype=np.float32)
        self.ret = np.zeros(c, dtype=np.float32)

    # --- introspection ----------------------------------------------------

    @property
    def fields(self) -> tuple[str, ...]:
        """Every stored field, in the order they are saved and loaded."""
        return OBS_FIELDS + ("privileged",) + SCALAR_FIELDS

    def array(self, name: str) -> np.ndarray:
        """The whole backing array for one field, capacity included."""
        if name not in self.fields:
            raise KeyError(f"{name!r} is not a buffer field")
        return getattr(self, name)

    def __len__(self) -> int:
        return self._n

    @property
    def full(self) -> bool:
        return self._n >= self.capacity

    def reset(self) -> None:
        """Empty the buffer.

        The arrays are zeroed rather than only the counter. A stale tail
        is invisible until something reads past the length -- a save, a
        learner indexing with a mask built before the reset -- and then
        it is data from a previous policy wearing the current one's
        label.
        """
        for name in self.fields:
            getattr(self, name).fill(0)
        self._n = 0

    # --- writing ----------------------------------------------------------

    def add(
        self,
        *,
        history: np.ndarray,
        history_len: int,
        context: np.ndarray,
        unseen: np.ndarray,
        planes: np.ndarray,
        cards: np.ndarray,
        candidates: np.ndarray,
        candidate_mask: np.ndarray,
        privileged: np.ndarray,
        action: int,
        logp: float,
        value: float,
        seat: int,
        episode: int,
        reward: float = 0.0,
        reward_points: float = 0.0,
        reward_level: float = 0.0,
        done: bool = False,
    ) -> int:
        """Append one decision and return the row index it landed in.

        Keyword-only, because the caller is passing nine arrays of four
        different shapes and a positional mix-up between, say, ``unseen``
        and a plane row would be caught by nothing here and by nothing in
        the network either.

        The row index is the point of the return value: the trick that
        this decision belongs to has not closed yet, so its reward is
        written later through ``buffer.reward[row] += ...``.

        Two things are refused outright: an ``action`` that does not
        index a legal candidate slot, and a non-finite float in any of
        the reduced fields. Both are cheap to catch here and impossible
        to attribute to a row once the buffer has been reduced over.
        """
        if self._n >= self.capacity:
            raise RuntimeError(
                f"rollout buffer is full at {self.capacity} rows; "
                f"size it for the hands being collected"
            )
        i = self._n

        shapes = {
            "history": (history, self.history.shape[1:]),
            "context": (context, self.context.shape[1:]),
            "unseen": (unseen, self.unseen.shape[1:]),
            "planes": (planes, self.planes.shape[1:]),
            "cards": (cards, self.cards.shape[1:]),
            "candidates": (candidates, self.candidates.shape[1:]),
            "candidate_mask": (candidate_mask, self.candidate_mask.shape[1:]),
            "privileged": (privileged, self.privileged.shape[1:]),
        }
        for name, (value_in, want) in shapes.items():
            got = np.shape(value_in)
            if got != want:
                raise ValueError(f"{name} has shape {got}, expected {want}")

        length = int(history_len)
        if not 0 <= length <= self.history.shape[1]:
            raise ValueError(
                f"history_len {length} outside 0..{self.history.shape[1]}"
            )
        slot = int(action)
        if not 0 <= slot < self.candidates.shape[1]:
            raise ValueError(
                f"action {slot} outside 0..{self.candidates.shape[1] - 1}"
            )
        if not bool(np.asarray(candidate_mask)[slot]):
            # An illegal action in the buffer is a poisoned gradient that
            # looks like ordinary data, so it is refused at the door.
            raise ValueError(f"action {slot} is not a legal candidate slot")
        for name, number in zip(
            _ADDED_SCALARS, (logp, value, reward, reward_points, reward_level)
        ):
            # A NaN here is not caught anywhere downstream. It survives
            # the GAE recursion -- which propagates it backwards over
            # every earlier row of the same trajectory -- and reaches
            # the optimiser, where one row's NaN advantage becomes a
            # NaN gradient and then a dead network. The learner's own
            # non-finite guard fires a whole minibatch too late to say
            # which row did it; here the field has a name.
            if not np.isfinite(float(number)):
                raise ValueError(
                    f"{name} is {number!r}; a non-finite reduced field "
                    "poisons the GAE recursion and then the optimiser"
                )

        self.history[i] = history
        self.history_len[i] = length
        self.context[i] = context
        self.unseen[i] = unseen
        self.planes[i] = planes
        self.cards[i] = cards
        self.candidates[i] = candidates
        self.candidate_mask[i] = candidate_mask
        self.privileged[i] = privileged
        self.action[i] = slot
        self.logp[i] = logp
        self.value[i] = value
        self.reward[i] = reward
        self.reward_points[i] = reward_points
        self.reward_level[i] = reward_level
        self.done[i] = bool(done)
        self.seat[i] = int(seat)
        self.episode[i] = int(episode)
        self.advantage[i] = 0.0
        self.ret[i] = 0.0

        self._n = i + 1
        return i

    # --- reading ----------------------------------------------------------

    def rows(self, idx: Any) -> np.ndarray:
        """Normalise a selection into an int64 array of filled rows."""
        if isinstance(idx, slice):
            picked = np.arange(*idx.indices(self._n), dtype=np.int64)
        elif isinstance(idx, (int, np.integer)):
            picked = np.array([int(idx)], dtype=np.int64)
        else:
            picked = np.asarray(idx)
            if picked.dtype == bool:
                if picked.shape != (self._n,) and picked.shape != (self.capacity,):
                    raise IndexError(
                        f"boolean selection of {picked.shape} does not match "
                        f"{self._n} filled rows"
                    )
                picked = np.nonzero(picked)[0]
            picked = picked.astype(np.int64, copy=False).reshape(-1)
        if picked.size == 0:
            raise ValueError("empty selection")
        if picked.min() < 0 or picked.max() >= self._n:
            raise IndexError(
                f"rows {picked.min()}..{picked.max()} outside the "
                f"{self._n} filled rows"
            )
        return picked

    def as_torch(
        self, idx: Any, device: torch.device | str | None = None
    ) -> tuple[ObsBatch, Tensor, dict[str, Tensor]]:
        """``(ObsBatch, privileged[B, 5, 54], scalars)`` for a selection.

        Every float16 field is promoted to float32 on the way out, which
        is what the network's Linear weights expect and what keeps the
        promotion in one place rather than at each call site.

        ``idx`` may be an index array, a boolean mask over the filled
        rows, a slice or a single integer. Reading past the filled length
        is an error rather than zeros: an unwritten row is a legal-looking
        observation with no legal candidate at all.
        """
        picked = self.rows(idx)
        obs = obs_batch_from_arrays(
            self.history[picked],
            self.history_len[picked],
            self.context[picked],
            self.unseen[picked],
            self.planes[picked],
            self.cards[picked],
            self.candidates[picked],
            self.candidate_mask[picked],
            device=device,
        )
        privileged = torch.from_numpy(
            self.privileged[picked].astype(np.float32)
        )
        scalars = {
            "action": torch.from_numpy(self.action[picked]),
            "logp": torch.from_numpy(self.logp[picked]),
            "value": torch.from_numpy(self.value[picked]),
            "reward": torch.from_numpy(self.reward[picked]),
            "reward_points": torch.from_numpy(self.reward_points[picked]),
            "reward_level": torch.from_numpy(self.reward_level[picked]),
            "done": torch.from_numpy(self.done[picked]),
            "seat": torch.from_numpy(self.seat[picked]),
            "episode": torch.from_numpy(self.episode[picked]),
            "advantage": torch.from_numpy(self.advantage[picked]),
            "ret": torch.from_numpy(self.ret[picked]),
        }
        if device is not None:
            privileged = privileged.to(device)
            scalars = {k: v.to(device) for k, v in scalars.items()}
        return obs, privileged, scalars

    # --- persistence ------------------------------------------------------

    def save_npz(self, path: str | Path) -> Path:
        """Write the filled rows to ``path``, returning where they went.

        Uncompressed: a round is gigabytes of float16 that deflate badly
        and slowly, and the file exists so a crashed learner can pick the
        round back up rather than to be archived.
        """
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(path.suffix + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {name: getattr(self, name)[: self._n] for name in self.fields}
        payload["capacity"] = np.int64(self.capacity)
        payload["n_rows"] = np.int64(self._n)
        payload["max_history"] = np.int64(self.cfg.max_history)
        payload["max_candidates"] = np.int64(self.cfg.max_candidates)
        np.savez(path, **payload)
        return path

    def _validate(self, n: int, source: object) -> None:
        """Re-run ``add``'s legality checks over rows that bypassed it.

        Everything written through ``add`` is checked a row at a time.
        Rows that arrive any other way -- ``load_npz`` is the only such
        path today -- have had none of it, and ``load_npz`` is used
        precisely when nobody is watching: a learner restarting after a
        crash, on a file some earlier process may have been in the
        middle of writing. A truncated write, a bit flip, or a buffer
        saved by a build whose reward code had a bug all produce a file
        that passes every dtype and shape check and then trains on
        nonsense.

        Two checks, the same two ``add`` makes. Every stored action must
        index a True slot of its own ``candidate_mask``: an action
        pointing at padding gathers ``MASK_LOGIT``, its ratio underflows
        to zero, and that row silently contributes nothing. And every
        floating-point field must be finite, the observation blocks
        included -- a NaN in ``history`` propagates through the encoder
        to every output of the forward pass, not just to its own row's.

        Vectorised, so the cost is one pass over the loaded arrays,
        roughly what reading them off disk already cost -- but in row
        blocks rather than whole fields, because ``np.isfinite`` over
        all of ``candidates`` at 100,000 rows would want a 691 MB
        boolean, and asking for that on a recovery path is a poor way
        to greet a process that has already crashed once. The message
        names the field and the first offending row, because a file this
        large is not something anyone is going to bisect by hand.
        """
        if n == 0:
            return

        action = self.action[:n]
        low, high = int(action.min()), int(action.max())
        slots = self.candidate_mask.shape[1]
        if low < 0 or high >= slots:
            row = int(np.argmax((action < 0) | (action >= slots)))
            raise ValueError(
                f"{source}: row {row} stores action {int(action[row])}, "
                f"outside 0..{slots - 1}"
            )
        legal = self.candidate_mask[np.arange(n), action]
        if not legal.all():
            row = int(np.argmax(~legal))
            raise ValueError(
                f"{source}: row {row} stores action {int(action[row])}, "
                f"which is not a legal candidate slot "
                f"({int(self.candidate_mask[row].sum())} slots are legal)"
            )

        for name in self.fields:
            stored = getattr(self, name)
            if not np.issubdtype(stored.dtype, np.floating):
                continue
            for start in range(0, n, _VALIDATE_ROWS):
                block = stored[start : min(start + _VALIDATE_ROWS, n)]
                finite = np.isfinite(block.reshape(block.shape[0], -1)).all(axis=1)
                if not finite.all():
                    row = start + int(np.argmax(~finite))
                    raise ValueError(
                        f"{source}: row {row} of {name} holds a non-finite "
                        f"value; it would reach the optimiser unchecked"
                    )

    @classmethod
    def load_npz(
        cls, path: str | Path, cfg: ModelConfig | None = None
    ) -> "RolloutBuffer":
        """Rebuild a buffer saved by ``save_npz``.

        The capacity, the length and every field come back exactly as
        they were; the unfilled tail is zeros, which is what it was in
        the buffer that was saved, since ``reset`` and ``__init__`` both
        zero it.

        The loaded rows are then put through ``_validate``, which is the
        pair of checks ``add`` makes per row -- dtype and shape agreeing
        with the schema says nothing about whether the contents are
        trainable.
        """
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            cfg = cfg or ModelConfig()
            saved_h = int(data["max_history"])
            saved_k = int(data["max_candidates"])
            if (saved_h, saved_k) != (cfg.max_history, cfg.max_candidates):
                raise ValueError(
                    f"{path} was written with max_history {saved_h} and "
                    f"max_candidates {saved_k}; this config says "
                    f"{cfg.max_history} and {cfg.max_candidates}"
                )
            buffer = cls(int(data["capacity"]), cfg)
            n = int(data["n_rows"])
            if n > buffer.capacity:
                raise ValueError(
                    f"{path} holds {n} rows, more than its own capacity "
                    f"{buffer.capacity}"
                )
            for name in buffer.fields:
                stored = getattr(buffer, name)
                incoming = data[name]
                if incoming.dtype != stored.dtype:
                    raise ValueError(
                        f"{name} was saved as {incoming.dtype}, expected "
                        f"{stored.dtype}"
                    )
                if incoming.shape != (n,) + stored.shape[1:]:
                    raise ValueError(
                        f"{name} was saved with shape {incoming.shape}, "
                        f"expected {(n,) + stored.shape[1:]}"
                    )
                stored[:n] = incoming
            buffer._n = n
        buffer._validate(n, path)
        return buffer
