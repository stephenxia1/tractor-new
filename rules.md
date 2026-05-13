# Tractor — Canonical Rules (v1)

This document defines **the exact variant of Tractor (升级 / 拖拉机 / Sheng Ji) the engine will implement**. Tractor has many regional and house variants; this file picks one and is explicit about it. Variant alternatives are listed inline with `**[VARIANT]**` so they can be parameterized later.

> The variant chosen here is "4-player, 2-deck, 升级 with progression from 2 → A, with throws (甩牌), with kitty multiplier on the final trick." It matches the most common North-American Chinese-community house rules.

---

## 1. Terminology

| Term | Chinese | Meaning |
|---|---|---|
| Level / Rank | 级 | The current rank a team is "on" (2, 3, ..., A). Determines which card rank is elevated to trump. |
| Trump rank | — | The rank equal to the defending team's current level. All four cards of this rank in both decks are trump. |
| Trump suit | 主花色 | The suit declared as trump for the hand. |
| Landlord | 庄家 | The specific player who holds the persistent "house" role: they start the draw and receive the kitty. May be **set** or **unset** across hands (see §4.1). |
| Defending team | 庄家方 / 防守方 | The landlord's team. Holds the kitty; tries to prevent attackers from scoring 80+. |
| Attacking team | 闲家方 / 进攻方 | The other team; collects point cards to reach 80+. |
| Trump declarer | — | The player who made the **final trump declaration** during dealing in this hand. This sets the trump suit. May be on either team and is **not** necessarily the landlord (see §5). |
| Kitty / Bottom | 底牌 / 底 | The 8 face-down cards set aside at deal time; awarded to the winner of the last trick. |
| Point card | 分牌 | 5 (5 pts), 10 (10 pts), K (10 pts). 200 points total in the deck. |
| Trick | 一墩 | One round of play: lead + each other player's follow. |
| Throw | 甩牌 | A multi-component lead (e.g., a pair + a single) that succeeds only if unbeatable in every component. |
| Tractor | 拖拉机 | Two or more consecutive pairs in the same effective suit. Named after the game. |

---

## 2. Components

- **Deck**: Two standard 54-card decks (108 cards total). Each deck includes one small joker (小王) and one big joker (大王). All identical cards across the two decks are interchangeable; pairs are formed from two copies of the same physical card (same suit and rank).
- **Players**: Exactly 4, seated `North, East, South, West`. Partnerships are fixed: **N+S** vs **E+W**. Partners sit across.
- **Starting state**:
  - Both teams begin at level **2**.
  - The landlord starts **unset**. It will be set during the first hand by whoever first declares trump (see §4.1, §5).

---

## 3. Card Ordering

Card ordering depends on the trump suit and trump rank in effect for the current hand. Let `R` be the trump rank and `S` be the trump suit.

### 3.1 Within a non-trump suit
Ranks from low to high, with rank `R` removed (it's a trump):
`3, 4, 5, ..., (skip R), ..., J, Q, K, A`

For example, if `R = 5`: `3, 4, 6, 7, 8, 9, 10, J, Q, K, A`.

### 3.2 Within the trump category
From **low to high**:

1. Trump-suit ranks in normal order, with `R` removed:
   `3♥, 4♥, ..., (skip R♥), ..., J♥, Q♥, K♥, A♥` (assuming `S = ♥`)
2. **Off-suit trump-rank cards** (rank `R` in the three non-trump suits). These are all treated as **a single tier** for tractor and ordering purposes. Any off-suit `R` of any suit is higher than `A` of the trump suit and lower than the trump-suit `R`. They are equal in rank among themselves; ties are broken by order of play (earlier wins).
3. **Trump-suit trump-rank**: `R♥` (in our example).
4. **Small joker** (小王).
5. **Big joker** (大王).

### 3.3 Special declarations (no-trump)
If trump is declared via jokers ("无主" / "joker trump"), there is no trump suit. The trump category contains only:
- Off-suit `R` cards (all four suits — there is no "trump suit" tier; all are equal in this tier).
- Small jokers.
- Big jokers.

All four standard suits behave as non-trump in this mode.

---

## 4. Game Setup (per hand)

A "hand" or "deal" (一副) proceeds through the phases below. A full "game" is a sequence of hands ending when one team advances past rank A (see §10).

### 4.1 The landlord lifecycle

The **landlord** (庄家) is a persistent role that carries across hands. The landlord's team is the **defending team**.

**Setting the landlord:**
- If the landlord is currently **unset** at the start of a hand, the first player who successfully declares trump during dealing (§5) becomes the new landlord. The trump rank `R` for this hand is then determined retroactively from the new landlord's team's level.
- If the landlord is **set**, they remain landlord for this hand. Anyone may still declare trump to set the suit (§5), but the landlord receives the kitty regardless.

**Persistence and rotation:**
- After a hand, if the **defending team won** (attackers scored < 80), the landlord role **rotates to the landlord's partner** for the next hand. The defending team's level is incremented per §8.3.
- If the **attacking team won** (attackers scored ≥ 80), the landlord is **unset**. The attacking team's level is incremented per §8.3. The next hand begins with no landlord; whoever declares first becomes the new landlord.
- If no one declares and no landlord is set (special edge case — see §5.4), a random player becomes landlord via the kitty-flip fallback.

**Trump rank `R` for a hand:**
- = the level of the landlord's team. When the landlord is unset entering a hand, `R` is determined the moment a landlord is set (during the declaration window).

### 4.2 Starting the draw
- The 108-card stack is shuffled into a single deck.
- If a landlord is set: the landlord "starts the draw" — they conceptually initiate dealing and receive the first card. The remaining cards are distributed one at a time, clockwise.
- If no landlord is set: the server picks an arbitrary seat to receive the first card. (In a digital implementation, since cards are randomly shuffled, the starting seat is largely cosmetic.)
- The **last 8 cards** of the stack are reserved as the **kitty** and are not dealt to any player. After dealing, each player holds 25 cards.

> **Note on "dealing"**: In physical Tractor, a designated dealer shuffles and distributes cards. In this digital implementation, the engine performs both. "Starting the draw" refers only to seat order (who is dealt to first) and to the visual cue for which player initiates the round.

---

## 5. Trump Declaration (亮主)

Trump is declared **during dealing** in this variant, using cards that match the current trump rank `R`.

### 5.1 Declaration types, from weakest to strongest

| # | Declaration | What it shows | Trump becomes |
|---|---|---|---|
| 1 | Single | One card of rank `R` in some suit `X` | Suit `X` is trump |
| 2 | Pair | Both copies of rank `R` in suit `X` | Suit `X` is trump |
| 3 | Small-joker pair | Both small jokers | No-trump (joker trump) |
| 4 | Big-joker pair | Both big jokers | No-trump (joker trump) |

Strictly: `(4) > (3) > (2) > (1)`. A higher-tier declaration **overrides** any prior lower-tier declaration. Within the same tier, the **later** declaration overrides the earlier **only** if it is in a different suit AND the declarer has not yet played a card from this hand — i.e., re-declarations in the same tier are not allowed; you must escalate.

### 5.2 Reverse declaration (反主)
A player may **override their own partner's declaration** by escalating tiers. A player **may not override their own prior declaration in the same tier** (e.g., if you played a single ♥`R`, you cannot later play a single ♠`R`; you would need to escalate to a pair).

### 5.3 Timing window
Declarations may be made from the moment a player receives a matching card, and remain open **until the landlord picks up the kitty**. The landlord picks up the kitty whenever they choose, subject to the constraint below.

- **Minimum lockout**: The kitty may NOT be picked up until **at least 3 seconds after the last card has been dealt**. This guarantees other players a final window to override the current declaration after dealing ends.
- **Maximum**: There is no hard upper bound — the landlord may wait as long as they like before claiming the kitty (e.g., to see if a higher-tier declaration emerges that they would prefer to make themselves). The engine should expose a soft timer for UX (e.g., auto-pickup after 30s of inactivity), but this is a UX concern, not a rules concern.
- **Closing event**: Once the landlord picks up the kitty, declarations are locked. The currently-standing trump suit is final.
- **Landlord-not-yet-set case**: If the landlord is unset entering the hand, the player whose declaration is **standing at the moment of kitty pickup** becomes the new landlord. Provisional declarations that get overridden by a higher-tier declaration before pickup do not confer landlord status. In effect: the landlord is the "winner of the declaration contest," not the first to declare.

### 5.4 No declaration
If the 3-second lockout ends and no declaration has been made:
- The **top card of the kitty is flipped face-up**. Its suit becomes the trump suit. Trump rank `R` remains the defending team's level.
- **Landlord assignment**:
  - If the landlord is already set: the landlord remains. The flip merely sets the trump suit.
  - If the landlord is unset: a **random player** is chosen as the new landlord. `R` becomes that player's team's current level.
- Declarations are immediately closed at this point.
- The flipped card is shuffled back into the kitty before pickup.

### 5.5 Who declares vs. who is the landlord
Declarations set the **trump suit**. The **landlord** (who gets the kitty and whose team is the defending team) is a separate concept governed by §4.1.

- Any player at the table may declare on any hand. Declarations may come from members of either team.
- If a landlord is already set: the landlord receives the kitty regardless of who declared. The declaration only sets the trump suit. (Strategically, this means an attacker can declare to choose a trump suit favorable to their team, even though the landlord will keep the kitty.)
- If the landlord is unset: the player whose declaration stands at kitty-pickup becomes the new landlord (see §4.1 and §5.3).

---

## 6. Kitty Exchange (扣底)

Immediately after declaration closes:

1. The 8 kitty cards are dealt face-down to the **landlord only**. The landlord's hand temporarily holds 33 cards.
2. The landlord chooses **any 8 cards** from their hand and places them face-down as the new kitty. These need not be the same cards they picked up. **[VARIANT: some tables forbid placing trumps in the kitty; we allow anything.]**
3. The kitty is set aside until the final trick.
4. Each player now holds exactly 25 cards.

---

## 7. Trick Play

### 7.1 Trick order

- The **landlord leads the first trick** of the hand.
- The winner of each trick leads the next.
- A hand consists of exactly **25 tricks**.

### 7.2 Lead types

The lead may be one of:

- **Single**: one card.
- **Pair**: two copies of the same physical card (same rank, same suit, both decks).
- **Tractor**: `k ≥ 2` pairs of **consecutive ranks** in the **same effective suit**, where "effective suit" means: all cards are non-trump of one suit, OR all are trump (per §3.2).
  - **Non-trump tractors**: pairs are consecutive in the normal rank order with rank `R` skipped (since `R` is trump). E.g., if `R = 5` and the suit is ♣: pairs of `4♣` and `6♣` are consecutive; pairs of `7♣` and `8♣` are consecutive; pairs of `4♣` and `5♣` are not (and `5♣` is trump anyway).
  - **Trump tractors**: the trump category from §3.2 partitions into tiers:
    1. Natural trump-suit ranks `3, 4, ..., A` (skipping `R`) — multiple ranks, each consecutive in normal order.
    2. Off-suit-`R` tier — three cards (rank `R` in each non-trump suit), all at the same effective rank level.
    3. Trump-suit-`R` — single rank.
    4. Small joker — single rank.
    5. Big joker — single rank.

    **Adjacency rule**: every tier boundary (1↔2, 2↔3, 3↔4, 4↔5) is a single consecutive step. Within tier 1, consecutivity is the natural rank order. Within tier 2, there is **no internal consecutivity** — two different off-suit-`R` pairs (e.g., `5♠5♠` and `5♦5♦`) are at the same effective rank and are NOT consecutive. Tiers 3, 4, 5 each contain only one rank. A consequence: **at most one off-suit-`R` pair can participate in any single tractor.**
  - **Concrete examples** (assume ♥ trump, `R = 5`):
    - `(Q♥Q♥, K♥K♥, A♥A♥)` — length-3 tractor (all tier 1).
    - `(A♥A♥, 5♠5♠)` — length-2 tractor (bridges tier 1↔2).
    - `(K♥K♥, A♥A♥, 5♣5♣)` — length-3 tractor (bridges tier 1↔2).
    - `(5♠5♠, 5♦5♦)` — **NOT** a tractor (both tier 2, no internal consecutivity).
    - `(A♥A♥, 5♠5♠, 5♦5♦)` — **NOT** a length-3 tractor. `(A♥A♥, 5♠5♠)` alone is a length-2 tractor; `5♦5♦` cannot extend it.
    - `(5♦5♦, 5♥5♥)` — length-2 tractor (bridges tier 2↔3).
    - `(5♥5♥, small-joker-pair)` — length-2 tractor (bridges tier 3↔4).
    - `(small-joker-pair, big-joker-pair)` — length-2 tractor (bridges tier 4↔5).
    - `(K♥K♥, A♥A♥, 5♣5♣, 5♥5♥, small-joker-pair, big-joker-pair)` — length-6 tractor (bridges all four tier boundaries).
  - **[VARIANT: stricter tables disallow some or all tier-boundary bridging; see `tractorTierBridging` in §13.]**
- **Throw (甩牌)**: a set of cards **all of the same effective suit**, consisting of multiple "components" (singles, pairs, tractors). The leader claims that **every component is unbeatable** by any single opponent given the cards remaining in play. See §7.6 for resolution.

A lead has an **effective suit**: the suit of all its cards (trump counts as a single suit for this purpose).

### 7.3 Follow rules (non-throw lead)

When the lead is a single, pair, or tractor of effective suit `L`, each subsequent player must play exactly the same **number** of cards as the lead. The cards they play are constrained as follows.

Let the player's hand contain `c` cards of suit `L`. Let the lead's structure be the multiset of pair-and-tractor requirements.

**Order of obligation (most specific first):**

1. **Match the structure in suit `L` first.**
   - If the lead is a **tractor of length k**, the player must play any tractor of length `k` in `L` from their hand if they have one. If they don't have a length-`k` tractor, they must play the longest available tractor (any length ≥ 2) of `L` if it can substitute for part of the lead.
   - If the lead is a **pair**, the player must play any pair in `L` if they have one.
2. **After matching available structure, fill remaining cards with any cards of suit `L`** from their hand, until either the count is met or no cards of `L` remain.
3. **If the player runs out of suit `L`**, the remaining cards may be any cards (from any other suit, including trump).

**Trumping** is allowed in step 3 only — i.e., a player can only trump when they are void in the led suit.

**Specifically, by lead structure:**

- **Lead is a single of `L`**:
  - If `c ≥ 1`: must play one card of `L` (any choice).
  - If `c = 0`: may play any one card.

- **Lead is a pair of `L`**:
  - If the player has any pair in `L`: must play one such pair (player's choice if multiple).
  - Else if `c ≥ 2`: must play any 2 cards of `L`.
  - Else if `c = 1`: must play that one card + any other card.
  - Else (`c = 0`): may play any 2 cards.

- **Lead is a tractor of length `k` in `L` (so 2k cards)**:
  - If the player has a tractor of length `k` in `L`: must play one such tractor.
  - Else if the player has tractors of length `< k` in `L`: must play the longest available, then fill the remaining slots greedily by remaining pairs in `L`, then remaining singles in `L`. **[VARIANT: precise greedy-fill rules vary; we use "longest tractor first, then pairs, then singles, all from `L` exhaustively before leaving `L`".]**
  - Else if the player has any pairs in `L`: must play all available pairs in `L`, plus singles in `L` to fill the count.
  - Else if `c ≥ 2k`: must play any `2k` cards of `L`.
  - Else if `0 < c < 2k`: must play all cards of `L`, then fill the rest with any cards.
  - Else (`c = 0`): may play any `2k` cards.

The principle: a player cannot **withhold** matching structure from `L`. They may choose between equivalent matchings (e.g., which pair to play if they have two pairs in `L`), but they may not avoid following with structure they hold.

### 7.4 Winning a trick (non-throw lead)

After all four players have played, the winner is determined as follows:

1. Identify all plays that **legally match the lead's structure** (i.e., contain the same combo structure as the lead, in either the led suit or trump).
   - A play "matches the structure" if it contains a combo of the same shape (single → single, pair → pair, tractor of length `k` → tractor of length `k`) using cards from a **single effective suit** — either the led suit, or trump.
2. Among matching plays:
   - **Trump combos beat non-trump combos.**
   - Within the same effective suit, the **higher** combo (by §3 ordering of its highest card) wins. Ties broken by **earlier play order** (lead first, then in clockwise order).
3. If no follower played a matching combo (everyone "broke up" the structure or trumped without a full match), the lead wins.

A combo "beats" the lead only by matching its full structure in one suit. **You cannot beat a pair with two singles, even two trump singles. You cannot beat a tractor of length 2 with two non-consecutive trump pairs.**

### 7.5 Trick collection

All cards played in the trick go into the **winning team's** captured pile. Point cards captured by the **attacking team** count toward their score. Point cards captured by the defending team do not directly count but may matter for the kitty multiplier (§9).

### 7.6 Throws (甩牌)

A throw is a multi-component lead all in one effective suit. It resolves as follows.

1. The leader places the throw face-up, declaring the components (e.g., "pair of 9♥, pair of 10♥, single A♥").
2. **Before others play**, the engine checks whether the throw is "safe":
   - For each component `c` of the throw, check whether **any single opponent** (i.e., either of the two players on the opposing team) holds cards that can beat `c` according to §7.4 — i.e., a higher combo of the same structure in the led suit, or any trump combo of the same structure.
   - If **every component is safe** (no opponent can beat any one component), the throw succeeds.
3. **If the throw succeeds**:
   - All four players play to the throw in turn. Each follower must match the throw's full structure per §7.3, treating the throw as a compound lead (each component must be matched if possible in the led suit before substituting from other suits).
   - The leader wins the trick.
4. **If the throw fails** (any component is beatable):
   - The throw is **rejected**. The engine identifies the **smallest component the opponent can beat** (lowest single, or lowest pair, etc.) **[VARIANT: some tables penalize with a fixed 10-point fine; we use the "forced lead" penalty.]**.
   - The leader is forced to lead **only that smallest component** (the rest of the throw returns to hand). Play proceeds normally with that lead.
   - This is a significant penalty: opponents now know the leader holds the rest of the throw.

The "no single opponent can beat" rule means the test is per-opponent. Teammates' cards are irrelevant (they can't break a throw).

---

## 8. Scoring (per hand)

### 8.1 Point cards

Only the attacking team's captured points count toward their hand score. Point values:

| Rank | Points | Copies in deck |
|---|---|---|
| 5 | 5 | 8 |
| 10 | 10 | 8 |
| K | 10 | 8 |

Total points in the deck: `8×5 + 8×10 + 8×10 = 200`.

### 8.2 Kitty multiplier (底分翻倍)

After the **25th trick**:

- If the **defenders won the last trick**: the kitty is captured by the defenders. No bonus to attackers.
- If the **attackers won the last trick**: the kitty is revealed and its point-card values are summed (call this `K`). The attackers receive `K × m` added to their score, where `m = 2 × n` and `n` is the **total number of cards in the winning combo of the last trick**.

| Winning combo | Cards `n` | Multiplier `m = 2n` |
|---|---|---|
| Single | 1 | 2 |
| Pair | 2 | 4 |
| Tractor of length 2 | 4 | 8 |
| Tractor of length `k` | `2k` | `4k` |
| Throw (compound) | sum of all component card counts | `2 × (sum)` |

The throw rule applies whether the throw is built from singles, pairs, tractors, or any mix; only the total card count matters. **[VARIANT: many tables use `2^n` instead; we use `2n` for predictability.]**

### 8.3 Determining the hand result

Let `P` be the attackers' final point total (captures + kitty bonus, if applicable).

| `P` range | Result | Effect on landlord |
|---|---|---|
| `0` | Defenders +3 levels | Landlord rotates to their partner |
| `1–39` | Defenders +2 levels | Landlord rotates to their partner |
| `40–79` | Defenders +1 level | Landlord rotates to their partner |
| `80–119` | No level change | **Landlord is unset** |
| `120–159` | Attackers +1 level | **Landlord is unset** |
| `160–199` | Attackers +2 levels | **Landlord is unset** |
| `200` | Attackers +3 levels | **Landlord is unset** |

Notes:
- Only one team's level changes per hand — never both.
- "Landlord rotates to partner" means: the role passes to the other player on the same team. This is the traditional Tractor behavior after a successful defense.
- "Landlord is unset" means: the next hand begins with no landlord. Whoever first declares trump (or, in the no-declaration fallback, a random player) becomes the new landlord (§4.1, §5.4).

---

## 9. Level Progression

- Both teams maintain independent levels in `{2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A}`.
- "Levels gained" advance the team's level by that many steps. E.g., a team at `10` gaining 2 levels moves to `Q`.
- The trump rank `R` for a hand is **the landlord's team's current level** at the start of that hand. If the landlord is unset entering a hand, `R` is set retroactively from the new landlord's team's level once they are determined.

### 9.1 Defending at A (打A)
- A team must "defend at A": after reaching level A, they must win at least one more hand **as the defending team** (i.e., with the landlord on their team) to win the game.
- If a team reaches level A but is then beaten as defenders without winning a hand at A, they remain at A until they successfully defend (the landlord role having been transferred away and back).
- A team that gains levels which would push them past A simply caps at A and must defend.

### 9.2 Game end
The game ends when one team successfully **wins a hand as the defending team while at level A**. That team wins the game.

---

## 10. State Machine Summary

Each hand cycles through these phases (the engine should explicitly model each):

1. `SETUP` — read carried-over state: team levels, landlord (set or unset). Trump rank `R` is determined from the landlord's team's level when known; otherwise deferred.
2. `DEAL` — cards distributed; declarations may occur asynchronously during this phase. If the landlord is unset, the first standing declaration provisionally assigns the landlord (re-assignable until kitty pickup).
3. `DECLARE_LOCKOUT` — 3-second window after the last card is dealt. Declarations remain open; kitty pickup is forbidden. If no declaration stands when this window ends and the landlord is unset, transition to `KITTY_FLIP`.
4. `DECLARE_OPEN` — (entered if a declaration is standing OR the landlord is already set) Declarations remain open. The landlord may pick up the kitty at any time, transitioning to `KITTY_PICKUP`. Other players may still escalate the trump suit (but the landlord remains, if set).
5. `KITTY_FLIP` — (only when no declaration exists AND landlord is unset) Top kitty card is flipped; its suit becomes trump; a random player becomes the landlord; declarations close.
6. `KITTY_PICKUP` — landlord receives the 8 kitty cards. Declarations are now locked. Landlord identity is locked. `R` is finalized.
7. `KITTY_REPLACE` — landlord chooses 8 cards to set aside as the new kitty.
8. `PLAY` — 25 tricks, each consisting of `LEAD` + 3 × `FOLLOW`, then `RESOLVE`.
9. `SCORE` — compute attackers' total, apply kitty multiplier, determine result.
10. `LEVEL_UP` — update team levels per §8.3. Update landlord state: rotate to partner (defenders won) or unset (attackers won). Return to `SETUP`.

Throws within `PLAY` introduce a substate: `THROW_PROPOSED` → `THROW_CHECK` → either `THROW_FORCED_LEAD` (failed) or `THROW_RESOLVED` (succeeded, proceeds as compound lead).

---

## 11. Engine-Visible Edge Cases

These are the cases the engine MUST get right (and the test suite must cover):

1. **Pair-vs-pair tie in trump**: earlier play wins. Verify with two off-suit `R` pairs of different suits.
2. **Off-suit `R` ordering**: off-suit `5♠` and off-suit `5♣` (when hearts is trump, level 5) are equal in rank. First played wins.
3. **Tractor spanning trump tiers**: per §7.2, all tier boundaries are consecutive, but the off-suit-`R` tier has no internal consecutivity. So `(K♥K♥, A♥A♥, 5♠5♠)` IS a length-3 tractor (when ♥ trump, `R=5`), but `(A♥A♥, 5♠5♠, 5♦5♦)` is NOT. At most one off-suit-`R` pair may appear in any single tractor.
4. **Pair-following with a single available**: if the lead is a pair and the player has only one card of `L`, they must play that one + any other card. They CANNOT instead play a pair from another suit.
5. **Throw failure where the "smallest beatable component" is itself a tractor**: if the entire throw is a pair + a tractor and only the tractor is beatable, the forced lead is the tractor (not just a pair from within it). **[VARIANT: clarify in implementation; we force the smallest **beatable** component, not the absolute smallest.]**
6. **Multiple players able to beat a throw component**: the throw still fails. The check is "does any single opponent have a beat for at least one component".
7. **Landlord plays all 25 cards but never wins a trick**: legal. Kitty multiplier hinges on last trick only.
8. **Attackers score exactly 80**: defenders lose at the lowest tier, no level change, landlord is unset. Edge of the 80-threshold.
9. **No-trump (joker) declaration**: only jokers and rank-`R` cards are trump. All four standard suits behave as non-trump. Engine must treat all rank-`R` cards equally with no trump-suit tier.
10. **Trump-rank promotion mid-game**: if defenders win at level 10 and advance to J, the **next** hand's trump rank is J. Existing card values don't change; only the next hand's setup uses the new rank.
11. **Attacker declares trump, defenders keep the kitty**: e.g., a member of the attacking team plays a single rank-`R` to set trump. The landlord still receives the kitty. Engine must track declarer and landlord as separate fields, not assume `declarer.team == landlord.team`.
12. **Landlord unset entering hand 2 (and beyond)**: after the attackers win, the landlord is unset. The next hand begins with no landlord; `R` is undefined until a landlord is set. Engine must allow `landlord = null` as a valid state through phases `SETUP` → `DEAL` → `DECLARE_LOCKOUT`.
13. **Landlord rotation across hand boundaries**: on a successful defense, the landlord role passes to the partner. This is a deterministic update at scoring time, not subject to player choice.

---

## 12. Out of Scope for v1 (but design-affecting)

The following are not implemented in v1 but the **engine architecture must accommodate them** without rewriting the core. They are not just "switches to flip later" — they change fundamental structure, so the engine should be written with abstractions that allow them:

### 12.1 Variants that change the partnership model
- **Finding Friends (找朋友 / 找主)**: the declarer names one or more "friend cards" (e.g., "the first 10♥ played"). Whoever holds and plays that card becomes the declarer's partner. Partnerships are **dynamic and hidden** until those cards are played. This means:
  - The "team identity" of a player is not constant during a hand.
  - The state visible to each player must reflect what *they personally* know about partnerships, not a global truth.
  - Scoring still divides between declaring side and attackers, but team membership resolves over time.
  - **Engine implication**: don't hard-code `team = { N+S, E+W }`. Model team membership as a per-hand assignment that may be partially hidden and may change at specific events.

### 12.2 Variants that change the table size / deck count
- **Variable player count**: 3, 5, 6, 8 players. Partnerships scale: 2v3 with one "lone hand," 3v3, 4v4, etc.
- **Variable deck count**: 1 deck (for fewer players), 3 decks (for 6 players), 4 decks (for 8 players). With more decks, "pairs" become "n-of-a-kind" and tractors become more complex (e.g., triple-tractors).
- **Engine implication**: parameterize `numPlayers`, `numDecks`, `kittySize`, and the combo definitions. The follow-rule logic must generalize from "pair = 2 of same card" to "n-of-a-kind = n copies." Tractors must generalize from "consecutive pairs" to "consecutive n-of-a-kind groups."

### 12.3 Other deferred variants
- **Bidding for kitty** instead of declaring trump.
- **"Heaven and earth" (天地)** kitty rules where specific kitty contents grant bonuses.
- **Out-of-turn / post-deal declarations** (some tables permit a brief post-deal declaration round).
- **Anti-cheating rules** around premature plays and table talk.
- **Alternative scoring thresholds** (e.g., the +N level brackets tuned differently).
- **Trump-suit-rank → joker tractor bridging** and other tier-spanning tractor rules (see §7.2).

The engine should expose these as configuration; v1 ships with the defaults documented above.

---

## 13. Variant Switches the Engine Should Expose

For configurability later. The v1 defaults are noted; the engine should read these from a config rather than hard-coding them.

```ts
type TractorVariant = {
  // Table size & deck — engine must generalize, even though v1 only ships 4/2.
  numPlayers: number;                                          // v1 default: 4
  numDecks: number;                                            // v1 default: 2
  kittySize: number;                                           // v1 default: 8

  // Partnership model — most invasive future variant.
  partnershipModel: 'fixed' | 'finding-friends';               // v1 default: 'fixed'

  // Dealing.
  dealDirection: 'left' | 'right';                             // v1 default: 'right'

  // Declarations.
  declarationLockoutMs: number;                                // v1 default: 3000
  noDeclarationFallback: 'redeal' | 'kitty-flip-random-landlord';
                                                               // v1 default: 'kitty-flip-random-landlord'
  allowDeclarationByAnyPlayer: boolean;                        // v1 default: true
  allowSameTierReDeclaration: boolean;                         // v1 default: false

  // Landlord lifecycle.
  landlordOnDefense: 'rotate-to-partner' | 'stay';             // v1 default: 'rotate-to-partner'
  landlordOnLoss: 'unset' | 'pass-to-opponent';                // v1 default: 'unset'
  declarerVsLandlord: 'separate' | 'declarer-becomes-landlord-each-hand';
                                                               // v1 default: 'separate'
                                                               // 'separate' = landlord is persistent; declarer only sets trump suit (v1).
                                                               // 'declarer-becomes-landlord-each-hand' = old model; landlord is fully per-hand.

  // Combos & follow rules.
  tractorTierBridging: boolean;                                // v1 default: true
    // true  → every trump-tier boundary (natural↔off-suit-R, off-suit-R↔trump-R,
    //         trump-R↔small-joker, small-joker↔big-joker) is consecutive.
    // false → tractors must stay within a single tier; within the off-suit-R
    //         tier, only one pair can appear in any tractor regardless.
  throwFailurePenalty: 'forced-lead' | 'fixed-point-fine';     // v1 default: 'forced-lead'

  // Scoring.
  kittyMultiplierFormula: 'linear-2n' | 'powers-of-two';       // v1 default: 'linear-2n'
  scoreThresholds: { defendersWin: number[]; attackersWin: number[] };
    // v1 default: defenders [40, 80] → +1, +2, +3; attackers [120, 160, 200] → +1, +2, +3

  // Progression.
  startingLevel: Rank;                                         // v1 default: '2'
  winCondition: 'defend-at-A' | 'reach-A-then-defend-as-defenders';
                                                               // v1 default: 'defend-at-A'
};
```

These flags are the lever for adapting to other house rules without rewriting the engine. The two most invasive future flags — `partnershipModel` and the `(numPlayers, numDecks)` pair — must be accounted for in the core data model from day one (see §12.1, §12.2).
