// How much of the legal follow set the evaluation host offers.
//
//   npx tsx src/followCoverage.ts [hands] [uniform|big]      (from research/)
//
// The host (`evalhost.ts` through `env.ts`) hands the network
// `enumerateCandidates` at 128, where training shows it the Python
// simulator's MoveSet (`scripts/follow_coverage.py`). This plays engine
// deals, and at every multi-card follow compares the host's candidates
// with every legal follow, found by brute force and judged by the
// engine's own validator: how often the offered set is incomplete, what
// share of the distinct legal follows it covers, and how often a follow
// that would take the trick as it stands is legal but not offered.
// `big` weights the random policy toward the largest plays.
import { applyEvent, classifyCombo, createGame, decomposePlay, decomposeThrow, decomposedBeats, defaultV1Config, effectiveSuit, mulberry32, nextToPlay, pickBotAction, validateFollow, validateFollowThrow, type Card } from '@tractor/engine';
import { enumerateCandidates, winningPlay } from './candidates';
import { cardSlot } from './cards';
const HANDS = Number(process.argv[2] ?? 300); const POLICY = process.argv[3] ?? 'big';
const rng = mulberry32(99); let seed = 2026;
const sig = (cs: ReadonlyArray<Card>) => cs.map(cardSlot).sort((a, b) => a - b).join(',');
const st = { follows: 0, multi: 0, incomplete: 0, tooBig: 0, legalSum: 0, offeredSum: 0, winAvail: 0, winLost: 0 };
const byK = new Map<number, { f: number; inc: number; wa: number; wl: number }>();
function* multisets(groups: Card[][], k: number): Generator<Card[]> {
  const cur: Card[] = [];
  function* rec(i: number, left: number): Generator<Card[]> {
    if (left === 0) { yield [...cur]; return; }
    if (i >= groups.length) return;
    const g = groups[i]!;
    for (let t = Math.min(g.length, left); t >= 0; t--) { for (let j = 0; j < t; j++) cur.push(g[j]!); yield* rec(i + 1, left - t); for (let j = 0; j < t; j++) cur.pop(); }
  }
  yield* rec(0, k);
}
for (let h = 0; h < HANDS; h++) {
  let s = createGame(defaultV1Config());
  s = applyEvent(s, { kind: 'start_hand', seed: seed++ }).state;
  for (let guard = 0; s.phase !== 'between_hands' && s.phase !== 'game_over' && guard < 4000; guard++) {
    if (s.phase === 'trick_review') { s = applyEvent(s, { kind: 'advance_trick' }).state; continue; }
    if (s.phase === 'declare_lockout') { s = applyEvent(s, { kind: 'release_lockout', seed: seed++ }).state; continue; }
    let acted = false;
    for (let seat = 0; seat < 4 && !acted; seat++) {
      if (s.phase === 'play' && nextToPlay(s) !== seat) continue;
      if (s.phase === 'play') {
        const hand = s.currentHand!; const trick = hand.currentTrick; const n = 2;
        const ctx = { trumpRank: hand.trumpRank!, trumpSuit: hand.trumpSuit };
        const myKitty = seat === hand.landlord ? hand.kitty : [];
        const cands = enumerateCandidates(s, seat, { maxCandidates: 128, myKitty });
        if (trick !== null && trick.plays.length > 0) {
          const lead = trick.plays[0]!.cards; const k = lead.length; st.follows++;
          if (k > 1) {
            st.multi++;
            const my = hand.hands[seat] ?? []; const ledSuit = effectiveSuit(lead[0]!, ctx);
            const cls = classifyCombo(lead, ctx, n); const thr = cls.ok ? null : decomposeThrow(lead, ctx, n);
            const ok = (c: Card[]) => cls.ok ? validateFollow(cls.combo, c, my, ctx, n).ok : (thr !== null && thr.ok ? validateFollowThrow(thr.throw, c, my, ctx, n).ok : false);
            const inLed = my.filter((c) => effectiveSuit(c, ctx) === ledSuit); const rest = my.filter((c) => effectiveSuit(c, ctx) !== ledSuit);
            const group = (cs: Card[]) => { const m = new Map<number, Card[]>(); for (const c of cs) { const g = m.get(cardSlot(c)); if (g) g.push(c); else m.set(cardSlot(c), [c]); } return [...m.values()]; };
            const req = Math.min(inLed.length, k); const legal = new Map<string, Card[]>(); let count = 0; let big = false;
            outer: for (const ledPart of multisets(group(inLed), req)) for (const fill of multisets(group(rest), k - req)) {
              if (++count > 60000) { big = true; break outer; }
              const c = [...ledPart, ...fill]; if (ok(c)) legal.set(sig(c), c);
            }
            if (big) { st.tooBig++; } else {
              const off = new Set(cands.map((c) => sig(c.cards)));
              const e = byK.get(k) ?? { f: 0, inc: 0, wa: 0, wl: 0 }; e.f++;
              st.legalSum += legal.size; st.offeredSum += [...legal.keys()].filter((x) => off.has(x)).length;
              if ([...legal.keys()].some((x) => !off.has(x))) { st.incomplete++; e.inc++; }
              const win = winningPlay(trick, ctx, n);
              const beats = (c: Card[]) => { const d = decomposePlay(c, ctx, n); return d !== null && win !== null && decomposedBeats(d, win, ctx, n); };
              const winners = [...legal.entries()].filter(([, c]) => beats(c));
              if (winners.length > 0) { st.winAvail++; e.wa++; if (!winners.some(([x]) => off.has(x))) { st.winLost++; e.wl++; } }
              byK.set(k, e);
            }
          }
        }
        if (cands.length > 0) {
          const w = cands.map((c) => POLICY === 'big' ? c.cards.length ** 2 : 1);
          let x = rng() * w.reduce((a, b) => a + b, 0); let pick = cands[cands.length - 1]!;
          for (let i = 0; i < cands.length; i++) { x -= w[i]!; if (x <= 0) { pick = cands[i]!; break; } }
          const r = applyEvent(s, { kind: 'play_cards', seat, cards: pick.cards });
          if (r.ok) { s = r.state; acted = true; continue; }
        }
      }
      const a = pickBotAction(s, seat, 'lowball'); if (a.kind === 'wait') continue;
      const r = applyEvent(s, (a.kind === 'pick_up_kitty' ? { kind: a.kind, seat } : { ...a, seat }) as never); if (r.ok) { s = r.state; acted = true; }
    }
    if (!acted) break;
  }
}
const pct = (a: number, b: number) => `${(100 * a / Math.max(b, 1)).toFixed(1)}%`;
console.log(`${HANDS} hands, policy ${POLICY}: follows ${st.follows}, multi-card ${st.multi}, exact too big ${st.tooBig}`);
console.log(`multi-card: incomplete ${st.incomplete} (${pct(st.incomplete, st.multi - st.tooBig)}), coverage ${(st.offeredSum / Math.max(1, st.legalSum)).toFixed(3)} of distinct legal follows`);
console.log(`win available ${st.winAvail}, none offered ${st.winLost} (${pct(st.winLost, st.winAvail)})`);
for (const [k, e] of [...byK.entries()].sort((a, b) => a[0] - b[0])) console.log(`  lead size ${k}: follows ${e.f}, incomplete ${e.inc}, win available ${e.wa}, win lost ${e.wl}`);
