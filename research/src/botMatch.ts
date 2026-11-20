// Two engine bots against each other on duplicate deals, the instrument
// evalhost.ts points at a network, pointed at two heuristics instead:
//
//   npx tsx research/src/botMatch.ts --a smart-v35 --b smart-v34 \
//     --deals 400 --seed 20261119 --pre-play random --procs 8 --out runs/v35-v34.jsonl
//
// Every deal is played twice from the same cards (GameSlot's duplicate
// mode, the same seeding evalhost uses): half 0 puts bot A on seats 0
// and 2, half 1 on seats 1 and 3, so what the cards were worth cancels in
// the pair. `--pre-play random` starts both halves from the identical
// play-start position (the training generator's declaration and burial);
// `--pre-play bot` lets each team declare and bury with its own bot.
//
// Reported from A's side, with 95% intervals clustered by deal (a deal's
// two hands are not independent): the win rate (a hand is won by taking
// or keeping the landlord title, rules.md §8.3), the level swing and the
// attacker points per deal -- the three numbers research/README.md
// reports -- plus each bot's seconds per hand and per decision, measured
// in the same processes on the same deals.
//
// With --procs N the deals are split over N child processes (deal ids
// congruent to the shard index), each appending one line per deal to its
// own file, and the parent merges them and prints the summary. A record
// file already holding some deals is resumed: those deals are skipped.
// `--summary FILE...` only summarises existing records.

import { spawn } from 'node:child_process';
import { appendFileSync, existsSync, readFileSync, rmSync } from 'node:fs';

import { computeHandResult, pickBotAction, type BotType } from '@tractor/engine';

import { cardSlot, NUM_SLOTS } from './cards';
import { GameSlot } from './env';

const argv = process.argv.slice(2);
const has = (k: string): boolean => argv.includes(`--${k}`);
const arg = (k: string, d: string): string => {
  const i = argv.indexOf(`--${k}`);
  return i >= 0 ? (argv[i + 1] ?? d) : d;
};

type HalfRecord = {
  readonly landlord: number;
  readonly attacker_points: number;
  readonly digest: number;
  // Seconds A and B spent choosing plays, and how many plays each chose.
  readonly a_s: number;
  readonly b_s: number;
  readonly a_plays: number;
  readonly b_plays: number;
};

type DealRecord = {
  readonly a: string;
  readonly b: string;
  readonly seed: number;
  readonly pre_play: string;
  readonly game: number;
  readonly halves: [HalfRecord, HalfRecord];
};

// Seats bot A holds, by half. Each half is one whole team.
const A_SEATS: ReadonlyArray<ReadonlyArray<boolean>> = [
  [true, false, true, false],
  [false, true, false, true],
];

function playDeal(
  a: BotType,
  b: BotType,
  seed: number,
  prePlay: string,
  game: number,
): DealRecord {
  const halves: HalfRecord[] = [];
  for (let half = 0; half < 2; half++) {
    const aSeats = A_SEATS[half]!;
    // A's seats are "learner" seats, so its moves surface as pending
    // decisions and are timed here; B plays inside `advance`.
    const slot = new GameSlot({
      gameId: game,
      seed: seed * 1_000_003 + game * 7919 + half,
      learnerSeats: aSeats,
      baselineBot: b,
      prePlayBot: b,
      prePlayBotBySeat: aSeats.map((isA) => (isA ? a : undefined)),
      randomPrePlay: prePlay === 'random',
      maxCandidates: 128,
      randomize: false,
      dealSeedBase: seed,
    });
    let aMs = 0;
    let bMs = 0;
    let aPlays = 0;
    let bPlays = 0;
    for (;;) {
      const t0 = performance.now();
      const { episodes, pending } = slot.advance();
      bMs += performance.now() - t0;
      if (episodes.length > 0) {
        const e = episodes[0]!;
        // Every seat plays once a trick, so B made as many plays as A.
        bPlays = e.numTricks * 4 - aPlays;
        halves.push({
          landlord: e.landlordSeat,
          attacker_points: e.attackerPoints,
          digest: e.dealDigest,
          a_s: aMs / 1000,
          b_s: bMs / 1000,
          a_plays: aPlays,
          b_plays: bPlays,
        });
        break;
      }
      if (pending === null) throw new Error(`deal ${game} half ${half}: stalled`);
      const t1 = performance.now();
      const action = pickBotAction(slot.gameState, pending.seat, a);
      aMs += performance.now() - t1;
      aPlays += 1;
      if (action.kind !== 'play_cards') {
        throw new Error(`${a} answered a play with ${action.kind}`);
      }
      // The engine's reducer judges the move exactly as for B's.
      const counts = new Array<number>(NUM_SLOTS).fill(0);
      for (const c of action.cards) counts[cardSlot(c)] = counts[cardSlot(c)]! + 1;
      slot.playCounts(counts);
    }
  }
  if (halves[0]!.digest !== halves[1]!.digest) {
    throw new Error(`deal ${game}: the halves were dealt different cards`);
  }
  return { a, b, seed, pre_play: prePlay, game, halves: [halves[0]!, halves[1]!] };
}

// --- statistics -------------------------------------------------------------

const Z95 = 1.959963984540054;

function meanCi(x: ReadonlyArray<number>): [number, number] {
  const n = x.length;
  const m = x.reduce((s, v) => s + v, 0) / n;
  const v = x.reduce((s, y) => s + (y - m) ** 2, 0) / Math.max(n - 1, 1);
  return [m, Z95 * Math.sqrt(v / n)];
}

// A's result on one hand: won, signed levels, signed attacker points.
function scoreHalf(h: HalfRecord, half: number): { win: number; levels: number; points: number; attacking: boolean } {
  const attacking = !A_SEATS[half]![h.landlord]!;
  const res = computeHandResult(h.attacker_points);
  const won = (res.winner === 'attackers') === attacking;
  return {
    win: won ? 1 : 0,
    levels: won ? res.levelDelta : -res.levelDelta,
    points: attacking ? h.attacker_points : -h.attacker_points,
    attacking,
  };
}

export function summarise(records: ReadonlyArray<DealRecord>): string {
  if (records.length === 0) return 'no deals';
  const dealWin: number[] = [];
  const dealLevels: number[] = [];
  const dealPoints: number[] = [];
  const winAtt: number[] = [];
  const winDef: number[] = [];
  let aS = 0;
  let bS = 0;
  let aPlays = 0;
  let bPlays = 0;
  for (const r of records) {
    const s = r.halves.map((h, i) => scoreHalf(h, i));
    dealWin.push((s[0]!.win + s[1]!.win) / 2);
    dealLevels.push((s[0]!.levels + s[1]!.levels) / 2);
    dealPoints.push((s[0]!.points + s[1]!.points) / 2);
    for (const x of s) (x.attacking ? winAtt : winDef).push(x.win);
    for (const h of r.halves) {
      aS += h.a_s;
      bS += h.b_s;
      aPlays += h.a_plays;
      bPlays += h.b_plays;
    }
  }
  const [w, hw] = meanCi(dealWin);
  const [l, hl] = meanCi(dealLevels);
  const [p, hp] = meanCi(dealPoints);
  const hands = 2 * records.length;
  const mean = (x: number[]): number => x.reduce((s, v) => s + v, 0) / Math.max(x.length, 1);
  const r0 = records[0]!;
  return [
    `${r0.a} (A) vs ${r0.b} (B), pre-play ${r0.pre_play}, seed ${r0.seed}: ${records.length} deals, ${hands} hands`,
    `  A win ${w.toFixed(3)} ± ${hw.toFixed(3)}   levels ${l >= 0 ? '+' : ''}${l.toFixed(3)} ± ${hl.toFixed(3)}   ` +
      `points ${p >= 0 ? '+' : ''}${p.toFixed(2)} ± ${hp.toFixed(2)}   (per deal, 95%, clustered by deal)`,
    `  A win attacking ${mean(winAtt).toFixed(3)}, defending ${mean(winDef).toFixed(3)}`,
    `  time choosing plays: A ${(aS / hands).toFixed(2)} s/hand (${((1000 * aS) / Math.max(aPlays, 1)).toFixed(0)} ms/play), ` +
      `B ${(bS / hands).toFixed(2)} s/hand (${((1000 * bS) / Math.max(bPlays, 1)).toFixed(0)} ms/play, engine included); ` +
      `A/B ${(aS / Math.max(bS, 1e-9)).toFixed(2)}x`,
  ].join('\n');
}

function readRecords(path: string): DealRecord[] {
  if (!existsSync(path)) return [];
  return readFileSync(path, 'utf8')
    .split('\n')
    .filter((line) => line.trim() !== '')
    .map((line) => JSON.parse(line) as DealRecord);
}

// --- main -------------------------------------------------------------------

async function main(): Promise<void> {
  if (has('summary')) {
    const files = argv.filter((x, i) => i > argv.indexOf('--summary') && !x.startsWith('--'));
    console.log(summarise(files.flatMap(readRecords)));
    return;
  }
  const a = arg('a', 'smart-v35') as BotType;
  const b = arg('b', 'smart-v34') as BotType;
  const deals = Number(arg('deals', '16'));
  const seed = Number(arg('seed', '20261119'));
  const prePlay = arg('pre-play', 'random');
  const out = arg('out', '');
  const procs = Number(arg('procs', '1'));
  if (prePlay !== 'random' && prePlay !== 'bot') {
    throw new Error(`--pre-play must be random or bot, got ${prePlay}`);
  }
  if (!Number.isInteger(deals) || deals < 1) throw new Error(`--deals ${deals}`);

  if (procs > 1) {
    if (out === '') throw new Error('--procs needs --out');
    const self = process.argv[1]!;
    const children = [];
    for (let shard = 0; shard < procs; shard++) {
      const args = [
        ...process.execArgv, self,
        '--a', a, '--b', b, '--deals', String(deals), '--seed', String(seed),
        '--pre-play', prePlay, '--shards', String(procs), '--shard', String(shard),
        '--out', `${out}.shard${shard}`,
      ];
      children.push(
        new Promise<void>((resolve, reject) => {
          const child = spawn(process.execPath, args, { stdio: ['ignore', 'ignore', 'inherit'] });
          child.on('exit', (code) =>
            code === 0 ? resolve() : reject(new Error(`shard ${shard} exited ${code}`)),
          );
        }),
      );
    }
    await Promise.all(children);
    const merged = [...readRecords(out)];
    const have = new Set(merged.map((r) => r.game));
    for (let shard = 0; shard < procs; shard++) {
      for (const r of readRecords(`${out}.shard${shard}`)) {
        if (have.has(r.game)) continue;
        have.add(r.game);
        merged.push(r);
        appendFileSync(out, JSON.stringify(r) + '\n');
      }
      rmSync(`${out}.shard${shard}`, { force: true });
    }
    merged.sort((x, y) => x.game - y.game);
    console.log(summarise(merged.filter((r) => r.game < deals)));
    return;
  }

  const shards = Number(arg('shards', '1'));
  const shard = Number(arg('shard', '0'));
  // Resume: deals already in the record (or this shard's file) are done.
  const done = new Set([...readRecords(out.replace(/\.shard\d+$/, '')), ...readRecords(out)].map((r) => r.game));
  const records: DealRecord[] = [];
  for (let game = shard; game < deals; game += shards) {
    if (done.has(game)) continue;
    const r = playDeal(a, b, seed, prePlay, game);
    records.push(r);
    if (out !== '') appendFileSync(out, JSON.stringify(r) + '\n');
    process.stderr.write(`botMatch: deal ${game} done\n`);
  }
  if (out === '') console.log(summarise(records));
}

main().catch((err: unknown) => {
  process.stderr.write(`botMatch: ${err instanceof Error ? err.stack : String(err)}\n`);
  process.exit(1);
});
