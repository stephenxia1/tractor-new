'use client';
import Link from 'next/link';
import { useEffect, useState } from 'react';
import type { MatchupStats, RoleStats, StatsSummary } from '@tractor/protocol';
import { statsUrl } from '../../lib/ws';

const REFRESH_MS = 30_000;

function pct(x: number): string {
  return `${(100 * x).toFixed(1)}%`;
}

function signed(x: number, digits: number): string {
  const s = x.toFixed(digits);
  return x > 0 ? `+${s}` : s;
}

function withHalfWidth(mean: number, half: number | null, digits: number): string {
  return half === null ? signed(mean, digits) : `${signed(mean, digits)} ± ${half.toFixed(digits)}`;
}

// Who is ahead, by the balanced point margin's 95% interval (it needs
// about a quarter of the hands the win rate does to show the same gap).
function verdict(m: MatchupStats): { text: string; cls: string } {
  const b = m.balanced ?? null;
  if (b === null) return { text: 'Too few hands', cls: 'verdict-open' };
  if (b.pointMargin - b.pointMarginHalfWidth > 0) return { text: 'Humans ahead', cls: 'verdict-humans' };
  if (b.pointMargin + b.pointMarginHalfWidth < 0) return { text: 'Bots ahead', cls: 'verdict-bots' };
  return { text: 'Not yet clear', cls: 'verdict-open' };
}

// The "Both" row: both roles weighted equally once each has two hands,
// until then the plain totals, marked *.
function BothCells({ m }: { m: MatchupStats }) {
  const b = m.balanced ?? null;
  if (b === null) {
    return (
      <>
        <td className="num">
          {pct(m.winRate)}*{' '}
          <span className="muted">
            ({pct(m.winRateLow)}–{pct(m.winRateHigh)})
          </span>
        </td>
        <td className="num">{withHalfWidth(m.pointMargin, m.pointMarginHalfWidth, 1)}*</td>
        <td className="num">{withHalfWidth(m.levels, m.levelsHalfWidth, 2)}*</td>
      </>
    );
  }
  return (
    <>
      <td className="num">
        {pct(b.winRate)} <span className="muted">± {(100 * b.winRateHalfWidth).toFixed(1)}</span>
      </td>
      <td className="num">{withHalfWidth(b.pointMargin, b.pointMarginHalfWidth, 1)}</td>
      <td className="num">{withHalfWidth(b.levels, b.levelsHalfWidth, 2)}</td>
    </>
  );
}

function RoleCells({ r }: { r: RoleStats }) {
  // A server older than the role split sends only hands and wins.
  if (typeof r.winRate !== 'number') {
    return (
      <>
        <td className="num">
          {r.wins}/{r.hands}
        </td>
        <td className="num">—</td>
        <td className="num">—</td>
      </>
    );
  }
  if (r.hands === 0) {
    return (
      <>
        <td className="num muted">—</td>
        <td className="num muted">—</td>
        <td className="num muted">—</td>
      </>
    );
  }
  return (
    <>
      <td className="num">
        {pct(r.winRate)}{' '}
        <span className="muted">
          ({pct(r.winRateLow)}–{pct(r.winRateHigh)})
        </span>
      </td>
      <td className="num">{withHalfWidth(r.pointMargin, r.pointMarginHalfWidth, 1)}</td>
      <td className="num">{withHalfWidth(r.levels, r.levelsHalfWidth, 2)}</td>
    </>
  );
}

function MatchupRows({ m }: { m: MatchupStats }) {
  const v = verdict(m);
  return (
    <>
      <tr className="group-start">
        <td rowSpan={3}>{m.humanSide}</td>
        <td rowSpan={3}>{m.botSide}</td>
        <td>Both</td>
        <td className="num">{m.hands}</td>
        <td className="num">{m.players}</td>
        <td className={v.cls}>{v.text}</td>
        <BothCells m={m} />
        <td className="num">
          {m.gamesWon}/{m.games}
        </td>
      </tr>
      {(['defending', 'attacking'] as const).map((role) => (
        <tr key={role} className="role-row">
          <td>{role === 'defending' ? 'Defending' : 'Attacking'}</td>
          <td className="num">{m[role].hands}</td>
          <td />
          <td />
          <RoleCells r={m[role]} />
          <td />
        </tr>
      ))}
    </>
  );
}

export default function StatsPage() {
  const [summary, setSummary] = useState<StatsSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    const load = async (): Promise<void> => {
      try {
        const res = await fetch(statsUrl(), { cache: 'no-store' });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const body = (await res.json()) as StatsSummary;
        if (alive) {
          setSummary(body);
          setError(null);
        }
      } catch (e) {
        if (alive) setError(String(e));
      }
    };
    void load();
    const timer = setInterval(() => void load(), REFRESH_MS);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);

  return (
    <div className="stats-page">
      <p>
        <Link href="/">← Tractor</Link>
      </p>
      <h1 className="h1">Results vs bots</h1>

      {error !== null && summary === null && (
        <p className="muted">Couldn&apos;t load results ({error}).</p>
      )}
      {summary === null && error === null && <p className="muted">Loading…</p>}

      {summary !== null && (
        <>
          <p className="muted">
            {summary.totalHands === 0
              ? 'No hands recorded yet.'
              : `${summary.totalHands} hand${summary.totalHands === 1 ? '' : 's'} played at tables with a human, by ${summary.players} player${summary.players === 1 ? '' : 's'}, since ${new Date(summary.since!).toLocaleDateString()}. ${summary.matchupHands} of them were humans against bots with the seating unchanged, and count below.`}
          </p>

          {summary.matchups.length === 0 ? (
            <p className="muted">
              Sit at a table with bots on the other team and finish a hand: it will show up here.
            </p>
          ) : (
            <div className="stats-scroll">
              <table className="stats-table">
                <thead>
                  <tr>
                    <th>Human team</th>
                    <th>Bots</th>
                    <th>Humans&apos; role</th>
                    <th className="num">Hands</th>
                    <th className="num">Players</th>
                    <th>So far</th>
                    <th className="num">Hands won</th>
                    <th className="num">Point margin</th>
                    <th className="num">Levels</th>
                    <th className="num">Games won</th>
                  </tr>
                </thead>
                <tbody>
                  {summary.matchups.map((m) => (
                    <MatchupRows key={m.key} m={m} />
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <div className="stats-notes muted">
            <p>
              Every finished hand at a table with at least one human is recorded. A matchup is one
              human team (two humans, or a human with a bot partner) against two bots; hands where
              someone changed seats midway, or with humans on both teams, are left out. All figures
              are from the human team&apos;s side, and ± is a 95% interval.
            </p>
            <p>
              <b>Defending</b> and <b>Attacking</b> are the hands where the humans held the deal or
              tried to take it. <b>Hands won:</b> held the attackers under 80 as defenders, or
              reached 80 as attackers. <b>Point margin:</b> 80 − attacker points when defending,
              attacker points − 80 when attacking. <b>Levels:</b> the levels the hand gave, counted
              against the humans when the bots won it. The two roles have different baselines
              (equal teams need not win half their hands in either), so these rows show how the
              humans fare in each role, not who is better.
            </p>
            <p>
              <b>Both</b> averages the defending and attacking figures with equal weight, so it
              isn&apos;t moved by how often a team defended (the winners keep the deal). Equal teams
              score 50% and 0 there; its point margin is the research evaluations&apos;, (points
              taken − points conceded) / 2. Until each role has two hands it shows the plain totals,
              marked *.
            </p>
            <p>
              <b>So far</b> says who is ahead once Both&apos;s point-margin interval excludes zero.
              Against a gap like 55% of hands won, that takes about 160 hands; the win rate alone
              would need about 700.
            </p>
          </div>
        </>
      )}
    </div>
  );
}
