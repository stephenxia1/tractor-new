'use client';
import Link from 'next/link';
import { useEffect, useState } from 'react';
import type { MatchupStats, StatsSummary } from '@tractor/protocol';
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

// Who is ahead, by the point margin's 95% interval (it needs about a
// quarter of the hands the win rate does to show the same gap).
function verdict(m: MatchupStats): { text: string; cls: string } {
  const h = m.pointMarginHalfWidth;
  if (h === null) return { text: 'Too few hands', cls: 'verdict-open' };
  if (m.pointMargin - h > 0) return { text: 'Humans ahead', cls: 'verdict-humans' };
  if (m.pointMargin + h < 0) return { text: 'Bots ahead', cls: 'verdict-bots' };
  return { text: 'Not yet clear', cls: 'verdict-open' };
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
                    <th className="num">Hands</th>
                    <th className="num">Players</th>
                    <th>So far</th>
                    <th className="num">Hands won (95% interval)</th>
                    <th className="num">Point margin</th>
                    <th className="num">Levels</th>
                    <th className="num">Won defending</th>
                    <th className="num">Won attacking</th>
                    <th className="num">Games won</th>
                  </tr>
                </thead>
                <tbody>
                  {summary.matchups.map((m) => {
                    const v = verdict(m);
                    return (
                      <tr key={m.key}>
                        <td>{m.humanSide}</td>
                        <td>{m.botSide}</td>
                        <td className="num">{m.hands}</td>
                        <td className="num">{m.players}</td>
                        <td className={v.cls}>{v.text}</td>
                        <td className="num">
                          {pct(m.winRate)}{' '}
                          <span className="muted">
                            ({pct(m.winRateLow)}–{pct(m.winRateHigh)})
                          </span>
                        </td>
                        <td className="num">{withHalfWidth(m.pointMargin, m.pointMarginHalfWidth, 1)}</td>
                        <td className="num">{withHalfWidth(m.levels, m.levelsHalfWidth, 2)}</td>
                        <td className="num">
                          {m.defending.wins}/{m.defending.hands}
                        </td>
                        <td className="num">
                          {m.attacking.wins}/{m.attacking.hands}
                        </td>
                        <td className="num">
                          {m.gamesWon}/{m.games}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          <div className="stats-notes muted">
            <p>
              Every finished hand at a table with at least one human is recorded. A row is one
              human team (two humans, or a human with a bot partner) against two bots; hands where
              someone changed seats midway, or with humans on both teams, are left out. All figures
              are from the human team&apos;s side.
            </p>
            <p>
              <b>Hands won:</b> held the attackers under 80 as defenders, or reached 80 as
              attackers. <b>Point margin:</b> average a hand of attacker points − 80 when the
              humans attack, 80 − attacker points when they defend. <b>Levels:</b> average levels
              a hand, counted against the humans when the bots win the hand. ± is a 95% interval.
            </p>
            <p>
              <b>So far</b> says who is ahead once the point margin&apos;s interval excludes zero.
              Against a gap like 55% of hands won, that takes about 160 hands; the win rate alone
              would need about 700.
            </p>
          </div>
        </>
      )}
    </div>
  );
}
