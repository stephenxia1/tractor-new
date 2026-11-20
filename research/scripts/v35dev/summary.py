"""Scratch: summarise match.ts JSONL (A's view), win/levels/points with 95% CIs."""
import json, math, sys

Z = 1.959963984540054

def hand_result(p):
    if p == 0: return ('def', 3)
    if p < 40: return ('def', 2)
    if p < 80: return ('def', 1)
    if p < 120: return ('att', 0)
    if p < 160: return ('att', 1)
    if p < 200: return ('att', 2)
    return ('att', 3)

def mci(x):
    n = len(x); m = sum(x) / n
    v = sum((a - m) ** 2 for a in x) / max(n - 1, 1)
    return m, Z * math.sqrt(v / n)

A_SEATS = ({0, 2}, {1, 3})
for path in sys.argv[1:]:
    recs = [json.loads(l) for l in open(path) if l.strip()]
    wins, dwin, lev, pts, ams, bms, adec = [], [], [], [], 0.0, 0.0, 0
    for r in recs:
        w2, l2, p2 = [], [], []
        for h, half in enumerate(r['halves']):
            p = half['attacker_points']
            attacking = half['landlord'] not in A_SEATS[h]
            winner, d = hand_result(p)
            iwin = (winner == 'att') == attacking
            w2.append(1.0 if iwin else 0.0)
            l2.append(d if iwin else -d)
            p2.append(p if attacking else -p)
            ams += half['a_ms']; bms += half['b_ms']; adec += half['a_dec']
        wins += w2
        dwin.append(sum(w2) / 2)
        lev.append(sum(l2) / 2)
        pts.append(sum(p2) / 2)
    n = len(recs)
    w, hw = mci(wins); dw, hdw = mci(dwin); l, hl = mci(lev); p, hp = mci(pts)
    print(f"{path}: {n} deals  win {w:.3f}±{hw:.3f} (deal-clustered ±{hdw:.3f})  "
          f"levels {l:+.3f}±{hl:.3f}  points {p:+.2f}±{hp:.2f}  "
          f"A {ams/1000/n/2:.2f}s/hand ({ams/max(adec,1):.0f} ms/dec)  B+engine {bms/1000/n/2:.2f}s/hand")
