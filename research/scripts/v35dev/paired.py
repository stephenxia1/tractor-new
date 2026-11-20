"""Scratch: B - A per deal for two match.ts runs on the same seed/deals."""
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

A_SEATS = ({0, 2}, {1, 3})

def per_deal(path):
    out = {}
    for l in open(path):
        if not l.strip(): continue
        r = json.loads(l)
        w = lv = pt = 0.0
        for h, half in enumerate(r['halves']):
            p = half['attacker_points']
            att = half['landlord'] not in A_SEATS[h]
            win, d = hand_result(p)
            iw = (win == 'att') == att
            w += (1 if iw else 0) / 2; lv += (d if iw else -d) / 2; pt += (p if att else -p) / 2
        out[r['game']] = (w, lv, pt)
    return out

def mci(x):
    n = len(x); m = sum(x) / n
    v = sum((a - m) ** 2 for a in x) / max(n - 1, 1)
    return m, Z * math.sqrt(v / n)

a, b = per_deal(sys.argv[1]), per_deal(sys.argv[2])
games = sorted(set(a) & set(b))
for i, name in enumerate(('win', 'levels', 'points')):
    m, h = mci([b[g][i] - a[g][i] for g in games])
    print(f"{name}: B - A {m:+.3f} ± {h:.3f}  ({len(games)} deals)")
