import json, math, sys
Z = 1.96
def mci(x):
    n = len(x); m = sum(x) / n
    v = sum((a - m) ** 2 for a in x) / max(n - 1, 1)
    return m, Z * math.sqrt(v / n)
for path in sys.argv[1:]:
    r = [json.loads(l) for l in open(path) if l.strip()]
    # Landlord's view: fewer attacker points is better; win = attackers < 80.
    dp = [x['y_att'] - x['x_att'] for x in r]
    dw = [(x['x_att'] < 80) - (x['y_att'] < 80) for x in r]
    m, h = mci(dp); mw, hw = mci(dw)
    print(f"{path}: {len(r)} deals  X better by {m:+.2f}±{h:.2f} attacker points, "
          f"landlord win {mw:+.3f}±{hw:.3f}; kitty pts X {sum(x['x_kitty'] for x in r)/len(r):.1f} "
          f"Y {sum(x['y_kitty'] for x in r)/len(r):.1f}")
