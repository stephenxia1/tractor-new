"""Reports what this project has cost, separately from the rest of the account.

    python research/scripts/aws_cost.py            # last 14 days
    python research/scripts/aws_cost.py --days 30
    python research/scripts/aws_cost.py --all      # every project, side by side

Separation rests on the `Project` tag, which `provision2.py` puts on the
instance, its EBS volume and the spot request, and which is activated as a
cost allocation tag. Two caveats worth knowing before trusting a number:

* Tag activation is **not retroactive**. Usage from before the tag was
  activated (2026-09-07) is real but carries no tag, so it lands in
  `(untagged)` and cannot be split out after the fact.
* This account is running on **credits**. `UnblendedCost` is the gross
  charge; the credit is a separate record type that cancels it. The gross
  number is the one to watch against a budget, since it is what will be
  charged once credits run out -- so that is what this prints, with the
  net shown alongside.
"""

from __future__ import annotations

import argparse
import datetime

import boto3

TAG_KEY = "Project"
PROJECT = "tractor-rl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--all", action="store_true", help="break down by project instead")
    ap.add_argument("--region", default="us-east-1")
    args = ap.parse_args()

    ce = boto3.session.Session(region_name=args.region).client("ce")
    end = datetime.date.today() + datetime.timedelta(days=1)
    start = end - datetime.timedelta(days=args.days)
    window = {"Start": str(start), "End": str(end)}

    gross = ce.get_cost_and_usage(
        TimePeriod=window, Granularity="MONTHLY", Metrics=["UnblendedCost"],
        Filter={"Not": {"Dimensions": {"Key": "RECORD_TYPE", "Values": ["Credit", "Refund"]}}},
        GroupBy=[{"Type": "TAG", "Key": TAG_KEY}],
    )

    totals: dict[str, float] = {}
    for period in gross["ResultsByTime"]:
        for group in period["Groups"]:
            raw = group["Keys"][0]
            name = raw.split("$", 1)[1] if "$" in raw else raw
            name = name or "(untagged)"
            totals[name] = totals.get(name, 0.0) + float(group["Metrics"]["UnblendedCost"]["Amount"])

    print(f"{start} to {end - datetime.timedelta(days=1)}  (gross, before credits)")
    if args.all:
        for name, amount in sorted(totals.items(), key=lambda kv: -kv[1]):
            print(f"  {name:<24} ${amount:8.2f}")
    else:
        print(f"  {PROJECT:<24} ${totals.get(PROJECT, 0.0):8.2f}")
        other = sum(v for k, v in totals.items() if k != PROJECT)
        print(f"  {'everything else':<24} ${other:8.2f}")
    print(f"  {'-' * 24} {'-' * 9}")
    print(f"  {'total':<24} ${sum(totals.values()):8.2f}")

    net = ce.get_cost_and_usage(TimePeriod=window, Granularity="MONTHLY", Metrics=["UnblendedCost"])
    net_total = sum(float(p["Total"]["UnblendedCost"]["Amount"]) for p in net["ResultsByTime"])
    print(f"\n  net after credits:       ${net_total:8.2f}")

    if totals.get("(untagged)", 0) > 0 and not args.all:
        print("\n  note: untagged spend predates tag activation and cannot be split by project.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
