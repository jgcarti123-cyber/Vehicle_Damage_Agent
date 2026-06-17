"""Stage 2 VLM pseudo-labeling via OpenAI gpt-4o-mini.

Subcommands:
    label    Sample crops and send to gpt-4o-mini for severity labels (default).
    filter   Filter labels.csv by confidence -> labels_filtered.csv.
    review   Print labeled crops for manual spot-checking.

Examples:
    python stage2_severity/label_crops.py label \\
        --crops-dir outputs/stage2_crops/crops \\
        --extra-crops-dir outputs/stage2_crops_val/crops \\
        --output stage2_severity/data/labels.csv

    python stage2_severity/label_crops.py filter --min-confidence 0.75

    python stage2_severity/label_crops.py review --limit 50
    python stage2_severity/label_crops.py review --severity mild --limit 20
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEVERITY_LABELS = {"mild", "moderate", "severe", "total_loss"}
CLASS_NAMES = {0: "dent", 1: "scratch", 2: "glass_damage", 3: "light_damage", 4: "mirror_damage"}

# Target sample sizes per class from the *main* (train) crop pool.
# None = take all available. Val supplements are applied on top for classes 2 & 4.
DEFAULT_TARGETS: dict[int, int | None] = {
    0: 1246,  # dent          — proportional sample
    1: 650,   # scratch        — proportional sample
    2: None,  # glass_damage   — keep all (thin class, also pull val)
    3: 200,   # light_damage   — proportional sample
    4: None,  # mirror_damage  — keep all (very thin, also pull val)
}

# V2 targets: bias toward moderate/severe-rich classes from the unlabeled pool.
# light_damage = all remaining (~986), dent = 1000 more, scratch = 300 more.
EXTRA_TARGETS: dict[int, int | None] = {
    0: 1000,  # dent          — more from the large unlabeled pool
    1: 300,   # scratch        — small top-up
    2: 0,     # glass_damage   — already labeled all
    3: None,  # light_damage   — label ALL remaining (~986)
    4: 0,     # mirror_damage  — already labeled all
}

OPENAI_MODEL = "gpt-4o-mini"

# Use DAMAGE_TYPE as a plain placeholder — avoids .format() conflicts with
# the literal { } braces in the JSON example.
PROMPT_TEMPLATE = """\
You are a vehicle insurance damage assessor. Classify the severity of the damage visible in this crop.

Damage type: DAMAGE_TYPE

SEVERITY DEFINITIONS — use visual evidence only, not estimated costs:

mild
  - Superficial scratches < 10 cm with no paint depth penetration
  - Small dents < 2 cm diameter with paint intact
  - Minor scuffs or paint transfers
  - No cracked or broken parts

moderate
  - Dents 2–10 cm or deep scratches exposing primer or bare metal
  - Cracked or shattered lights/lenses (not broken off entirely)
  - Multiple or compound scratches covering a significant panel area
  - Panel deformation without structural impact; car fully driveable

severe
  - Dents > 10 cm, panel crush, or bent panel edges
  - Broken or missing parts (mirror casing detached, bumper hanging off)
  - Shattered windshield or major glass loss
  - Structural deformation visible; car may not be driveable

total_loss
  - Multiple severe panels, frame/chassis damage, or fire damage
  - Repair cost would clearly exceed the car's market value

If uncertain between two adjacent levels, pick the more severe one.

Respond with valid JSON only. No other text:
{
  "severity": "<mild|moderate|severe|total_loss>",
  "confidence": <0.0-1.0>,
  "reasoning": "<one sentence citing the specific visual evidence>"
}"""

CSV_FIELDS = ["crop_path", "image_stem", "damage_type", "severity",
              "confidence", "reasoning", "model"]
ERROR_FIELDS = ["crop_path", "error"]


# ---------------------------------------------------------------------------
# Crop discovery and sampling
# ---------------------------------------------------------------------------

def _class_id(p: Path) -> int | None:
    try:
        return int(p.stem.split("_")[0])
    except (ValueError, IndexError):
        return None


def gather_crops(crops_dir: Path) -> dict[int, list[Path]]:
    by_class: dict[int, list[Path]] = {c: [] for c in CLASS_NAMES}
    for p in sorted(crops_dir.rglob("*.jpg")):
        cls = _class_id(p)
        if cls is not None and cls in by_class:
            by_class[cls].append(p)
    return by_class


def build_sample(
    main: dict[int, list[Path]],
    extra: dict[int, list[Path]] | None,
    targets: dict[int, int | None] = DEFAULT_TARGETS,
    seed: int = 42,
) -> list[Path]:
    """Stratified sample: proportional for abundant classes, all for thin ones.
    Val crops (extra) supplement glass_damage and mirror_damage only."""
    rng = random.Random(seed)
    sample: list[Path] = []

    for cls_id, target in targets.items():
        pool = list(main.get(cls_id, []))
        # Supplement thin classes (glass_damage=2, mirror_damage=4) with val crops.
        if extra and cls_id in (2, 4):
            pool = pool + extra.get(cls_id, [])

        if target is None or target >= len(pool):
            chosen = pool
        else:
            chosen = rng.sample(pool, target)

        name = CLASS_NAMES[cls_id]
        tgt_str = "all" if (target is None or target >= len(pool)) else str(target)
        print(f"  {name:15s}: {len(chosen):5d} selected  (pool={len(pool)}, target={tgt_str})")
        sample.extend(chosen)

    rng.shuffle(sample)
    return sample


# ---------------------------------------------------------------------------
# OpenAI API  (openai >= 1.0)
# ---------------------------------------------------------------------------

def _init_client(api_key: str):
    from openai import OpenAI
    return OpenAI(api_key=api_key, timeout=45.0)


class DailyQuotaExhausted(Exception):
    """Raised when the account quota/credit is exhausted — stop the run."""


def _label_one(client, crop_path: Path, model_name: str) -> dict:
    import base64

    cls_id = _class_id(crop_path)
    damage_type = CLASS_NAMES.get(cls_id, "unknown")
    prompt = PROMPT_TEMPLATE.replace("DAMAGE_TYPE", damage_type)
    img_b64 = base64.b64encode(crop_path.read_bytes()).decode("utf-8")

    last_exc: Exception | None = None
    for _attempt, backoff in enumerate([0, 15, 30, 60]):
        if backoff:
            time.sleep(backoff)
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}",
                                "detail": "low",  # 85 tokens/image — minimises cost
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=256,
            )
            break
        except Exception as e:
            msg = str(e)
            # Billing / hard quota exhausted — stop the run.
            if any(kw in msg.lower() for kw in ("insufficient_quota", "billing", "credit")):
                raise DailyQuotaExhausted(
                    f"OpenAI quota/credits exhausted: {msg}"
                ) from e
            # Transient errors — retry with backoff.
            if any(code in msg for code in ("503", "500", "502", "timed out", "timeout")):
                last_exc = e
                continue
            # Per-minute rate limit — retry with backoff.
            if "429" in msg:
                last_exc = e
                continue
            raise
    else:
        raise last_exc  # all retries failed

    text = response.choices[0].message.content.strip()

    # Strip markdown code fences if model ignores response_format.
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1].lstrip("json").strip() if len(parts) > 1 else text

    data = json.loads(text)
    severity = data.get("severity", "")
    if severity not in SEVERITY_LABELS:
        raise ValueError(f"Invalid severity: {severity!r}")

    return {
        "severity": severity,
        "confidence": float(data["confidence"]),
        "reasoning": str(data.get("reasoning", "")),
        "model": model_name,
    }


# ---------------------------------------------------------------------------
# label subcommand
# ---------------------------------------------------------------------------

def label_command(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv
    import os
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY") or getattr(args, "api_key", None)
    if not api_key:
        sys.exit("OPENAI_API_KEY not set — add it to .env or pass --api-key.")

    crops_dir = Path(args.crops_dir)
    extra_dir = Path(args.extra_crops_dir) if args.extra_crops_dir else None
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    errors_path = output.with_name(output.stem + "_errors.csv")
    sample_path = Path(args.sample_file) if args.sample_file else output.with_name(output.stem + "_sample.txt")

    # Load crops to exclude (already labeled in a previous run).
    excluded: set[str] = set()
    if args.exclude_csv:
        for excl_path in args.exclude_csv:
            ep = Path(excl_path)
            if ep.exists():
                with open(ep, newline="") as f:
                    for row in csv.DictReader(f):
                        excluded.add(row["crop_path"])
        print(f"Excluding {len(excluded)} already-labeled crops.")

    # Gather
    print(f"Gathering crops from {crops_dir} ...")
    main_by_class = gather_crops(crops_dir)
    extra_by_class = gather_crops(extra_dir) if extra_dir else None
    if extra_dir:
        print(f"Extra (val) crops from {extra_dir}")

    # Filter excluded crops from pool.
    if excluded:
        for cls_id in main_by_class:
            main_by_class[cls_id] = [p for p in main_by_class[cls_id] if str(p) not in excluded]
        if extra_by_class:
            for cls_id in extra_by_class:
                extra_by_class[cls_id] = [p for p in extra_by_class[cls_id] if str(p) not in excluded]
        print("Pool after exclusion:")
        for cls_id, name in CLASS_NAMES.items():
            n = len(main_by_class.get(cls_id, []))
            print(f"  {name:15s}: {n}")

    # Choose targets
    targets = EXTRA_TARGETS if args.extra else DEFAULT_TARGETS

    # Build or reload sample
    if sample_path.exists():
        sample = [Path(p.strip()) for p in sample_path.read_text().splitlines() if p.strip()]
        print(f"Reloaded existing sample ({len(sample)} crops) from {sample_path}")
    else:
        print("Building stratified sample:")
        sample = build_sample(main_by_class, extra_by_class, targets=targets)
        sample_path.write_text("\n".join(str(p) for p in sample))
        print(f"Sample of {len(sample)} crops saved → {sample_path}")

    # Load checkpoint
    done: set[str] = set()
    if output.exists():
        with open(output, newline="") as f:
            for row in csv.DictReader(f):
                done.add(row["crop_path"])
        print(f"Checkpoint: {len(done)} already labeled, {len(sample) - len(done)} remaining.")

    remaining = [p for p in sample if str(p) not in done]
    if not remaining:
        print("All crops already labeled — run 'filter' next.")
        return

    # Init OpenAI
    print("Connecting to OpenAI ...")
    client = _init_client(api_key)
    model_name = args.model or OPENAI_MODEL
    eta_min = len(remaining) * args.rate_sleep / 60
    print(f"Model: {model_name}  |  rate-sleep: {args.rate_sleep}s  |  "
          f"todo: {len(remaining)}  |  ETA: ~{eta_min:.0f} min")

    # Open CSV for appending
    write_header = not output.exists() or output.stat().st_size == 0
    out_f = open(output, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()
        out_f.flush()

    err_write_header = not errors_path.exists() or errors_path.stat().st_size == 0
    err_f = open(errors_path, "a", newline="", encoding="utf-8")
    err_writer = csv.DictWriter(err_f, fieldnames=ERROR_FIELDS)
    if err_write_header:
        err_writer.writeheader()

    n_ok = n_err = 0
    total = len(remaining)

    try:
        for i, crop_path in enumerate(remaining, 1):
            cls_id = _class_id(crop_path)
            damage_type = CLASS_NAMES.get(cls_id, "unknown")
            try:
                result = _label_one(client, crop_path, model_name)
                writer.writerow({
                    "crop_path": str(crop_path),
                    "image_stem": crop_path.parent.name,
                    "damage_type": damage_type,
                    **result,
                })
                out_f.flush()
                n_ok += 1
            except DailyQuotaExhausted as e:
                print(f"\n[STOP] {e}")
                print(f"Progress saved. Re-run to continue from crop {i}/{total}.")
                break
            except Exception as e:
                err_writer.writerow({"crop_path": str(crop_path), "error": str(e)})
                err_f.flush()
                n_err += 1
                print(f"  ERROR [{i}/{total}] {crop_path.name}: {e}")

            if i % 10 == 0 or i == total:
                pct = i / total * 100
                print(f"[{i}/{total} {pct:.0f}%] labeled={n_ok} errors={n_err}")

            if i < total:
                time.sleep(args.rate_sleep)
    finally:
        out_f.close()
        err_f.close()

    print(f"\nDone. {n_ok} labeled, {n_err} errors.")
    print(f"Labels  : {output}")
    if n_err:
        print(f"Errors  : {errors_path}")
    print("Next    : python stage2_severity/label_crops.py filter --min-confidence 0.75")


# ---------------------------------------------------------------------------
# filter subcommand
# ---------------------------------------------------------------------------

def filter_command(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        sys.exit(f"Labels file not found: {input_path}")

    rows: list[dict] = []
    with open(input_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    filtered = [r for r in rows if float(r["confidence"]) >= args.min_confidence]
    print(f"Total: {len(rows)} | Kept (conf>={args.min_confidence}): {len(filtered)} | "
          f"Dropped: {len(rows) - len(filtered)}")

    print("\nPer damage-type:")
    by_dmg: dict[str, int] = {}
    for r in filtered:
        by_dmg[r["damage_type"]] = by_dmg.get(r["damage_type"], 0) + 1
    for name in ["dent", "scratch", "glass_damage", "light_damage", "mirror_damage"]:
        print(f"  {name:15s}: {by_dmg.get(name, 0)}")

    print("\nSeverity distribution:")
    by_sev: dict[str, int] = {}
    for r in filtered:
        by_sev[r["severity"]] = by_sev.get(r["severity"], 0) + 1
    for sev in ["mild", "moderate", "severe", "total_loss"]:
        print(f"  {sev:12s}: {by_sev.get(sev, 0)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(filtered)
    print(f"\nWritten → {output_path}")


# ---------------------------------------------------------------------------
# review subcommand
# ---------------------------------------------------------------------------

def review_command(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Labels file not found: {input_path}")

    with open(input_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if args.severity:
        rows = [r for r in rows if r["severity"] == args.severity]

    rows = rows[:args.limit]
    print(f"Reviewing {len(rows)} crops:\n")
    for r in rows:
        print(f"  path      : {r['crop_path']}")
        print(f"  damage    : {r['damage_type']}")
        print(f"  severity  : {r['severity']}  (conf={float(r['confidence']):.2f})")
        print(f"  reasoning : {r['reasoning']}")
        print()


# ---------------------------------------------------------------------------
# relabel subcommand  — re-label specific severity classes from an existing CSV
# ---------------------------------------------------------------------------

def relabel_command(args: argparse.Namespace) -> None:
    """Re-label moderate/severe crops from an existing CSV with the improved prompt."""
    from dotenv import load_dotenv
    import os
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY") or getattr(args, "api_key", None)
    if not api_key:
        sys.exit("OPENAI_API_KEY not set — add it to .env or pass --api-key.")

    input_path = Path(args.input_csv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    errors_path = output.with_name(output.stem + "_errors.csv")

    filter_sevs = set(args.filter_severity)

    # Load crop paths to re-label from the input CSV.
    to_relabel: list[tuple[Path, str]] = []  # (crop_path, damage_type)
    with open(input_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["severity"] in filter_sevs:
                p = Path(row["crop_path"])
                if p.exists():
                    to_relabel.append((p, row["damage_type"]))

    print(f"Loaded {len(to_relabel)} crops from {input_path.name} "
          f"with severity in {filter_sevs}")

    # Checkpoint: skip crop_paths already in output.
    done: set[str] = set()
    if output.exists():
        with open(output, newline="") as f:
            for row in csv.DictReader(f):
                done.add(row["crop_path"])
        print(f"Checkpoint: {len(done)} already re-labeled, "
              f"{len(to_relabel) - len(done)} remaining.")

    remaining = [(p, dt) for p, dt in to_relabel if str(p) not in done]
    if not remaining:
        print("All crops already re-labeled.")
        return

    # Init OpenAI
    print("Connecting to OpenAI ...")
    client = _init_client(api_key)
    model_name = args.model or OPENAI_MODEL
    eta_min = len(remaining) * args.rate_sleep / 60
    print(f"Model: {model_name}  |  rate-sleep: {args.rate_sleep}s  |  "
          f"todo: {len(remaining)}  |  ETA: ~{eta_min:.0f} min")

    write_header = not output.exists() or output.stat().st_size == 0
    out_f = open(output, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()
        out_f.flush()

    err_write_header = not errors_path.exists() or errors_path.stat().st_size == 0
    err_f = open(errors_path, "a", newline="", encoding="utf-8")
    err_writer = csv.DictWriter(err_f, fieldnames=ERROR_FIELDS)
    if err_write_header:
        err_writer.writeheader()

    n_ok = n_err = 0
    total = len(remaining)

    try:
        for i, (crop_path, damage_type) in enumerate(remaining, 1):
            try:
                result = _label_one(client, crop_path, model_name)
                writer.writerow({
                    "crop_path": str(crop_path),
                    "image_stem": crop_path.parent.name,
                    "damage_type": damage_type,
                    **result,
                })
                out_f.flush()
                n_ok += 1
            except DailyQuotaExhausted as e:
                print(f"\n[STOP] {e}")
                print(f"Progress saved. Re-run to continue from crop {i}/{total}.")
                break
            except Exception as e:
                err_writer.writerow({"crop_path": str(crop_path), "error": str(e)})
                err_f.flush()
                n_err += 1
                print(f"  ERROR [{i}/{total}] {crop_path.name}: {e}")

            if i % 10 == 0 or i == total:
                pct = i / total * 100
                print(f"[{i}/{total} {pct:.0f}%] labeled={n_ok} errors={n_err}")

            if i < total:
                time.sleep(args.rate_sleep)
    finally:
        out_f.close()
        err_f.close()

    print(f"\nDone. {n_ok} re-labeled, {n_err} errors.")
    print(f"Output  : {output}")
    print("Next    : merge relabeled CSV with cleaned mild CSV, then filter + train")


# ---------------------------------------------------------------------------
# merge subcommand
# ---------------------------------------------------------------------------

def merge_command(args: argparse.Namespace) -> None:
    """Concatenate multiple label CSVs, deduplicate by crop_path, write merged file."""
    inputs = [Path(p) for p in args.inputs]
    output_path = Path(args.output)

    seen: set[str] = set()
    merged: list[dict] = []
    for inp in inputs:
        if not inp.exists():
            print(f"  WARNING: {inp} not found, skipping.")
            continue
        with open(inp, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["crop_path"] not in seen:
                    seen.add(row["crop_path"])
                    merged.append(row)
        print(f"  Loaded {inp.name}: cumulative {len(merged)} unique rows")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(merged)

    by_sev: dict[str, int] = {}
    for r in merged:
        by_sev[r["severity"]] = by_sev.get(r["severity"], 0) + 1
    print(f"\nMerged {len(merged)} rows → {output_path}")
    print("Severity distribution:")
    for sev in ["mild", "moderate", "severe", "total_loss"]:
        print(f"  {sev:12s}: {by_sev.get(sev, 0)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 2 VLM pseudo-labeling (OpenAI)")
    sub = p.add_subparsers(dest="command")

    lp = sub.add_parser("label", help="Label crops with gpt-4o-mini (checkpointed)")
    lp.add_argument("--crops-dir", required=True,
                    help="Train crops root, e.g. outputs/stage2_crops/crops")
    lp.add_argument("--extra-crops-dir", default=None,
                    help="Val crops root — supplements mirror_damage and glass_damage")
    lp.add_argument("--output", default="stage2_severity/data/labels.csv")
    lp.add_argument("--sample-file", default=None,
                    help="Override sample .txt path (default: <output_stem>_sample.txt)")
    lp.add_argument("--exclude-csv", nargs="*", default=None,
                    help="One or more CSV files of already-labeled crops to exclude from sampling")
    lp.add_argument("--extra", action="store_true",
                    help="Use EXTRA_TARGETS (bias toward light_damage/dent for v2 run)")
    lp.add_argument("--api-key", default=None,
                    help="OpenAI API key (overrides OPENAI_API_KEY env var)")
    lp.add_argument("--model", default=None,
                    help=f"OpenAI model name (default: {OPENAI_MODEL})")
    lp.add_argument("--rate-sleep", type=float, default=0.5,
                    help="Seconds between API calls (default 0.5s)")

    rl = sub.add_parser("relabel", help="Re-label specific severity classes with the improved prompt")
    rl.add_argument("--input-csv", required=True,
                    help="Existing labels CSV (e.g. labels_filtered.csv)")
    rl.add_argument("--filter-severity", nargs="+",
                    default=["moderate", "severe"],
                    choices=list(SEVERITY_LABELS),
                    help="Only re-label crops with these original severity labels")
    rl.add_argument("--output", default="stage2_severity/data/labels_relabeled.csv")
    rl.add_argument("--api-key", default=None)
    rl.add_argument("--model", default=None)
    rl.add_argument("--rate-sleep", type=float, default=0.5)

    mp = sub.add_parser("merge", help="Merge multiple label CSVs, deduplicate by crop_path")
    mp.add_argument("inputs", nargs="+", help="Input CSV files to merge")
    mp.add_argument("--output", default="stage2_severity/data/labels_merged.csv")

    fp = sub.add_parser("filter", help="Filter labels.csv by confidence")
    fp.add_argument("--input", default="stage2_severity/data/labels.csv")
    fp.add_argument("--output", default="stage2_severity/data/labels_filtered.csv")
    fp.add_argument("--min-confidence", type=float, default=0.75)

    rp = sub.add_parser("review", help="Spot-check labeled crops in terminal")
    rp.add_argument("--input", default="stage2_severity/data/labels.csv")
    rp.add_argument("--limit", type=int, default=50)
    rp.add_argument("--severity", default=None, choices=list(SEVERITY_LABELS),
                    help="Show only crops with this severity label")

    return p


def main() -> None:
    p = _build_parser()
    args = p.parse_args()
    if args.command is None:
        p.print_help()
        sys.exit(1)
    {"label": label_command, "filter": filter_command, "review": review_command,
     "merge": merge_command, "relabel": relabel_command}[args.command](args)


if __name__ == "__main__":
    main()
