#!/usr/bin/env python3
"""Plot aggregate scorecard-summary CSV outputs."""

import argparse
import csv
import os
from pathlib import Path
import re

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")


MODEL_LABELS = {
    "control": "Pre-finetuned",
    "mse5000": "MSE-5000",
    "amse5000": "AMSE-5000",
    "amse25000": "AMSE-25000",
    "lamse0p1_lmax32_5000": "LAMSE-0.1-LMAX32",
    "lamse0p5_lmax127_5000": "LAMSE-0.5-LMAX127",
    "lamse0p3_lmax32_5000": "LAMSE-0.3-LMAX32",
}

MODEL_ORDER = {
    "control": 0,
    "mse5000": 1,
    "amse5000": 2,
    "amse25000": 3,
    "lamse0p1_lmax32_5000": 4,
    "lamse0p5_lmax127_5000": 5,
    "lamse0p3_lmax32_5000": 6,
}

DEFAULT_FIELDS = ["z:500", "t:850", "2t", "10m_wind_speed", "msl"]

FIELD_TITLES = {
    ("z", "500"): "z500",
    ("t", "850"): "t850",
    ("2t", ""): "2m temperature",
    ("10m_wind_speed", ""): "10m wind speed",
    ("msl", ""): "MSL pressure",
}


def parse_field(spec):
    if ":" in spec:
        variable, level = spec.split(":", 1)
    else:
        variable, level = spec, ""
    variable = variable.strip()
    level = level.strip()
    if not variable:
        raise argparse.ArgumentTypeError(f"Invalid field spec {spec!r}")
    label = variable if not level else f"{variable}{level}"
    return {"spec": spec, "variable": variable, "level": level, "label": label}


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def norm_level(value):
    value = str(value).strip()
    if value in ("", "nan", "None"):
        return ""
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return str(int(number))
    return str(number)


def read_summary(path):
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["lead_hours"] = float(row["lead_hours"])
                row["mean"] = float(row["mean"])
                row["median"] = float(row["median"])
                row["count"] = int(float(row["count"]))
            except (TypeError, ValueError):
                continue
            row["level"] = norm_level(row.get("level", ""))
            rows.append(row)
    return rows


def select_rows(rows, field, stat, metric, lead_hours):
    selected = []
    allowed_leads = set(float(x) for x in lead_hours) if lead_hours else None
    for row in rows:
        if row.get("variable") != field["variable"]:
            continue
        if norm_level(row.get("level", "")) != norm_level(field["level"]):
            continue
        if row.get("stat") != stat:
            continue
        if allowed_leads is not None and row["lead_hours"] not in allowed_leads:
            continue
        if not np.isfinite(row[metric]):
            continue
        selected.append(row)
    return selected


def group_series(rows, metric):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["model"], []).append((row["lead_hours"], row[metric]))
    for model in grouped:
        grouped[model] = sorted(grouped[model])
    return grouped


def model_label(model):
    return MODEL_LABELS.get(model, model.replace("_", " "))


def field_title(field):
    key = (field["variable"], norm_level(field["level"]))
    return FIELD_TITLES.get(key, field["label"])


def iter_models(grouped):
    return sorted(grouped.items(), key=lambda item: (MODEL_ORDER.get(item[0], 100), item[0]))


def plot_error(field, grouped, metric, stat, out_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    for model, values in iter_models(grouped):
        leads = [lead for lead, _ in values]
        errors = [error for _, error in values]
        ax.plot(leads, errors, marker="o", linewidth=2, label=model_label(model))

    ax.set_title(f"{field['label']} 2022 aggregate {stat} ({metric})")
    ax.set_xlabel("lead time (h)")
    ax.set_ylabel("weighted error")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_improvement(field, grouped, metric, stat, reference_model, out_path):
    import matplotlib.pyplot as plt

    if reference_model not in grouped:
        return False
    ref = dict(grouped[reference_model])

    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    any_line = False
    for model, values in iter_models(grouped):
        if model == reference_model:
            continue
        leads = []
        improvements = []
        for lead, error in values:
            ref_error = ref.get(lead)
            if ref_error is None or not np.isfinite(ref_error) or ref_error == 0:
                continue
            leads.append(lead)
            improvements.append(100.0 * (ref_error - error) / ref_error)
        if leads:
            ax.plot(leads, improvements, marker="o", linewidth=2, label=model_label(model))
            any_line = True

    if not any_line:
        plt.close(fig)
        return False

    ax.axhline(0.0, color="black", linewidth=1, linestyle=":")
    ax.set_title(f"{field['label']} improvement vs {model_label(reference_model)}")
    ax.set_xlabel("lead time (h)")
    ax.set_ylabel(f"{metric} improvement (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


def plot_overview(fields, rows, metric, stat, lead_hours, out_path, title):
    import matplotlib.pyplot as plt

    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 6.2), constrained_layout=True)
    flat_axes = axes.ravel()
    legend_handles = []
    legend_labels = []
    any_panel = False

    for index, field in enumerate(fields[: ncols * nrows - 1]):
        ax = flat_axes[index]
        selected = select_rows(rows, field, stat, metric, lead_hours)
        if not selected:
            ax.set_visible(False)
            print(f"Skipping overview panel {field['spec']}: no matching rows")
            continue
        grouped = group_series(selected, metric)
        for model, values in iter_models(grouped):
            leads_days = [lead / 24.0 for lead, _ in values]
            errors = [error for _, error in values]
            (line,) = ax.plot(
                leads_days,
                errors,
                marker="o",
                linewidth=1.5,
                markersize=3.0,
                label=model_label(model),
            )
            if model_label(model) not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(model_label(model))
        ax.set_title(field_title(field), fontsize=11)
        ax.set_xlabel("Lead time (days)")
        ax.set_ylabel(f"Weighted {stat} error")
        ax.grid(True, alpha=0.25)
        any_panel = True

    for ax in flat_axes[len(fields[: ncols * nrows - 1]) :]:
        ax.set_visible(False)

    legend_ax = flat_axes[-1]
    legend_ax.set_visible(True)
    legend_ax.axis("off")
    if legend_handles:
        legend_ax.legend(legend_handles, legend_labels, loc="center", frameon=True)

    fig.suptitle(title, fontsize=14)
    if any_panel:
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        return True
    plt.close(fig)
    return False


def write_rank_table(field, grouped, metric, out_path):
    leads = sorted({lead for values in grouped.values() for lead, _ in values})
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["field", "lead_hours", "rank", "model", metric])
        for lead in leads:
            values = []
            for model, series in grouped.items():
                value_by_lead = dict(series)
                if lead in value_by_lead:
                    values.append((value_by_lead[lead], model))
            for rank, (value, model) in enumerate(sorted(values), start=1):
                writer.writerow([field["label"], lead, rank, model, value])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--fields", nargs="+", type=parse_field, default=[parse_field(x) for x in DEFAULT_FIELDS])
    parser.add_argument("--stat", default="std")
    parser.add_argument("--metric", choices=["mean", "median"], default="mean")
    parser.add_argument("--reference-model", default="amse5000")
    parser.add_argument("--lead-hours", type=float, nargs="*", default=None)
    parser.add_argument(
        "--overview-output",
        type=Path,
        default=None,
        help="Optional multi-panel deterministic error overview PNG.",
    )
    parser.add_argument(
        "--overview-title",
        default="Full-year 2022 deterministic error summaries (lower is better)",
        help="Title for --overview-output.",
    )
    args = parser.parse_args()

    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise SystemExit("Install matplotlib first: python3 -m pip install matplotlib==3.8.3") from exc
    matplotlib.use("Agg", force=True)

    out_dir = args.out_dir or args.summary_csv.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_summary(args.summary_csv)
    if not rows:
        raise SystemExit(f"No usable rows found in {args.summary_csv}")

    written = []
    for field in args.fields:
        selected = select_rows(rows, field, args.stat, args.metric, args.lead_hours)
        if not selected:
            print(f"Skipping {field['spec']}: no matching rows")
            continue
        grouped = group_series(selected, args.metric)
        stem = safe_name(f"{field['label']}_{args.stat}_{args.metric}")

        error_path = out_dir / f"{stem}_error_by_lead.png"
        plot_error(field, grouped, args.metric, args.stat, error_path)
        written.append(error_path)

        improvement_path = out_dir / f"{stem}_improvement_vs_{safe_name(args.reference_model)}.png"
        if plot_improvement(field, grouped, args.metric, args.stat, args.reference_model, improvement_path):
            written.append(improvement_path)

        rank_path = out_dir / f"{stem}_rankings.csv"
        write_rank_table(field, grouped, args.metric, rank_path)
        written.append(rank_path)

    if args.overview_output is not None:
        args.overview_output.parent.mkdir(parents=True, exist_ok=True)
        if plot_overview(
            args.fields,
            rows,
            args.metric,
            args.stat,
            args.lead_hours,
            args.overview_output,
            args.overview_title,
        ):
            written.append(args.overview_output)

    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
