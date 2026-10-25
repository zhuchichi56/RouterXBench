#!/usr/bin/env python3
"""Render strict 5,955 or paper-reference tables as a 3-page preview.

Usage:
  python3 render_preview.py --scope strict
  python3 render_preview.py --scope paper_reference

The PDF is vector text; each page is also written as an editable-text SVG and
a 300 dpi PNG.  All text is placed with explicit boxes and checked for overflow.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import struct
import textwrap
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
MPLCONFIG = HERE / ".mplconfig"
MPLCONFIG.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

PAGE_W, PAGE_H = 8.5, 11.0
LEFT = 0.65
TABLE_W = 7.20
RIGHT = LEFT + TABLE_W
METHOD_W = 2.05
NUM_W = (TABLE_W - METHOD_W) / 6.0
DATASETS = ("alpaca_5k_test", "mmlu_test", "big_math_5k_test", "magpie_5k_test", "mmlu_pro", "math")
INK = "#111111"
MUTED = "#444444"
PENDING = "#555555"
WARN = "#7A1F1F"
STRICT_REPEAT_NOTE = (
    "Repeat identity: SelfAsk/Entropy/MaxLogits/Conf. Margin = 3 fixed executions; "
    "Emb. MLP/HybridLLM/RouteLLM/MeanProbe = 3 training seeds; Jev = 3 actual API executions. "
    "A reported 0.00 is a measured sample SD for identical fixed executions, not a missing value."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_font() -> str:
    for family in ("STIXGeneral", "STIX Two Text", "DejaVu Serif"):
        try:
            findfont(FontProperties(family=family), fallback_to_default=False)
            return family
        except ValueError:
            continue
    raise RuntimeError("no serif font available")


def col_left(index: int) -> float:
    return LEFT + METHOD_W + index * NUM_W


def col_right(index: int) -> float:
    return col_left(index) + NUM_W


class FitChecker:
    def __init__(self, fig):
        self.fig = fig
        self.items = []

    def add(self, obj, x0, x1, y0, y1, label):
        self.items.append((obj, x0, x1, y0, y1, label))

    def run(self):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        failures = []
        for obj, x0, x1, y0, y1, label in self.items:
            bbox = obj.get_window_extent(renderer=renderer)
            left = bbox.x0 / self.fig.dpi
            right = bbox.x1 / self.fig.dpi
            top = PAGE_H - bbox.y1 / self.fig.dpi
            bottom = PAGE_H - bbox.y0 / self.fig.dpi
            if left < x0 - 0.012 or right > x1 + 0.012:
                failures.append(
                    {
                        "label": label,
                        "text": obj.get_text(),
                        "horizontal": [round(left, 3), round(right, 3)],
                        "allowed": [round(x0, 3), round(x1, 3)],
                    }
                )
            if top < y0 - 0.012 or bottom > y1 + 0.012:
                failures.append(
                    {
                        "label": label,
                        "text": obj.get_text(),
                        "vertical": [round(top, 3), round(bottom, 3)],
                        "allowed": [round(y0, 3), round(y1, 3)],
                    }
                )
        return failures


def load_scope(scope: str):
    cfg = {
        "strict": {
            "csv": "main_tables.csv",
            "prefix": "main_tables",
            "kicker": "RouterXBench strict 5,955 main table",
        },
        "paper_reference": {
            "csv": "paper_reference.csv",
            "prefix": "paper_reference",
            "kicker": "Unchanged current paper table",
        },
    }[scope]
    with (HERE / cfg["csv"]).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    provenance = json.loads((HERE / "provenance.json").read_text(encoding="utf-8"))
    display = provenance["display"][scope]
    expected_methods = 21 if scope == "strict" else 9
    metrics = []
    for metric_index, metric_key in enumerate(("auroc", "bq", "tc")):
        methods = []
        by_name = {}
        for row in rows:
            if row["metric"] != metric_key:
                continue
            name = row["method"]
            if name not in by_name:
                method = {
                    "method": name,
                    "group": row["group"],
                    "separator_before": row.get("separator_before") == "true",
                    "repeat_identity": row.get("repeat_identity", ""),
                    "values": {},
                }
                by_name[name] = method
                methods.append(method)
            by_name[name]["values"][row["dataset"]] = row
        if len(methods) != expected_methods:
            raise RuntimeError(f"{scope}/{metric_key}: expected {expected_methods} methods, found {len(methods)}")
        for method in methods:
            if len(method["values"]) != 6:
                raise RuntimeError(f"{scope}/{metric_key}/{method['method']}: expected 6 datasets")
        metrics.append(
            {
                "key": metric_key,
                "number": metric_index + 1,
                "title": display["table_titles"][metric_index],
                "caption": display["captions"][metric_index],
                "methods": methods,
            }
        )
    return cfg, provenance, display, metrics


def draw_page(scope, page_number, metric, provenance, display, font):
    strict = scope == "strict"
    methods = metric["methods"]
    paper = not strict
    row_h = 0.245 if strict else 0.300
    cell_fs = 7.2 if strict else 7.8
    header_fs = 7.5 if strict else 7.9
    group_fs = 6.9 if strict else 7.2
    top_rule = 1.02 if strict else 1.20
    table_title_y = 0.56
    fig = plt.figure(figsize=(PAGE_W, PAGE_H), dpi=300)
    fig.patch.set_facecolor("white")
    checker = FitChecker(fig)
    snapshot = provenance["build"]["generated_at_utc"][:10]
    complete = provenance["build"]["complete_conditions"]
    expected = provenance["build"]["expected_conditions"]

    def text(
        x,
        y,
        value,
        size,
        *,
        weight="normal",
        style="normal",
        color=INK,
        ha="left",
        va="center",
        box=None,
        label="text",
    ):
        obj = fig.text(
            x / PAGE_W,
            1.0 - y / PAGE_H,
            value,
            fontsize=size,
            fontweight=weight,
            fontstyle=style,
            color=color,
            ha=ha,
            va=va,
            fontfamily=font,
        )
        if box is not None:
            checker.add(obj, box[0], box[1], box[2], box[3], label)
        return obj

    def hline(y, x0=LEFT, x1=RIGHT, lw=0.8, color=INK):
        fig.add_artist(
            Line2D(
                [x0 / PAGE_W, x1 / PAGE_W],
                [1.0 - y / PAGE_H, 1.0 - y / PAGE_H],
                transform=fig.transFigure,
                color=color,
                linewidth=lw,
                solid_capstyle="butt",
            )
        )

    kicker = (
        f"{'RouterXBench strict 5,955 main table' if strict else 'Unchanged current paper table'}"
        f"  |  {complete}/{expected} StartLux conditions complete  |  {snapshot}"
        if strict
        else f"{'Unchanged current paper table'}  |  protocol-mismatched reference  |  {snapshot}"
    )
    text(LEFT, 0.36, kicker, 6.8, color=MUTED, va="top", box=(0.40, PAGE_W - 0.40, 0.20, 0.52), label="kicker")
    text(
        LEFT,
        table_title_y,
        f"Table {metric['number']}. {metric['title']}",
        10.5,
        weight="bold",
        va="top",
        box=(0.40, PAGE_W - 0.40, 0.44, 0.84),
        label="title",
    )
    if paper:
        text(
            (LEFT + RIGHT) / 2.0,
            0.98,
            "PROTOCOL-MISMATCHED REFERENCE — NOT THE STRICT MAIN TABLE",
            7.4,
            weight="bold",
            color=WARN,
            ha="center",
            box=(LEFT, RIGHT, 0.86, 1.10),
            label="banner",
        )
    hline(top_rule, lw=1.1)

    if metric["key"] == "auroc":
        same_center = (col_left(0) + col_right(2)) / 2.0
        diff_center = (col_left(3) + col_right(5)) / 2.0
        text(same_center, top_rule + 0.12, "Same-source", 7.2, ha="center", box=(col_left(0), col_right(2), top_rule, top_rule + 0.25), label="panel-same")
        text(diff_center, top_rule + 0.12, "Different-source", 7.2, ha="center", box=(col_left(3), col_right(5), top_rule, top_rule + 0.25), label="panel-diff")
        hline(top_rule + 0.27, col_left(0) + 0.02, col_right(2) - 0.02, lw=0.5)
        hline(top_rule + 0.27, col_left(3) + 0.02, col_right(5) - 0.02, lw=0.5)
    else:
        panel = "Budget-limited quality: BQ ↑" if metric["key"] == "bq" else "Quality-first expenditure: TC ↓"
        text(
            (LEFT + METHOD_W + RIGHT) / 2.0,
            top_rule + 0.12,
            panel,
            7.4,
            ha="center",
            box=(LEFT + METHOD_W, RIGHT, top_rule, top_rule + 0.26),
            label="panel",
        )

    header_y = top_rule + 0.38
    text(LEFT, header_y, "Method", header_fs, weight="bold", box=(LEFT, LEFT + METHOD_W - 0.03, top_rule + 0.27, top_rule + 0.53), label="header-method")
    for index, label in enumerate(provenance["display"]["dataset_labels"]):
        text(
            col_right(index) - 0.035,
            header_y,
            label,
            header_fs,
            weight="bold",
            ha="right",
            box=(col_left(index), col_right(index), top_rule + 0.27, top_rule + 0.53),
            label=f"header-{label}",
        )
    hline(top_rule + 0.53, lw=0.6)

    y = top_rule + 0.58
    wrote_baseline_header = False
    wrote_startlux_header = False
    for method in methods:
        if method["group"] in ("strict_5955_baseline", "paper_reference") and not wrote_baseline_header:
            y += 0.05
            label = display["baseline_group_label"]
            text(
                LEFT + 0.02,
                y + 0.08,
                label,
                group_fs,
                style="italic",
                color=MUTED,
                box=(LEFT, RIGHT, y, y + 0.22),
                label="group-baseline",
            )
            y += 0.27 if strict else 0.32
            wrote_baseline_header = True
        if strict and method["group"] == "startlux_single_run" and not wrote_startlux_header:
            y += 0.04
            text(
                LEFT + 0.02,
                y + 0.08,
                display["startlux_group_label"],
                group_fs,
                style="italic",
                color=MUTED,
                box=(LEFT, RIGHT, y, y + 0.22),
                label="group-startlux",
            )
            y += 0.27
            wrote_startlux_header = True
        if method["separator_before"]:
            hline(y + 0.015, lw=0.5)
            y += 0.09
        text(
            LEFT,
            y + row_h / 2.0,
            method["method"],
            cell_fs,
            box=(LEFT, LEFT + METHOD_W - 0.035, y, y + row_h - 0.015),
            label=f"method-{method['method']}",
        )
        for index, dataset in enumerate(DATASETS):
            row = method["values"][dataset]
            if row["status"] == "pending":
                value = "pending"
                is_pending = True
            elif method["group"] == "paper_reference":
                value = f"{row['value_pct']} ± {row['sd_pct']}" if row["sd_pct"] else row["value_pct"]
                is_pending = False
            elif method["group"] == "strict_5955_baseline":
                value = f"{row['value_pct']} ± {row['sd_pct']}" if row["sd_pct"] else row["value_pct"]
                is_pending = False
            else:
                value = row["value_pct"]
                is_pending = False
            text(
                col_right(index) - 0.035,
                y + row_h / 2.0,
                value,
                cell_fs,
                ha="right",
                style="italic" if is_pending else "normal",
                color=PENDING if is_pending else INK,
                box=(col_left(index), col_right(index), y, y + row_h - 0.015),
                label=f"cell-{method['method']}-{dataset}",
            )
        y += row_h

    table_bottom = y + 0.02
    hline(table_bottom, lw=1.1)
    if table_bottom > 7.85:
        raise RuntimeError(f"{scope}/{metric['key']}: table body too tall ({table_bottom:.2f} in)")

    caption_y = table_bottom + 0.20
    for line in textwrap.wrap(metric["caption"], width=108):
        text(
            LEFT,
            caption_y,
            line,
            7.2,
            va="top",
            box=(0.40, PAGE_W - 0.40, caption_y - 0.02, caption_y + 0.18),
            label="caption",
        )
        caption_y += 0.148

    note_y = caption_y + 0.10
    if strict:
        notes = [
            "Baseline: 9 previous methods from the strict 5,955 sources; BQ/TC normalized by the actual 5,955 Qcloud/full_cloud_cost.",
            STRICT_REPEAT_NOTE,
            "StartLux: one exploratory execution per condition; no SD. pending = condition not complete, not zero.",
            "The paper reference is separate and protocol-mismatched (PAPER_REFERENCE.md); do not combine the two tables or rank them together.",
        ]
    else:
        notes = [
            "Values copied exactly from the current paper tables (table1_auroc.tex, table2_bq.tex, table3_tc.tex); no cell was changed.",
            "Protocol: 5,954 subset; historical greedy + sampled 42/43 rounds; changed local answers.",
            "Not strictly matched to the 5,955 StartLux protocol. Do not compare directly with MAIN_TABLES.md or combine ranks.",
        ]
    for note in notes:
        for line in textwrap.wrap(note, width=124):
            text(
                LEFT,
                note_y,
                line,
                6.6,
                color=MUTED,
                va="top",
                box=(0.40, PAGE_W - 0.40, note_y - 0.02, note_y + 0.16),
                label="note",
            )
            note_y += 0.118

    if paper:
        box_y = max(note_y + 0.25, 6.55)
        box_h = 1.32
        if box_y + box_h > 10.30:
            raise RuntimeError(f"{scope}: protocol box does not fit ({box_y:.2f})")
        fig.add_artist(
            Rectangle(
                (LEFT / PAGE_W, 1.0 - (box_y + box_h) / PAGE_H),
                TABLE_W / PAGE_W,
                box_h / PAGE_H,
                transform=fig.transFigure,
                fill=False,
                edgecolor="#777777",
                linewidth=0.6,
            )
        )
        text(LEFT + 0.14, box_y + 0.12, "Protocol difference (why this is a separate reference)", 7.2, weight="bold", box=(LEFT + 0.10, RIGHT - 0.10, box_y + 0.02, box_y + 0.24), label="box-title")
        box_notes = [
            "Subset is 5,954 requests, not the strict 5,955 clean split.",
            "Historical greedy + sampled 42/43 rounds; local answers changed.",
            "Exact paper values only. Do not compare directly with the strict table or combine ranks.",
        ]
        by = box_y + 0.38
        for note in box_notes:
            for line in textwrap.wrap(note, width=112):
                text(LEFT + 0.14, by, line, 6.8, color=MUTED, va="top", box=(LEFT + 0.10, RIGHT - 0.10, by - 0.02, by + 0.16), label="box-note")
                by += 0.122

    if note_y > 10.45:
        raise RuntimeError(f"{scope}/{metric['key']}: notes run too low ({note_y:.2f} in)")

    text(
        PAGE_W / 2.0,
        10.64,
        f"StartLux + RouterXBench  |  {snapshot}  |  {'strict 5,955' if strict else 'paper reference'}  |  page {page_number}/3",
        6.4,
        color=MUTED,
        ha="center",
        box=(0.40, PAGE_W - 0.40, 10.50, 10.80),
        label="footer",
    )

    failures = checker.run()
    if failures:
        raise RuntimeError(f"{scope}/{metric['key']}: {len(failures)} text containment failure(s): {json.dumps(failures[:5], indent=1)}")
    return fig


def png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()[:24]
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"not a PNG: {path}")
    return struct.unpack(">II", data[16:24])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("strict", "paper_reference"), default="strict")
    args = parser.parse_args()
    scope = args.scope
    cfg, provenance, display, metrics = load_scope(scope)
    font = choose_font()
    plt.rcParams.update(
        {
            "font.family": font,
            "font.size": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
            "savefig.dpi": 300,
        }
    )
    out = HERE / "preview"
    out.mkdir(exist_ok=True)
    prefix = cfg["prefix"]
    pdf_path = out / f"{prefix}_preview.pdf"
    artifacts = {}
    with PdfPages(
        pdf_path,
        metadata={
            "Title": "StartLux + RouterXBench tables",
            "Subject": "strict 5,955 main tables" if scope == "strict" else "protocol-mismatched paper reference",
            "Creator": "render_preview.py (matplotlib vector text)",
        },
    ) as pdf:
        for page_number, metric in enumerate(metrics, start=1):
            fig = draw_page(scope, page_number, metric, provenance, display, font)
            svg_path = out / f"{prefix}_{metric['key']}.svg"
            png_path = out / f"{prefix}_{metric['key']}.png"
            fig.savefig(svg_path, format="svg", bbox_inches=None, facecolor="white")
            fig.savefig(png_path, format="png", dpi=300, bbox_inches=None, facecolor="white")
            pdf.savefig(fig, bbox_inches=None)
            plt.close(fig)
            width, height = png_size(png_path)
            artifacts[metric["key"]] = {
                "pdf_page": page_number,
                "svg": {
                    "path": f"preview/{svg_path.name}",
                    "sha256": sha256_file(svg_path),
                    "text_elements": svg_path.read_text(encoding="utf-8").count("<text"),
                },
                "png": {
                    "path": f"preview/{png_path.name}",
                    "sha256": sha256_file(png_path),
                    "width_px": width,
                    "height_px": height,
                    "dpi": 300,
                },
            }
    manifest = {
        "schema": "routerxbench-startlux-preview/2",
        "scope": scope,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "font": font,
        "page_size_in": [PAGE_W, PAGE_H],
        "table_width_in": TABLE_W,
        "layout_checks": "passed (all fitted text inside declared boxes; no overflow)",
        "source": {
            cfg["csv"]: {"path": cfg["csv"], "sha256": sha256_file(HERE / cfg["csv"])},
            "provenance.json": {"path": "provenance.json", "sha256": sha256_file(HERE / "provenance.json")},
        },
        "status": {
            "complete_conditions": provenance["build"]["complete_conditions"],
            "expected_conditions": provenance["build"]["expected_conditions"],
            "pending_conditions": provenance["build"]["pending_conditions"],
        },
        "pdf": {"path": f"preview/{pdf_path.name}", "pages": 3, "sha256": sha256_file(pdf_path)},
        "artifacts": artifacts,
    }
    (out / f"{prefix}_preview_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {pdf_path} (3 pages), 3 SVG, 3 PNG at 300 dpi; scope={scope}; font={font}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
