#!/usr/bin/env python3
"""Refusal timeline: baseline vs suppressed (portable Matplotlib version)."""

import argparse, json, textwrap
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any, Optional, Union

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

# Colors / style
REFUSAL_COLORS = {True: "#e0585b", False: "#2d9c72"}  # refusal, not refusal
EDGE_DIFF = "#3e4a5b"
EDGE_NORMAL = "#e6e6ef"
TEXT_COLOR = "#1f1f1f"
INFO_BG = "#f7f7fc"
INFO_EDGE = "#dfe1f1"


def best_path(*candidates: Union[str, Path]) -> Path:
    """Return the first existing path from candidates."""
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    raise FileNotFoundError(
        "None of these paths exist:\n  " + "\n  ".join(str(c) for c in candidates)
    )


def resolve_path(arg: Optional[Path], *fallbacks: str) -> Path:
    """Prefer the CLI-provided path but fall back to known defaults."""
    candidates: List[Path] = []
    if arg is not None:
        candidates.append(arg)
    candidates.extend(Path(f) for f in fallbacks)
    return best_path(*candidates)


def load_modes(
    path: Path, example_index: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list of conversations.")
    try:
        conv = data[example_index]
    except IndexError as exc:
        raise ValueError(
            f"{path} has no conversation at index {example_index}."
        ) from exc
    modes = {m["mode"]: m for m in conv["modes"]}
    return modes["baseline"]["turns"], modes["suppressed"]["turns"]


def refusal_count(turns: Iterable[Dict[str, Any]]) -> int:
    return sum(1 for t in turns if bool(t.get("looks_like_refusal")))


def difference_indices(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> List[int]:
    n = max(len(a), len(b))
    idxs = []
    for i in range(n):
        af = a[i]["looks_like_refusal"] if i < len(a) else None
        bf = b[i]["looks_like_refusal"] if i < len(b) else None
        if af != bf:
            idxs.append(i)
    return idxs


def wrap(s: str, width: int = 58, max_lines: int = 3) -> str:
    lines = textwrap.wrap(" ".join(s.split()), width=width)
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + ["… " + lines[-1]]
    return "\n".join(lines)


def build_summary(
    baseline: List[Dict[str, Any]], suppressed: List[Dict[str, Any]]
) -> str:
    b = refusal_count(baseline)
    s = refusal_count(suppressed)
    return (
        f"Baseline refusals: {b}/{len(baseline)}\n"
        f"Suppressed refusals: {s}/{len(suppressed)}\n"
        f"Δ (suppressed — baseline): {s - b:+d}"
    )


def build_highlights(
    baseline: List[Dict[str, Any]], suppressed: List[Dict[str, Any]], k: int = 2
) -> str:
    idxs = difference_indices(baseline, suppressed)
    if not idxs:
        return (
            "No refusal changes observed.\nBoth runs follow the same trajectory here."
        )
    out = []
    for i in idxs[:k]:
        bi = i < len(baseline)
        si = i < len(suppressed)
        bflag = baseline[i]["looks_like_refusal"] if bi else False
        sflag = suppressed[i]["looks_like_refusal"] if si else False
        out.append(
            f"- Turn {i + 1}: baseline {'refuses' if bflag else 'complies'}, "
            f"suppressed {'refuses' if sflag else 'complies'}\n"
            f"  Baseline → {wrap(baseline[i]['response'] if bi else '(no baseline turn)')}\n"
            f"  Suppressed → {wrap(suppressed[i]['response'] if si else '(no suppressed turn)')}"
        )
    return "\n\n".join(out)


def draw_timeline(ax, baseline: List[Dict[str, Any]], suppressed: List[Dict[str, Any]]):
    rows = [("Baseline", baseline, 1), ("Suppressed", suppressed, 0)]
    total = max(len(baseline), len(suppressed))
    diffs = set(difference_indices(baseline, suppressed))

    for label, turns, y in rows:
        for i, t in enumerate(turns):
            rect = Rectangle(
                (i, y - 0.35),
                1.0,
                0.7,
                facecolor=REFUSAL_COLORS[bool(t["looks_like_refusal"])],
                edgecolor=EDGE_DIFF if i in diffs else EDGE_NORMAL,
                linewidth=1.3 if i in diffs else 0.9,
                joinstyle="round",
            )
            ax.add_patch(rect)
        ax.text(
            -0.45,
            y,
            label,
            ha="right",
            va="center",
            fontsize=10,
            color=TEXT_COLOR,
            clip_on=False,
        )

    ax.set_xlim(-0.75, total + 0.2)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([])
    ax.set_xticks([i + 0.5 for i in range(total)])
    ax.set_xticklabels([str(i + 1) for i in range(total)], fontsize=9)
    ax.tick_params(axis="x", bottom=True, top=False, length=0)
    ax.set_xlabel("Turn index", labelpad=6)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def add_info_box(ax, summary: str, highlights: str):
    ax.axis("off")
    ax.set_facecolor(INFO_BG)
    ax.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            facecolor=INFO_BG,
            edgecolor=INFO_EDGE,
            linewidth=0.8,
        )
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    title = (
        "Turn highlights" if "No refusal changes" not in highlights else "Commentary"
    )

    ax.text(
        0.06,
        0.92,
        summary,
        ha="left",
        va="top",
        fontsize=10,
        color=TEXT_COLOR,
        transform=ax.transAxes,
    )
    ax.text(
        0.06,
        0.64,
        title,
        ha="left",
        va="top",
        fontsize=10,
        color="#38384a",
        fontweight="semibold",
        transform=ax.transAxes,
    )
    ax.text(
        0.06,
        0.59,
        highlights,
        ha="left",
        va="top",
        fontsize=9,
        color=TEXT_COLOR,
        transform=ax.transAxes,
    )


def make_figure(paths: Dict[str, Path], out: Path, dpi: int, example_index: int):
    # Conservative rcParams (no version-specific keys)
    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#d9d9e3",
            "text.color": TEXT_COLOR,
            "axes.labelcolor": "#4b4b59",
            "xtick.color": "#4b4b59",
            "ytick.color": "#4b4b59",
            "font.size": 10,
        }
    )

    titles = ["Direct Elicitation", "Progressive Extraction", "Multi-turn Jailbreak"]
    fig = plt.figure(figsize=(13.6, 8.5))
    grid = fig.add_gridspec(3, 2, width_ratios=[2.5, 1.7], hspace=0.75, wspace=0.48)

    legend_handles = [
        Patch(
            facecolor=REFUSAL_COLORS[True],
            edgecolor="#c8c8d5",
            label="Looks like refusal",
        ),
        Patch(facecolor=REFUSAL_COLORS[False], edgecolor="#c8c8d5", label="No refusal"),
    ]

    for row, key in enumerate(["direct", "progressive", "jailbreak"]):
        baseline, suppressed = load_modes(paths[key], example_index)

        ax_tl = fig.add_subplot(grid[row, 0])
        ax_tl.set_title(titles[row], fontsize=13, loc="left", pad=6, color=TEXT_COLOR)
        draw_timeline(ax_tl, baseline, suppressed)

        ax_info = fig.add_subplot(grid[row, 1])
        add_info_box(
            ax_info,
            build_summary(baseline, suppressed),
            build_highlights(baseline, suppressed),
        )

    fig.suptitle(
        "Refusal timeline: baseline vs suppressed taboo behavior",
        fontsize=18,
        y=0.99,
        color=TEXT_COLOR,
        fontweight="semibold",
    )
    fig.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=10)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure to {out}")


def parse_args():
    p = argparse.ArgumentParser(description="Visualize refusal behavior across probes.")
    p.add_argument(
        "--direct", type=Path, default=None, help="Path to direct elicitation JSON"
    )
    p.add_argument(
        "--progressive",
        type=Path,
        default=None,
        help="Path to progressive extraction JSON",
    )
    p.add_argument(
        "--jailbreak", type=Path, default=None, help="Path to multi-turn jailbreak JSON"
    )
    p.add_argument("--example-index", type=int, default=0)
    p.add_argument("--dpi", type=int, default=240)
    p.add_argument(
        "--output", type=Path, default=Path("results/figures/refusal_profiles.png")
    )
    return p.parse_args()


def main():
    a = parse_args()
    paths = {
        "direct": resolve_path(
            a.direct,
            "results/direct_elicitation.json",
            "direct_elicitation_corrected.json",
            "results/direct_elicitation_corrected.json",
        ),
        "progressive": resolve_path(
            a.progressive,
            "results/progressive_extraction.json",
            "progressive_extraction.json",
            "results/progressive_extraction_corrected.json",
        ),
        "jailbreak": resolve_path(
            a.jailbreak,
            "results/multi_turn_jailbreak.json",
            "multi_turn_jailbreak_corrected.json",
            "results/multi_turn_jailbreak_corrected.json",
        ),
    }
    make_figure(paths, a.output, a.dpi, a.example_index)


if __name__ == "__main__":
    main()
