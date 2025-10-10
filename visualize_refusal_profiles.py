"""Create publication-quality refusal timeline visualization."""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import textwrap

DATASETS = [
    ("Direct Elicitation", Path("results/direct_elicitation.json")),
    ("Progressive Extraction", Path("results/progressive_extraction.json")),
    ("Multi-turn Jailbreak", Path("results/multi_turn_jailbreak.json")),
]

REFUSAL_COLOR = "#e74c3c"
COMPLY_COLOR = "#27ae60"
DIFFER_COLOR = "#2c3e50"


def resolve_dataset(path: Path) -> Path:
    corrected = path.with_name(path.stem + "_corrected" + path.suffix)
    return corrected if corrected.exists() else path


def load_conversation(path: Path, index: int = 0) -> dict:
    data = json.loads(resolve_dataset(path).read_text())
    try:
        return data[index]
    except (IndexError, TypeError) as exc:
        raise ValueError(f"{path} missing conversation at index {index}") from exc


def get_difference_indices(baseline: List[dict], suppressed: List[dict]) -> set:
    """Find turn indices where baseline and suppressed differ."""
    diff_indices = set()
    max_turns = max(len(baseline), len(suppressed))

    for idx in range(max_turns):
        b_refuse = baseline[idx]["looks_like_refusal"] if idx < len(baseline) else None
        s_refuse = (
            suppressed[idx]["looks_like_refusal"] if idx < len(suppressed) else None
        )
        if b_refuse != s_refuse:
            diff_indices.add(idx)

    return diff_indices


def count_refusals(turns: List[dict]) -> int:
    return sum(1 for t in turns if t["looks_like_refusal"])


def create_timeline_trace(
    turns: List[dict], row_y: float, diff_indices: set, name: str
) -> Tuple[go.Scatter, List[go.layout.Shape]]:
    """Create scatter trace and rectangle shapes for a timeline row."""

    # Create boxes as shapes
    shapes = []
    for idx, turn in enumerate(turns):
        color = REFUSAL_COLOR if turn["looks_like_refusal"] else COMPLY_COLOR
        border_width = 3 if idx in diff_indices else 1
        border_color = DIFFER_COLOR if idx in diff_indices else "#bdc3c7"

        shapes.append(
            dict(
                type="rect",
                x0=idx + 0.1,
                x1=idx + 0.9,
                y0=row_y - 0.35,
                y1=row_y + 0.35,
                fillcolor=color,
                line=dict(color=border_color, width=border_width),
                layer="below",
            )
        )

    # Create invisible trace for hover info
    hover_texts = []
    x_positions = []
    y_positions = []

    for idx, turn in enumerate(turns):
        status = "REFUSES" if turn["looks_like_refusal"] else "COMPLIES"
        snippet = (
            turn["response"][:100] + "..."
            if len(turn["response"]) > 100
            else turn["response"]
        )
        hover_texts.append(
            f"<b>Turn {idx + 1} - {name}</b><br>{status}<br><br>{snippet}"
        )
        x_positions.append(idx + 0.5)
        y_positions.append(row_y)

    trace = go.Scatter(
        x=x_positions,
        y=y_positions,
        mode="markers",
        marker=dict(size=0.1, color="rgba(0,0,0,0)"),
        hovertext=hover_texts,
        hoverinfo="text",
        name=name,
        showlegend=False,
    )

    return trace, shapes


def format_example(turn_idx: int, baseline: List[dict], suppressed: List[dict]) -> str:
    """Format a comparison example as HTML."""
    idx = turn_idx
    b_turn = baseline[idx] if idx < len(baseline) else None
    s_turn = suppressed[idx] if idx < len(suppressed) else None

    b_status = "refuses" if (b_turn and b_turn["looks_like_refusal"]) else "complies"
    s_status = "refuses" if (s_turn and s_turn["looks_like_refusal"]) else "complies"

    b_text = (
        b_turn["response"][:80] + "..."
        if b_turn and len(b_turn["response"]) > 80
        else (b_turn["response"] if b_turn else "N/A")
    )
    s_text = (
        s_turn["response"][:80] + "..."
        if s_turn and len(s_turn["response"]) > 80
        else (s_turn["response"] if s_turn else "N/A")
    )

    return (
        f"<b>Turn {turn_idx + 1}:</b> Baseline {b_status}, Suppressed {s_status}<br>"
        f"→ B: {b_text}<br>"
        f"→ S: {s_text}"
    )


def create_figure(example_index: int = 0):
    """Create the complete figure."""

    fig = make_subplots(
        rows=len(DATASETS),
        cols=1,
        subplot_titles=[name for name, _ in DATASETS],
        vertical_spacing=0.15,
        specs=[[{"type": "scatter"}] for _ in DATASETS],
    )

    all_shapes = []
    all_annotations = []

    for dataset_idx, (title, path) in enumerate(DATASETS):
        row = dataset_idx + 1

        # Load data
        conversation = load_conversation(path, example_index)
        modes = {m["mode"]: m for m in conversation["modes"]}
        baseline_turns = modes["baseline"]["turns"]
        suppressed_turns = modes["suppressed"]["turns"]

        # Get differences
        diff_indices = get_difference_indices(baseline_turns, suppressed_turns)
        max_turns = max(len(baseline_turns), len(suppressed_turns))

        # Create traces
        baseline_trace, baseline_shapes = create_timeline_trace(
            baseline_turns, 1, diff_indices, "Baseline"
        )
        suppressed_trace, suppressed_shapes = create_timeline_trace(
            suppressed_turns, 0, diff_indices, "Suppressed"
        )

        # Adjust shapes for subplot
        for shape in baseline_shapes + suppressed_shapes:
            shape["xref"] = f"x{row}"
            shape["yref"] = f"y{row}"
            all_shapes.append(shape)

        # Add traces
        fig.add_trace(baseline_trace, row=row, col=1)
        fig.add_trace(suppressed_trace, row=row, col=1)

        # Add row labels
        all_annotations.extend(
            [
                dict(
                    x=-0.5,
                    y=1,
                    xref=f"x{row}",
                    yref=f"y{row}",
                    text="<b>Baseline</b>",
                    showarrow=False,
                    xanchor="right",
                    font=dict(size=11),
                ),
                dict(
                    x=-0.5,
                    y=0,
                    xref=f"x{row}",
                    yref=f"y{row}",
                    text="<b>Suppressed</b>",
                    showarrow=False,
                    xanchor="right",
                    font=dict(size=11),
                ),
            ]
        )

        # Statistics summary
        b_refusals = count_refusals(baseline_turns)
        s_refusals = count_refusals(suppressed_turns)
        delta = s_refusals - b_refusals

        summary = (
            f"<b>Baseline:</b> {b_refusals}/{len(baseline_turns)} refusals  "
            f"<b>Suppressed:</b> {s_refusals}/{len(suppressed_turns)} refusals  "
            f"<b>Δ:</b> {delta:+d}"
        )

        all_annotations.append(
            dict(
                x=max_turns / 2,
                y=-1.2,
                xref=f"x{row}",
                yref=f"y{row}",
                text=summary,
                showarrow=False,
                font=dict(size=10),
                xanchor="center",
            )
        )

        # Key differences
        if diff_indices:
            examples = [
                format_example(i, baseline_turns, suppressed_turns)
                for i in sorted(diff_indices)[:2]
            ]
            diff_text = "<br><br>".join(examples)
        else:
            diff_text = "No behavioral differences observed"

        all_annotations.append(
            dict(
                x=max_turns / 2,
                y=-2.0,
                xref=f"x{row}",
                yref=f"y{row}",
                text=diff_text,
                showarrow=False,
                font=dict(size=9, family="monospace"),
                xanchor="center",
                align="left",
            )
        )

        # Update axes
        fig.update_xaxes(
            title_text="Turn Index",
            range=[-1, max_turns],
            tickmode="linear",
            tick0=1,
            dtick=1,
            row=row,
            col=1,
        )
        fig.update_yaxes(range=[-2.8, 1.8], showticklabels=False, row=row, col=1)

    # Update layout
    fig.update_layout(
        title=dict(
            text="<b>Refusal Timeline: Baseline vs Suppressed Taboo Behavior</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=18),
        ),
        shapes=all_shapes,
        annotations=all_annotations,
        height=350 * len(DATASETS),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.8)",
        ),
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # Add legend items manually
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=12, color=REFUSAL_COLOR, symbol="square"),
            name="Looks like refusal",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=12, color=COMPLY_COLOR, symbol="square"),
            name="No refusal",
        )
    )

    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize refusal behavior across probes."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/refusal_profiles.html"),
        help="Where to save the figure (HTML or PNG)",
    )
    parser.add_argument(
        "--example-index", type=int, default=0, help="Conversation index to visualize"
    )
    parser.add_argument(
        "--format", choices=["html", "png"], default="html", help="Output format"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    fig = create_figure(args.example_index)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "html" or args.output.suffix == ".html":
        fig.write_html(str(args.output))
        print(f"Saved interactive HTML to {args.output}")
    else:
        # Requires kaleido: pip install kaleido
        fig.write_image(
            str(args.output), width=1400, height=350 * len(DATASETS), scale=2
        )
        print(f"Saved static image to {args.output}")


if __name__ == "__main__":
    main()
