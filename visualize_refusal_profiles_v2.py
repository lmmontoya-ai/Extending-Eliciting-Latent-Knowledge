"""Create publication-quality refusal timeline visualization with improved layout."""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict

import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATASETS = [
    ("Direct Elicitation", Path("results/direct_elicitation.json")),
    ("Progressive Extraction", Path("results/progressive_extraction.json")),
    ("Multi-turn Jailbreak", Path("results/multi_turn_jailbreak.json")),
]

REFUSAL_COLOR = "#e74c3c"
COMPLY_COLOR = "#27ae60"
DIFFER_COLOR = "#f39c12"
GRAY = "#95a5a6"


def resolve_dataset(path: Path) -> Path:
    """Check for _corrected version of dataset."""
    corrected = path.with_name(path.stem + "_corrected" + path.suffix)
    return corrected if corrected.exists() else path


def load_conversation(path: Path, index: int = 0) -> dict:
    """Load a specific conversation from dataset."""
    data = json.loads(resolve_dataset(path).read_text())
    try:
        return data[index]
    except (IndexError, TypeError) as exc:
        raise ValueError(f"{path} missing conversation at index {index}") from exc


def get_difference_indices(baseline: List[dict], suppressed: List[dict]) -> set:
    """Find turn indices where baseline and suppressed differ in refusal behavior."""
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


def truncate_text(text: str, max_length: int = 150) -> str:
    """Truncate text to max length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def create_timeline_trace(
    turns: List[dict], row_y: float, diff_indices: set, name: str
) -> Tuple[go.Scatter, List[dict]]:
    """Create scatter trace and rectangle shapes for a timeline row."""

    shapes = []
    hover_texts = []
    x_positions = []
    y_positions = []

    for idx, turn in enumerate(turns):
        is_refusal = turn["looks_like_refusal"]
        is_different = idx in diff_indices

        # Choose colors
        fill_color = REFUSAL_COLOR if is_refusal else COMPLY_COLOR
        border_color = DIFFER_COLOR if is_different else GRAY
        border_width = 3 if is_different else 1

        # Create box shape
        shapes.append(
            dict(
                type="rect",
                x0=idx + 0.05,
                x1=idx + 0.95,
                y0=row_y - 0.4,
                y1=row_y + 0.4,
                fillcolor=fill_color,
                line=dict(color=border_color, width=border_width),
                layer="below",
            )
        )

        # Create hover info
        status = "🚫 REFUSES" if is_refusal else "✅ COMPLIES"
        diff_marker = " 🔄 DIFFERS" if is_different else ""
        snippet = truncate_text(turn["response"], 200)

        hover_text = (
            f"<b>Turn {idx + 1} - {name}{diff_marker}</b><br>"
            f"{status}<br><br>"
            f"<i>{snippet}</i>"
        )

        hover_texts.append(hover_text)
        x_positions.append(idx + 0.5)
        y_positions.append(row_y)

    # Invisible trace for hover
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


def create_summary_text(
    baseline: List[dict], suppressed: List[dict], diff_indices: set
) -> str:
    """Create a concise summary of refusal statistics."""
    b_refusals = sum(1 for t in baseline if t["looks_like_refusal"])
    s_refusals = sum(1 for t in suppressed if t["looks_like_refusal"])
    delta = s_refusals - b_refusals

    delta_str = f"+{delta}" if delta > 0 else str(delta)
    arrow = "⬆️" if delta > 0 else "⬇️" if delta < 0 else "➡️"

    return (
        f"<b>Baseline:</b> {b_refusals}/{len(baseline)} refusals  |  "
        f"<b>Suppressed:</b> {s_refusals}/{len(suppressed)} refusals  |  "
        f"<b>Change:</b> {delta_str} {arrow}  |  "
        f"<b>Differs at:</b> {len(diff_indices)} turns"
    )


def create_example_text(
    turn_idx: int, baseline: List[dict], suppressed: List[dict]
) -> str:
    """Format a single comparison example."""
    b_turn = baseline[turn_idx] if turn_idx < len(baseline) else None
    s_turn = suppressed[turn_idx] if turn_idx < len(suppressed) else None

    if not b_turn or not s_turn:
        return ""

    b_status = "refuses" if b_turn["looks_like_refusal"] else "complies"
    s_status = "refuses" if s_turn["looks_like_refusal"] else "complies"

    b_text = truncate_text(b_turn["response"], 100)
    s_text = truncate_text(s_turn["response"], 100)

    return (
        f"<b>Turn {turn_idx + 1}:</b> B={b_status}, S={s_status}<br>"
        f"• <i>Baseline:</i> {b_text}<br>"
        f"• <i>Suppressed:</i> {s_text}"
    )


def get_interesting_turns(
    baseline: List[dict], suppressed: List[dict], diff_indices: set
) -> List[int]:
    """Get interesting turn indices to show as examples.

    Prioritizes:
    1. Turns where behavior differs (diff_indices)
    2. Turns where responses are substantially different even if behavior is same
    """
    if diff_indices:
        return sorted(diff_indices)[:2]

    # If no behavioral differences, find turns with most different responses
    interesting = []
    for idx in range(min(len(baseline), len(suppressed))):
        b_resp = baseline[idx]["response"]
        s_resp = suppressed[idx]["response"]

        # Check if responses are different enough to be interesting
        if len(b_resp) != len(s_resp) or b_resp != s_resp:
            interesting.append(idx)

    # Return first 2 interesting turns (preferring later turns which are more adversarial)
    if len(interesting) >= 2:
        return interesting[-2:]  # Last 2 turns
    return interesting[:2]


def create_figure(example_index: int = 0):
    """Create the complete refusal timeline figure."""

    # Calculate height based on number of datasets
    row_height = 500  # pixels per dataset
    fig_height = row_height * len(DATASETS)

    fig = make_subplots(
        rows=len(DATASETS),
        cols=1,
        subplot_titles=[f"<b>{name}</b>" for name, _ in DATASETS],
        vertical_spacing=0.12,
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

        # Create timeline traces
        baseline_trace, baseline_shapes = create_timeline_trace(
            baseline_turns, 1.0, diff_indices, "Baseline"
        )
        suppressed_trace, suppressed_shapes = create_timeline_trace(
            suppressed_turns, 0.0, diff_indices, "Suppressed"
        )

        # Adjust shapes for subplot
        for shape in baseline_shapes + suppressed_shapes:
            shape["xref"] = f"x{row}"
            shape["yref"] = f"y{row}"
            all_shapes.append(shape)

        # Add traces
        fig.add_trace(baseline_trace, row=row, col=1)
        fig.add_trace(suppressed_trace, row=row, col=1)

        # Row labels (left side)
        all_annotations.extend(
            [
                dict(
                    x=-0.2,
                    y=1.0,
                    xref=f"x{row}",
                    yref=f"y{row}",
                    text="<b>Baseline</b>",
                    showarrow=False,
                    xanchor="right",
                    font=dict(size=12, color="#2c3e50"),
                ),
                dict(
                    x=-0.2,
                    y=0.0,
                    xref=f"x{row}",
                    yref=f"y{row}",
                    text="<b>Suppressed</b>",
                    showarrow=False,
                    xanchor="right",
                    font=dict(size=12, color="#2c3e50"),
                ),
            ]
        )

        # Summary statistics (above timeline)
        summary = create_summary_text(baseline_turns, suppressed_turns, diff_indices)
        all_annotations.append(
            dict(
                x=max_turns / 2,
                y=1.8,
                xref=f"x{row}",
                yref=f"y{row}",
                text=summary,
                showarrow=False,
                font=dict(size=11),
                xanchor="center",
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor=GRAY,
                borderwidth=1,
                borderpad=4,
            )
        )

        # Key differences (below timeline)
        interesting_turns = get_interesting_turns(
            baseline_turns, suppressed_turns, diff_indices
        )

        if interesting_turns:
            # Show up to 2 most interesting turns
            examples = []
            for i in interesting_turns:
                example_text = create_example_text(i, baseline_turns, suppressed_turns)
                if example_text:
                    examples.append(example_text)

            diff_text = (
                "<br><br>".join(examples) if examples else "No examples available"
            )
        else:
            diff_text = "✓ Responses are identical"

        all_annotations.append(
            dict(
                x=max_turns / 2,
                y=-2.2,
                xref=f"x{row}",
                yref=f"y{row}",
                text=diff_text,
                showarrow=False,
                font=dict(size=9, family="Arial"),
                xanchor="center",
                align="left",
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor=GRAY,
                borderwidth=1,
                borderpad=6,
            )
        )

        # Update axes for this subplot
        fig.update_xaxes(
            title_text="<b>Turn Number</b>" if row == len(DATASETS) else "",
            range=[-0.5, max_turns - 0.5],
            tickmode="array",
            tickvals=list(range(max_turns)),
            ticktext=[str(i) for i in range(max_turns)],
            tickfont=dict(size=10),
            gridcolor="#ecf0f1",
            showgrid=True,
            row=row,
            col=1,
        )
        fig.update_yaxes(
            range=[-3.5, 2.5],
            showticklabels=False,
            showgrid=False,
            row=row,
            col=1,
        )

    # Overall layout
    fig.update_layout(
        title=dict(
            text=(
                "<b>Refusal Behavior Comparison: Baseline vs Suppressed Taboo Direction</b><br>"
                "<sup>Analyzing secret word elicitation across different probing strategies</sup>"
            ),
            x=0.5,
            xanchor="center",
            font=dict(size=20),
        ),
        shapes=all_shapes,
        annotations=all_annotations,
        height=fig_height,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.005,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor=GRAY,
            borderwidth=1,
        ),
        hovermode="closest",
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        margin=dict(l=120, r=50, t=120, b=80),
    )

    # Legend items
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=14, color=REFUSAL_COLOR, symbol="square"),
            name="🚫 Refusal",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=14, color=COMPLY_COLOR, symbol="square"),
            name="✅ Compliance",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                size=14,
                color="white",
                symbol="square",
                line=dict(color=DIFFER_COLOR, width=3),
            ),
            name="🔄 Differs",
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
        default=Path("results/figures/refusal_profiles_v2.html"),
        help="Where to save the figure (HTML or PNG)",
    )
    parser.add_argument(
        "--example-index", type=int, default=0, help="Conversation index to visualize"
    )
    parser.add_argument(
        "--format", choices=["html", "png"], default="html", help="Output format"
    )
    parser.add_argument(
        "--width", type=int, default=1600, help="Figure width for PNG export"
    )
    parser.add_argument("--dpi", type=int, default=2, help="DPI scale for PNG export")
    return parser.parse_args()


def main():
    args = parse_args()
    fig = create_figure(args.example_index)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "html" or args.output.suffix == ".html":
        fig.write_html(str(args.output))
        print(f"✅ Saved interactive HTML to {args.output}")
    else:
        # Requires kaleido: pip install kaleido
        height = 500 * len(DATASETS)
        fig.write_image(
            str(args.output), width=args.width, height=height, scale=args.dpi
        )
        print(f"✅ Saved static image to {args.output}")


if __name__ == "__main__":
    main()
