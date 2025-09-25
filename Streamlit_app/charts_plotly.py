# charts_plotly.py
import plotly.graph_objects as go

# Shared color palette
PALETTE = [
    "#2E86AB",  # blue
    "#6AA84F",  # green
    "#FF8C42",  # orange
    "#B35C9E",  # purple
    "#E5ECF6",  # light gray (for "Remaining")
]


def pie_for_score(score: float):
    """Full pie chart for activity score (0–100)."""
    done = max(0, min(100, float(score)))
    values = [done, 100 - done]
    labels = ["Score", "Remaining"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0,  # full circle
                sort=False,
                marker=dict(colors=[PALETTE[0], PALETTE[4]]),
                textinfo="label+percent",
                hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        template="plotly_white",
        height=280,
        margin=dict(l=10, r=10, t=40, b=10),
        title=f"Average activity score: {done:.1f}%",
        showlegend=False,
    )
    return fig


def pie_for_completed(done: int, total: int, title: str = "Completed"):
    """Full pie chart for completed vs total lessons/topics."""
    total = max(1, int(total))
    done = max(0, min(int(done), total))
    values = [done, total - done]
    labels = ["Done", "Remaining"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0,
                sort=False,
                marker=dict(colors=[PALETTE[1], PALETTE[4]]),
                textinfo="label+percent",
                hovertemplate="%{label}: %{value} of " + str(total) + "<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        template="plotly_white",
        height=280,
        margin=dict(l=10, r=10, t=40, b=10),
        title=f"{title} ({done}/{total})",
        showlegend=False,
    )
    return fig


def alltime_kpi_bars(labels, values, units):
    """Vertical bar chart for 4 KPIs (points, avg session, progress, hints)."""
    colors = [PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3]]
    texts = [f"{v:.1f}{u}" if u else f"{v:.1f}" for v, u in zip(values, units)]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=colors,
                text=texts,
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{x}: %{y:.1f}%{customdata}<extra></extra>",
                customdata=units,
            )
        ]
    )
    fig.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=10, r=10, t=40, b=40),
        title="All-time KPIs",
        xaxis=dict(tickangle=-10, categoryorder="array", categoryarray=labels),
        yaxis=dict(showgrid=True, zeroline=True, rangemode="tozero"),
        uniformtext_minsize=10,
        uniformtext_mode="show",
        showlegend=False,
    )
    return fig


def bar_mastery_subjects(data):
    """
    Bar chart of accuracy/mastery per subject.

    Accepts either:
      - dict {subject: accuracy_percent}
      - list of {"subject": ..., "avg": ...}  # we’ll accept this too
    """
    if isinstance(data, list):
        subjects = [d["subject"] for d in data]
        values = [float(d.get("avg") or d.get("accuracy") or 0.0) for d in data]
    else:
        subjects = list(data.keys())
        values = [data[s] for s in subjects]

    fig = go.Figure(
        data=[
            go.Bar(
                x=subjects,
                y=values,
                marker_color=PALETTE[2],
                text=[f"{v:.1f}%" for v in values],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        template="plotly_white",
        height=400,
        margin=dict(l=10, r=10, t=40, b=40),
        title="Accuracy & Mastery by subject",
        xaxis=dict(title="Subject"),
        yaxis=dict(title="Accuracy (%)", rangemode="tozero"),
        uniformtext_minsize=10,
        uniformtext_mode="show",
    )
    return fig


def bar_response_time_subjects(data: dict, unit_label: str = " mins"):
    """
    Bar chart of response/processing time per subject.
    data: {subject: avg_time_value}
    """
    subjects = list(data.keys())
    values = [data[s] for s in subjects]

    fig = go.Figure(
        data=[
            go.Bar(
                x=subjects,
                y=values,
                marker_color=PALETTE[3],
                text=[f"{v:.1f}{unit_label}" for v in values],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{x}: %{y:.1f}" + unit_label + "<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        template="plotly_white",
        height=400,
        margin=dict(l=10, r=10, t=40, b=40),
        title="Response time by subject",
        xaxis=dict(title="Subject"),
        yaxis=dict(title=f"Response Time {unit_label}", rangemode="tozero"),
        uniformtext_minsize=10,
        uniformtext_mode="show",
    )
    return fig
