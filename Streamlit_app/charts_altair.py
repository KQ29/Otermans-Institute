# charts_altair.py
import pandas as pd
import altair as alt

def meter_chart(value: float, max_value: float, title: str, unit: str = "", fmt: str = ".1f") -> alt.Chart:
    max_value = float(max(1.0, max_value, value))
    df = pd.DataFrame([{"name": title, "value": float(value)}])
    bar = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("value:Q", title=None, scale=alt.Scale(domain=[0, max_value])),
            y=alt.Y("name:N", title=None, axis=None),
            tooltip=[alt.Tooltip("value:Q", format=fmt)],
        )
        .properties(height=60, title=f"{title}: {value:{fmt}}{unit}")
    )
    text = bar.mark_text(align="left", dx=3, dy=0).encode(text=alt.Text("value:Q", format=fmt))
    return bar + text
