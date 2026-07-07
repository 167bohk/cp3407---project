import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def apply_plot_theme(fig, theme, height=None):
    layout = {
        "template": theme["plotly_template"],
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": theme["text_color"]},
    }
    if height is not None:
        layout["height"] = height

    fig.update_layout(**layout)
    fig.update_xaxes(tickfont={"color": theme["text_color"]}, gridcolor=theme["grid_color"])
    fig.update_yaxes(tickfont={"color": theme["text_color"]}, gridcolor=theme["grid_color"])
    return fig


def build_sentiment_gauge(sentiment, theme):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=sentiment,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Market Sentiment", "font": {"color": theme["text_color"]}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": theme["text_color"],
                    "tickfont": {"color": theme["text_color"]},
                },
                "bar": {"color": "#3b82f6"},
                "steps": [
                    {"range": [0, 40], "color": "#ef4444"},
                    {"range": [40, 60], "color": "#facc15"},
                    {"range": [60, 100], "color": "#22c55e"},
                ],
            },
        )
    )
    fig.update_layout(
        autosize=False,
        width=760,
        margin={"l": 40, "r": 40, "t": 60, "b": 20},
    )
    apply_plot_theme(fig, theme, height=420)
    fig.update_traces(number={"font": {"color": theme["text_color"]}})
    return fig


def build_price_chart(df, theme):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MA20"],
            line=dict(color="#60a5fa", width=2),
            name="MA20",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume",
            marker_color="rgba(120,160,255,0.3)",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        hovermode="x unified",
        dragmode="pan",
        legend=dict(
            bgcolor=theme["card_bg"],
            bordercolor=theme["grid_color"],
            borderwidth=1,
            font={"color": theme["text_color"]},
        ),
        xaxis=dict(rangeslider=dict(visible=True), type="date"),
    )
    return apply_plot_theme(fig, theme, height=650)


def build_heatmap_chart(heatmap_df, theme):
    fig = px.bar(
        heatmap_df,
        x="Ticker",
        y="Change",
        color="Change",
        text="Change",
        color_continuous_scale="RdYlGn",
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    return apply_plot_theme(fig, theme, height=450)
