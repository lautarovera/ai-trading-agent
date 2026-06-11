"""Watch-only Streamlit dashboard for the paper trading agent.

Usage:
    uv run streamlit run dashboard.py
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trading_agent.broker import make_broker
from trading_agent.config import load_config
from trading_agent.data import MarketData

st.set_page_config(page_title="AI Trading Agent", page_icon="📈", layout="wide")

cfg = load_config()
broker = make_broker(cfg)
data = MarketData(cache_dir=cfg["data"]["cache_dir"],
                  cache_ttl_minutes=cfg["data"]["cache_ttl_minutes"],
                  period=cfg["data"]["history_period"],
                  interval=cfg["data"]["interval"])

st.title("📈 AI Trading Agent — Paper Portfolio")
st.caption("Watch-only dashboard. Paper trading — no real money. "
           "Refresh the page for the latest state.")


@st.cache_data(ttl=120)
def get_prices(symbols: tuple[str, ...]) -> dict[str, float]:
    return data.prices(list(symbols))


prices = get_prices(tuple(cfg["universe"]["symbols"]))
positions = broker.positions()
equity = broker.equity(prices)
starting = cfg["portfolio"]["starting_cash"]

# -- wallet summary ----------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total equity", f"${equity:,.2f}",
          f"{(equity / starting - 1) * 100:+.2f}% all-time")
c2.metric("Cash", f"${broker.cash:,.2f}")
c3.metric("Invested", f"${equity - broker.cash:,.2f}")
c4.metric("Open positions", len(positions))

# -- positions ----------------------------------------------------------------
st.subheader("Holdings")
if positions:
    rows = []
    for sym, pos in positions.items():
        price = prices.get(sym, pos["avg_price"])
        value = pos["qty"] * price
        pnl = (price / pos["avg_price"] - 1) * 100
        rows.append({"Symbol": sym, "Shares": pos["qty"],
                     "Avg cost": round(pos["avg_price"], 2),
                     "Last price": round(price, 2),
                     "Market value": round(value, 2),
                     "P&L %": round(pnl, 2),
                     "Weight %": round(value / equity * 100, 1)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info("No open positions yet — the agent will buy when signals cross the threshold.")

# -- equity curve ---------------------------------------------------------------
st.subheader("Equity curve")
history = broker.equity_history()
if history:
    df = pd.DataFrame(history)
    df["ts"] = pd.to_datetime(df["ts"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["ts"], y=df["equity"], name="Equity",
                             mode="lines", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=df["ts"], y=df["cash"], name="Cash",
                             mode="lines", line=dict(width=1, dash="dot")))
    fig.add_hline(y=starting, line_dash="dash", line_color="gray",
                  annotation_text="Starting cash")
    fig.update_layout(height=350, margin=dict(t=10, b=10),
                      yaxis_title="USD", legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No equity history yet — run the agent: `uv run python run_agent.py --once`")

# -- trades & signals ------------------------------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Recent trades")
    trades = broker.trades(limit=50)
    if trades:
        tdf = pd.DataFrame(trades)[["ts", "symbol", "side", "qty", "price", "rationale"]]
        tdf["ts"] = pd.to_datetime(tdf["ts"]).dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(tdf, use_container_width=True, hide_index=True, height=400)
    else:
        st.info("No trades yet.")

with right:
    st.subheader("Latest signals")
    sigs = broker.recent_signals(limit=100)
    if sigs:
        sdf = pd.DataFrame(sigs)[["ts", "symbol", "strategy", "score",
                                  "confidence", "rationale"]]
        sdf["ts"] = pd.to_datetime(sdf["ts"]).dt.strftime("%Y-%m-%d %H:%M")
        sdf["score"] = sdf["score"].round(2)
        sdf["confidence"] = sdf["confidence"].round(2)
        st.dataframe(sdf, use_container_width=True, hide_index=True, height=400)
    else:
        st.info("No signals logged yet.")
