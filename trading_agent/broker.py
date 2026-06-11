"""Paper broker: simulated cash + positions persisted in SQLite.

No real money is involved anywhere in this module. All fills are simulated
at the provided market price.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from .config import resolve_path

SCHEMA = """
CREATE TABLE IF NOT EXISTS account (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    cash REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS positions (
    symbol TEXT PRIMARY KEY,
    qty REAL NOT NULL,
    avg_price REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    qty REAL NOT NULL,
    price REAL NOT NULL,
    rationale TEXT
);
CREATE TABLE IF NOT EXISTS equity_history (
    ts TEXT PRIMARY KEY,
    equity REAL NOT NULL,
    cash REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS signals_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    strategy TEXT NOT NULL,
    score REAL NOT NULL,
    confidence REAL NOT NULL,
    rationale TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PaperBroker:
    def __init__(self, db_path: str = "data/paper_broker.db",
                 starting_cash: float = 100_000.0):
        path = resolve_path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        cur = self.conn.execute("SELECT cash FROM account WHERE id = 1")
        if cur.fetchone() is None:
            self.conn.execute("INSERT INTO account (id, cash) VALUES (1, ?)",
                              (starting_cash,))
            self.conn.commit()

    # -- state -------------------------------------------------------------
    @property
    def cash(self) -> float:
        return float(self.conn.execute(
            "SELECT cash FROM account WHERE id = 1").fetchone()["cash"])

    def positions(self) -> dict[str, dict]:
        rows = self.conn.execute(
            "SELECT symbol, qty, avg_price FROM positions WHERE qty > 0").fetchall()
        return {r["symbol"]: {"qty": r["qty"], "avg_price": r["avg_price"]} for r in rows}

    def equity(self, prices: dict[str, float]) -> float:
        total = self.cash
        for sym, pos in self.positions().items():
            total += pos["qty"] * prices.get(sym, pos["avg_price"])
        return total

    # -- orders (simulated fills) -------------------------------------------
    def buy(self, symbol: str, qty: float, price: float, rationale: str = "") -> bool:
        cost = qty * price
        if qty <= 0 or cost > self.cash:
            return False
        pos = self.positions().get(symbol, {"qty": 0.0, "avg_price": 0.0})
        new_qty = pos["qty"] + qty
        new_avg = (pos["qty"] * pos["avg_price"] + cost) / new_qty
        with self.conn:
            self.conn.execute("UPDATE account SET cash = cash - ? WHERE id = 1", (cost,))
            self.conn.execute(
                "INSERT INTO positions (symbol, qty, avg_price) VALUES (?, ?, ?) "
                "ON CONFLICT(symbol) DO UPDATE SET qty = ?, avg_price = ?",
                (symbol, new_qty, new_avg, new_qty, new_avg))
            self.conn.execute(
                "INSERT INTO trades (ts, symbol, side, qty, price, rationale) "
                "VALUES (?, ?, 'BUY', ?, ?, ?)", (_now(), symbol, qty, price, rationale))
        return True

    def sell(self, symbol: str, qty: float, price: float, rationale: str = "") -> bool:
        pos = self.positions().get(symbol)
        if pos is None or qty <= 0 or qty > pos["qty"] + 1e-9:
            return False
        qty = min(qty, pos["qty"])
        with self.conn:
            self.conn.execute("UPDATE account SET cash = cash + ? WHERE id = 1",
                              (qty * price,))
            remaining = pos["qty"] - qty
            if remaining < 1e-9:
                self.conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
            else:
                self.conn.execute("UPDATE positions SET qty = ? WHERE symbol = ?",
                                  (remaining, symbol))
            self.conn.execute(
                "INSERT INTO trades (ts, symbol, side, qty, price, rationale) "
                "VALUES (?, ?, 'SELL', ?, ?, ?)", (_now(), symbol, qty, price, rationale))
        return True

    # -- logging -------------------------------------------------------------
    def snapshot_equity(self, prices: dict[str, float]) -> float:
        eq = self.equity(prices)
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO equity_history (ts, equity, cash) VALUES (?, ?, ?)",
                (_now(), eq, self.cash))
        return eq

    def log_signal(self, sig) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO signals_log (ts, symbol, strategy, score, confidence, rationale) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (_now(), sig.symbol, sig.strategy, sig.score, sig.confidence, sig.rationale))

    def trades(self, limit: int = 100) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def equity_history(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT ts, equity, cash FROM equity_history ORDER BY ts").fetchall()
        return [dict(r) for r in rows]

    def recent_signals(self, limit: int = 200) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM signals_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


def make_broker(cfg: dict) -> PaperBroker:
    """Build the configured broker: 'local' (SQLite sim) or 'alpaca' (paper API)."""
    p = cfg["portfolio"]
    kind = p.get("broker", "local")
    if kind == "alpaca":
        from .alpaca_broker import AlpacaPaperBroker
        return AlpacaPaperBroker(db_path=p.get("alpaca_log_db", "data/alpaca_log.db"),
                                 max_order_value=p.get("max_order_value", 20_000.0))
    return PaperBroker(db_path=p["db_path"], starting_cash=p["starting_cash"])
