"""Run the paper trading agent loop.

Usage:
    uv run python run_agent.py            # continuous loop (cycle_minutes from config)
    uv run python run_agent.py --once     # single evaluation cycle
"""

from __future__ import annotations

import argparse
import logging
import time

from trading_agent.agent import TradingAgent

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("run_agent")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI paper-trading agent")
    parser.add_argument("--once", action="store_true", help="run one cycle and exit")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args()

    agent = TradingAgent(args.config)
    log.info("Paper trading agent started — universe: %s", ", ".join(agent.symbols))

    while True:
        try:
            agent.run_cycle()
        except Exception:
            log.exception("Cycle failed; will retry next interval")
        if args.once:
            break
        minutes = agent.cfg["agent"]["cycle_minutes"]
        log.info("Sleeping %d minutes until next cycle", minutes)
        time.sleep(minutes * 60)


if __name__ == "__main__":
    main()
