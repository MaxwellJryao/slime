#!/usr/bin/env python3
"""CLI entry point for the shared W&B terminal-state reconciler."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slime.utils.wandb_terminal_state import main


if __name__ == "__main__":
    raise SystemExit(main())
