"""Small helper to load a local `.env` file into environment variables.

It first tries to use python-dotenv if available, otherwise falls back to a simple parser.
Importing this module will attempt to load `.env` in the repo root.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOTENV = ROOT / '.env'

def _simple_load(path: Path):
    try:
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' not in line:
                    continue
                key, val = line.split('=', 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and val and key not in os.environ:
                    os.environ[key] = val
    except FileNotFoundError:
        return

def load_dotenv():
    if DOTENV.exists():
        try:
            # prefer python-dotenv if available
            from dotenv import load_dotenv as _pd_load
            _pd_load(str(DOTENV))
        except Exception:
            _simple_load(DOTENV)

# load on import for convenience
load_dotenv()
