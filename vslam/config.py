import tomllib
from pathlib import Path

# every key in configs/default.toml becomes a module-level name, e.g. `from .config import MAX_LOST`
with open(Path(__file__).parent / "configs" / "default.toml", "rb") as f:
    globals().update(tomllib.load(f))
