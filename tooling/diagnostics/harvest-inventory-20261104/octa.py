"""Compatibility entry for tda.py; new callers should use tda.py."""
from pathlib import Path

_source = Path(__file__).with_name("tda.py")
__file__ = str(_source)
exec(compile(_source.read_bytes(), str(_source), "exec"), globals())
