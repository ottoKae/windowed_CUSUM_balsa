"""Compatibility entry for stp.py; new callers should use stp.py."""
from pathlib import Path

_source = Path(__file__).with_name("stp.py")
__file__ = str(_source)
exec(compile(_source.read_bytes(), str(_source), "exec"), globals())
