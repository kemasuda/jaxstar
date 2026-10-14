"""Expose the single frozen test reference to source-checkout benchmarks."""

from pathlib import Path
import runpy


_reference = runpy.run_path(str(
    Path(__file__).resolve().parents[1] / "tests/unittests/_mistgrid_reference.py"
))
MistGridIso = _reference["MistGridIso"]
NamedMistGridIso = _reference["NamedMistGridIso"]
