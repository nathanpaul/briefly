"""Briefly — meeting capture → transcript → per-person summary → Claude vault enrichment.

See PLAN.md and docs/ for architecture; knowledge/ for hardware/test facts and decisions.
The pipeline is file-based and stage-encapsulated: each stage reads files and writes files.
"""

__version__ = "0.1.0"
