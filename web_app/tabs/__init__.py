"""
tabs package
Các tab chức năng của SOC NIDS Live Console.
"""

from .single_flow import apply_sample_to_state, render_single_flow_tab
from .ab_showdown import render_ab_showdown_tab
from .batch_scanner import render_batch_scanner_tab
from .registry_findings import render_registry_findings_tab

__all__ = [
    "apply_sample_to_state",
    "render_single_flow_tab",
    "render_ab_showdown_tab",
    "render_batch_scanner_tab",
    "render_registry_findings_tab",
]
