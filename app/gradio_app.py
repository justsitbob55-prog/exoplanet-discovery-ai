"""Professional Gradio interface for Exoplanet Discovery AI."""

from __future__ import annotations

import hashlib
import html
import json
import logging
import math
import os
import threading
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from matplotlib.figure import Figure

from src.exoplanet_discovery.common.logging import configure_logging, get_logger
from src.exoplanet_discovery.common.models import LightCurve
from src.exoplanet_discovery.common.persistence import ResultStore
from src.exoplanet_discovery.detection.transit import detect_transits
from src.exoplanet_discovery.features.extractor import extract_features
from src.exoplanet_discovery.ingestion.archive import (
    archive_cache_dir,
    create_demo_light_curve,
    download_light_curve,
    search_targets,
)
from src.exoplanet_discovery.ingestion.loader import load_light_curve
from src.exoplanet_discovery.preprocessing.pipeline import preprocess_light_curve
from src.exoplanet_discovery.preprocessing.sampling import downsample_light_curve

configure_logging(level=logging.INFO)
logger = get_logger("exoplanet_discovery.gradio_app")
store = ResultStore()
MAX_PLOT_CADENCES = 10_000
_GRADIO_PROGRESS = gr.Progress(track_tqdm=True)
_analysis_cache: OrderedDict[
    bytes, tuple[LightCurve, LightCurve, dict[str, float]]
] = OrderedDict()
_analysis_cache_lock = threading.Lock()

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.cyan,
    secondary_hue=gr.themes.colors.violet,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "Segoe UI", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "Consolas", "monospace"],
).set(
    body_background_fill="#060b18",
    body_text_color="#e6f1ff",
    block_background_fill="rgba(10, 20, 40, 0.58)",
    block_border_color="rgba(103, 232, 249, 0.18)",
    block_label_text_color="#9fb6d1",
    block_title_text_color="#e6f1ff",
    input_background_fill="rgba(7, 17, 32, 0.75)",
    input_border_color="rgba(103, 232, 249, 0.22)",
    button_primary_background_fill="linear-gradient(135deg, #0891b2, #2563eb, #7c3aed)",
    button_primary_background_fill_hover="linear-gradient(135deg, #06b6d4, #3b82f6, #8b5cf6)",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="rgba(15, 30, 55, 0.75)",
    button_secondary_text_color="#d7e7ff",
    table_even_background_fill="rgba(10, 22, 42, 0.55)",
    table_odd_background_fill="rgba(13, 27, 50, 0.45)",
    border_color_primary="rgba(103, 232, 249, 0.18)",
)

CUSTOM_CSS = """
:root {
  --exo-bg: #060b18;
  --exo-bg-2: #081225;
  --exo-surface: rgba(10, 20, 40, 0.58);
  --exo-surface-strong: rgba(12, 24, 46, 0.78);
  --exo-border: rgba(103, 232, 249, 0.18);
  --exo-border-strong: rgba(103, 232, 249, 0.38);
  --exo-cyan: #22d3ee;
  --exo-blue: #3b82f6;
  --exo-violet: #8b5cf6;
  --exo-text: #e6f1ff;
  --exo-muted: #9fb6d1;
  --exo-success: #34d399;
  --exo-warn: #fbbf24;
  --exo-shadow: 0 20px 60px rgba(2, 8, 23, 0.55);
  --exo-radius: 18px;
}
html, body, gradio-app, .gradio-container, .gradio-container.gradio-container, #exo-app-root {
  background-color: var(--exo-bg) !important;
}
body, gradio-app, .gradio-container {
  background:
    radial-gradient(circle at 15% 10%, rgba(34,211,238,0.12), transparent 28%),
    radial-gradient(circle at 85% 15%, rgba(139,92,246,0.16), transparent 30%),
    radial-gradient(circle at 50% 90%, rgba(59,130,246,0.12), transparent 35%),
    linear-gradient(180deg, #030712 0%, #060b18 45%, #081225 100%) !important;
  color: var(--exo-text) !important;
  min-height: 100vh;
}
.gradio-container {
  position: relative;
  overflow-x: hidden;
  max-width: 1280px !important;
  margin: 0 auto !important;
  padding: 20px 18px 40px !important;
}
.gradio-container::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  background-image:
    radial-gradient(1px 1px at 10% 20%, rgba(255,255,255,0.60), transparent),
    radial-gradient(1px 1px at 30% 80%, rgba(255,255,255,0.38), transparent),
    radial-gradient(1.5px 1.5px at 70% 30%, rgba(103,232,249,0.50), transparent),
    radial-gradient(1px 1px at 90% 70%, rgba(255,255,255,0.42), transparent),
    radial-gradient(1px 1px at 55% 15%, rgba(167,139,250,0.55), transparent),
    radial-gradient(1px 1px at 40% 45%, rgba(255,255,255,0.28), transparent),
    radial-gradient(1px 1px at 18% 58%, rgba(103,232,249,0.30), transparent),
    radial-gradient(1px 1px at 82% 42%, rgba(139,92,246,0.35), transparent);
  opacity: 0.6;
  animation: exo-drift 18s ease-in-out infinite alternate;
  z-index: 0;
}
.gradio-container > * { position: relative; z-index: 1; }
@keyframes exo-drift { from { transform: translateY(0px); } to { transform: translateY(-10px); } }
@keyframes exo-glow { 0%,100% { box-shadow: 0 0 0 rgba(34,211,238,0.0); } 50% { box-shadow: 0 0 24px rgba(34,211,238,0.12); } }

.exo-hero {
  padding: 28px 30px;
  border: 1px solid var(--exo-border-strong);
  border-radius: 24px;
  background:
    linear-gradient(135deg, rgba(8,145,178,0.22), rgba(37,99,235,0.14), rgba(124,58,237,0.18)),
    rgba(7,17,32,0.72);
  box-shadow: var(--exo-shadow);
  backdrop-filter: blur(14px);
  margin-bottom: 14px;
  overflow: hidden;
  position: relative;
}
.exo-hero::after {
  content: "";
  position: absolute;
  right: -80px; top: -80px;
  width: 260px; height: 260px;
  background: radial-gradient(circle, rgba(34,211,238,0.22), transparent 65%);
  filter: blur(8px);
}
.exo-eyebrow {
  font-size: 12px;
  letter-spacing: .18em;
  text-transform: uppercase;
  color: var(--exo-cyan);
  font-weight: 700;
  margin-bottom: 8px;
}
.exo-hero h1 { margin: 0 0 8px 0; font-size: clamp(1.8rem, 3vw, 2.6rem); line-height: 1.1; }
.exo-hero p { margin: 0; color: var(--exo-muted); max-width: 900px; font-size: 1.02rem; }
.exo-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 14px;
  margin: 10px 0 16px 0;
}
.exo-card {
  border: 1px solid var(--exo-border);
  border-radius: var(--exo-radius);
  padding: 16px 18px;
  background: linear-gradient(180deg, rgba(13,27,50,0.72), rgba(8,18,36,0.62));
  box-shadow: 0 10px 30px rgba(2,8,23,0.35);
  backdrop-filter: blur(12px);
  transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
}
.exo-card:hover {
  transform: translateY(-2px);
  border-color: var(--exo-border-strong);
  box-shadow: 0 14px 36px rgba(34,211,238,0.10);
}
.exo-card .label { color: var(--exo-muted); font-size: .85rem; margin-bottom: 6px; }
.exo-card .value { font-size: 1.55rem; font-weight: 700; color: var(--exo-text); }
.exo-card .hint { color: #7f98b5; font-size: .82rem; margin-top: 4px; }
.exo-badge {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 6px 12px; border-radius: 999px;
  border: 1px solid var(--exo-border);
  background: rgba(15,30,55,0.6);
  color: var(--exo-text); font-size: .85rem; font-weight: 600;
}
.exo-badge.warn { border-color: rgba(251,191,36,0.4); color: #fde68a; background: rgba(120,53,15,0.22); }
.exo-badge.ok { border-color: rgba(52,211,153,0.4); color: #bbf7d0; background: rgba(6,78,59,0.22); }
.exo-badge.info { border-color: rgba(34,211,238,0.4); color: #cffafe; background: rgba(8,51,68,0.28); }
.exo-section {
  border: 1px solid var(--exo-border);
  border-radius: 20px;
  padding: 18px 18px 10px 18px;
  background: rgba(8,18,36,0.55);
  backdrop-filter: blur(10px);
  margin-bottom: 12px;
}
.exo-section h3 { margin: 0 0 4px 0; }
.exo-section p { color: var(--exo-muted); margin-top: 0; }
.exo-callout {
  border: 1px solid rgba(251,191,36,0.28);
  background: linear-gradient(135deg, rgba(120,53,15,0.18), rgba(15,23,42,0.5));
  border-radius: 16px; padding: 14px 16px; color: #fde68a;
}
.exo-muted { color: var(--exo-muted); }

.gradio-container .tabs { border: none !important; }
.gradio-container .tab-nav {
  background: rgba(7,17,32,0.55) !important;
  border: 1px solid var(--exo-border) !important;
  border-radius: 16px !important;
  padding: 6px !important;
  gap: 6px !important;
  backdrop-filter: blur(10px);
}
.gradio-container .tab-nav button {
  border-radius: 12px !important;
  border: 1px solid transparent !important;
  color: var(--exo-muted) !important;
  font-weight: 600 !important;
  transition: all .18s ease !important;
}
.gradio-container .tab-nav button.selected,
.gradio-container .tab-nav button[aria-selected="true"] {
  color: var(--exo-text) !important;
  background: linear-gradient(135deg, rgba(8,145,178,0.35), rgba(124,58,237,0.28)) !important;
  border-color: rgba(103,232,249,0.3) !important;
  box-shadow: 0 0 0 1px rgba(103,232,249,0.08) inset;
}
.gradio-container .tab-nav button:hover {
  color: #d7e7ff !important;
  border-color: rgba(103,232,249,0.22) !important;
  background: rgba(15,30,55,0.7) !important;
}
.gradio-container .tab-nav button,
.gradio-container .overflow-menu button,
.gradio-container .overflow-dropdown button {
  color: #334155 !important;
}
.gradio-container .tab-nav button.selected,
.gradio-container .tab-nav button[aria-selected="true"],
.gradio-container .overflow-menu button.selected,
.gradio-container .overflow-dropdown button.selected,
.gradio-container .overflow-dropdown [aria-selected="true"] {
  color: #0b1220 !important;
  font-weight: 700 !important;
}
.gradio-container .block, .gradio-container .form, .gradio-container .panel {
  border-radius: var(--exo-radius) !important;
}
.gradio-container .gr-box, .gradio-container .gr-panel, .gradio-container .gr-form {
  border-color: var(--exo-border) !important;
  background: var(--exo-surface) !important;
  backdrop-filter: blur(10px);
}
.gradio-container button.primary {
  border: 1px solid rgba(103,232,249,0.35) !important;
  box-shadow: 0 8px 24px rgba(37,99,235,0.25) !important;
}
.gradio-container button.primary:hover { filter: brightness(1.08); transform: translateY(-1px); }
.gradio-container button.secondary {
  border: 1px solid var(--exo-border) !important;
}
.gradio-container button { transition: all .18s ease !important; }
.gradio-container input, .gradio-container textarea, .gradio-container select,
.gradio-container .wrap-inner, .gradio-container .svelte-1gfkn6j {
  border-radius: 12px !important;
}
.gradio-container label span, .gradio-container .block .label-wrap span {
  color: var(--exo-muted) !important;
}
.gradio-container .dataframe, .gradio-container table {
  border-radius: 14px !important;
  overflow: hidden;
}
.gradio-container .plot {
  border: 1px solid var(--exo-border);
  border-radius: 16px;
  background: rgba(7,17,32,0.55);
  padding: 6px;
  animation: exo-glow 6s ease-in-out infinite;
}
.gradio-container footer { display: none !important; }

/* Targeted fixes: remove default white panels/backgrounds in module components */
.gradio-container .tabitem,
.gradio-container [role="tabpanel"],
.gradio-container .tabpanel,
.gradio-container .styler,
.gradio-container .wrap,
.gradio-container .container,
.gradio-container .form,
.gradio-container .block,
.gradio-container .block.padded,
.gradio-container .gr-box,
.gradio-container .gr-panel,
.gradio-container .gr-form,
.gradio-container [data-testid="block-label"],
.gradio-container [data-testid="block-info"],
.gradio-container [data-testid="textbox"],
.gradio-container [data-testid="markdown"],
.gradio-container [data-testid="dataframe"],
.gradio-container [data-testid="file"],
.gradio-container [data-testid="upload"] {
  background-color: transparent;
}
.gradio-container .tabitem,
.gradio-container [role="tabpanel"] {
  background: rgba(8, 18, 36, 0.35) !important;
  border: 1px solid var(--exo-border);
  border-radius: 16px;
  padding: 14px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
}

/* Markdown / methodology text */
.gradio-container .markdown, .gradio-container .prose, .gradio-container .gr-markdown,
.gradio-container [data-testid="markdown"] > div, .gradio-container .md {
  color: var(--exo-text) !important;
  background: rgba(10, 22, 42, 0.52) !important;
  border: 1px solid var(--exo-border);
  border-radius: 14px;
  padding: 14px 16px;
}
.gradio-container .prose h1, .gradio-container .prose h2, .gradio-container .prose h3,
.gradio-container .markdown h1, .gradio-container .markdown h2, .gradio-container .markdown h3,
.gradio-container .gr-markdown h1, .gradio-container .gr-markdown h2, .gradio-container .gr-markdown h3 {
  color: #67e8f9 !important;
}
.gradio-container .prose p, .gradio-container .prose li, .gradio-container .prose strong,
.gradio-container .markdown p, .gradio-container .markdown li, .gradio-container .markdown strong,
.gradio-container .gr-markdown p, .gradio-container .gr-markdown li, .gradio-container .gr-markdown strong {
  color: #d7e7ff !important;
}
.gradio-container .prose a, .gradio-container .markdown a { color: var(--exo-cyan) !important; }

/* Text inputs, textareas, dropdown-like inputs, labels */
.gradio-container input, .gradio-container textarea, .gradio-container select,
.gradio-container .wrap input, .gradio-container .wrap textarea, .gradio-container .input-container textarea,
.gradio-container [data-testid="textbox"] textarea, .gradio-container [data-testid="textbox"] input {
  background: rgba(7, 17, 32, 0.78) !important;
  color: var(--exo-text) !important;
  border: 1px solid rgba(103, 232, 249, 0.22) !important;
  box-shadow: none !important;
}
.gradio-container input::placeholder, .gradio-container textarea::placeholder {
  color: #7f98b5 !important;
}
.gradio-container input:focus, .gradio-container textarea:focus, .gradio-container select:focus {
  border-color: rgba(103, 232, 249, 0.55) !important;
  outline: none !important;
  box-shadow: 0 0 0 2px rgba(34, 211, 238, 0.12) !important;
}
.gradio-container input:disabled, .gradio-container textarea:disabled,
.gradio-container .wrap:has(textarea:disabled), .gradio-container [data-testid="textbox"] textarea[disabled] {
  background: rgba(9, 20, 38, 0.72) !important;
  color: #cfe3ff !important;
  -webkit-text-fill-color: #cfe3ff !important;
  opacity: 1 !important;
  border-color: rgba(103, 232, 249, 0.18) !important;
}
.gradio-container label, .gradio-container label span, .gradio-container .label-wrap,
.gradio-container .label-wrap span, .gradio-container .block .label, .gradio-container .info {
  color: var(--exo-muted) !important;
  background: transparent !important;
}

/* Dataframes / tables */
.gradio-container .dataframe, .gradio-container [data-testid="dataframe"],
.gradio-container .table-wrap, .gradio-container .table-container,
.gradio-container .wrap.svelte-1x7b0kg, .gradio-container .df-container {
  background: rgba(10, 22, 42, 0.55) !important;
  border: 1px solid var(--exo-border) !important;
  border-radius: 14px !important;
}
.gradio-container table, .gradio-container thead, .gradio-container tbody, .gradio-container tr {
  background: transparent !important;
  color: var(--exo-text) !important;
}
.gradio-container th {
  background: rgba(14, 28, 52, 0.92) !important;
  color: #67e8f9 !important;
  border-color: rgba(103, 232, 249, 0.18) !important;
}
.gradio-container td {
  background: rgba(9, 20, 38, 0.55) !important;
  color: #e6f1ff !important;
  border-color: rgba(103, 232, 249, 0.12) !important;
}
.gradio-container tr:nth-child(even) td { background: rgba(12, 24, 46, 0.55) !important; }
.gradio-container tr:hover td { background: rgba(34, 211, 238, 0.08) !important; }

/* File/export components */
.gradio-container [data-testid="file"], .gradio-container .file-preview, .gradio-container .file,
.gradio-container .upload-container, .gradio-container .download {
  background: rgba(10, 22, 42, 0.55) !important;
  color: var(--exo-text) !important;
  border: 1px solid var(--exo-border) !important;
  border-radius: 14px !important;
}

/* Buttons - consistent states */
.gradio-container button.secondary {
  background: rgba(15, 30, 55, 0.78) !important;
  color: #d7e7ff !important;
  border: 1px solid rgba(103, 232, 249, 0.35) !important;
}
.gradio-container button.secondary:hover, .gradio-container button.secondary:focus-visible {
  background: rgba(22, 42, 74, 0.9) !important;
  border-color: rgba(103, 232, 249, 0.55) !important;
  box-shadow: 0 0 0 2px rgba(34,211,238,0.10) !important;
}
.gradio-container button.primary:disabled, .gradio-container button.secondary:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

@media (max-width: 900px) {
  .exo-hero { padding: 20px; }
  .exo-grid { grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }
  .gradio-container .tab-nav { flex-wrap: wrap; }
  .gradio-container .tabitem, .gradio-container [role="tabpanel"] { padding: 10px; }
}
@media (max-width: 640px) {
  .exo-hero h1 { font-size: 1.6rem; }
  .exo-card .value { font-size: 1.3rem; }
  .gradio-container .markdown, .gradio-container .prose { padding: 12px; }
}
"""


def _style_axes(fig: Figure, axes: list[Any], titles: list[str], xlabels: list[str]) -> None:
    fig.patch.set_facecolor("#07111d")
    for axis, title, xlabel in zip(axes, titles, xlabels, strict=False):
        axis.set_facecolor("#0b1628")
        axis.set_title(title, color="#e6f1ff", fontsize=11, fontweight="600", pad=10)
        axis.set_xlabel(xlabel, color="#9fb6d1", fontsize=9)
        axis.set_ylabel(axis.get_ylabel() or "Value", color="#9fb6d1", fontsize=9)
        axis.tick_params(colors="#8ea8c3", labelsize=8)
        axis.grid(alpha=0.16, color="#67e8f9", linestyle="--", linewidth=0.6)
        for spine in axis.spines.values():
            spine.set_color("#26415f")


def _analysis_snapshot() -> dict[str, Any]:
    analyses = store.analyses(limit=500)
    candidates = store.candidates(limit=500)
    target_ids = {str(item.get("target_id", "")) for item in analyses if item.get("target_id")}
    synthetic = 0
    for item in analyses:
        raw_meta = item.get("metadata_json")
        metadata: dict[str, Any] = {}
        if isinstance(raw_meta, str):
            try:
                parsed = json.loads(raw_meta)
                if isinstance(parsed, dict):
                    metadata = parsed
            except json.JSONDecodeError:
                metadata = {}
        elif isinstance(raw_meta, dict):
            metadata = raw_meta
        if metadata.get("synthetic"):
            synthetic += 1
    model_frame, metrics_frame, model_msg = model_status()
    model_state = "Not trained"
    if not model_frame.empty and "status" in model_frame.columns:
        model_state = str(model_frame.iloc[0].get("status", "Not trained"))
    return {
        "analysis_count": len(analyses),
        "candidate_count": len(candidates),
        "target_count": len(target_ids),
        "synthetic_count": synthetic,
        "recent": analyses[:8],
        "model_state": model_state,
        "model_msg": model_msg,
        "metrics_available": not metrics_frame.empty,
    }


def _hero_html() -> str:
    return (
        '<div class="exo-hero">'
        '<div class="exo-eyebrow">EXOPLANET DISCOVERY AI · RESEARCH DASHBOARD</div>'
        "<h1>Premium astronomy intelligence for light-curve discovery</h1>"
        "<p>Explore NASA mission data, preprocess photometric signals, extract scientific features, "
        "and screen for transit-like candidates with transparent, evidence-based reporting. "
        "Predictions are never presented as confirmed exoplanets.</p>"
        "</div>"
    )


def _metric_card(label: str, value: str, hint: str = "") -> str:
    label_e = html.escape(label)
    value_e = html.escape(value)
    hint_e = html.escape(hint)
    return (
        '<div class="exo-card">'
        f'<div class="label">{label_e}</div>'
        f'<div class="value">{value_e}</div>'
        f'<div class="hint">{hint_e}</div>'
        "</div>"
    )


def _dashboard_html() -> str:
    snap = _analysis_snapshot()
    model_badge = (
        '<span class="exo-badge warn">Model: Not trained</span>'
        if snap["model_state"].lower() != "trained"
        else '<span class="exo-badge ok">Model: Trained</span>'
    )
    metrics_badge = (
        '<span class="exo-badge ok">Metrics: Available</span>'
        if snap["metrics_available"]
        else '<span class="exo-badge warn">Metrics: Not available</span>'
    )
    cards = "".join(
        [
            _metric_card("Total analyses", str(snap["analysis_count"]), "Recorded in local result store"),
            _metric_card("Candidate records", str(snap["candidate_count"]), "AI-predicted, unverified"),
            _metric_card("Unique targets", str(snap["target_count"]), "Distinct analyzed targets"),
            _metric_card(
                "Synthetic analyses",
                str(snap["synthetic_count"]),
                "Demo runs clearly labelled as synthetic",
            ),
        ]
    )
    return (
        f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin:4px 0 12px 0">'
        f'{model_badge}{metrics_badge}'
        '<span class="exo-badge info">Data sources: NASA archive + local upload + demo</span>'
        "</div>"
        f'<div class="exo-grid">{cards}</div>'
        '<div class="exo-callout"><strong>Scientific integrity:</strong> '
        "Random Forest status and evaluation metrics are shown only from real, available artifacts. "
        "No accuracy, precision, recall, or F1-score is fabricated.</div>"
    )


def _recent_activity_frame() -> pd.DataFrame:
    analyses = store.analyses(limit=12)
    if not analyses:
        return pd.DataFrame(
            [{"message": "No analysis activity recorded yet. Run an analysis to populate this view."}]
        )
    rows = []
    for item in analyses:
        meta = item.get("metadata_json", "{}")
        if isinstance(meta, str):
            try:
                meta_obj = json.loads(meta)
            except json.JSONDecodeError:
                meta_obj = {}
        else:
            meta_obj = meta if isinstance(meta, dict) else {}
        rows.append(
            {
                "time": item.get("created_at"),
                "target": item.get("target_id"),
                "source": item.get("source"),
                "status": item.get("status"),
                "cadences": item.get("cadence_count"),
                "candidates": meta_obj.get("candidates"),
                "synthetic": bool(meta_obj.get("synthetic", False)),
            }
        )
    return pd.DataFrame(rows)


def _module_header(
    eyebrow: str, title: str, subtitle: str, *, heading_color: str | None = None
) -> str:
    if heading_color:
        heading_style = (
            f' style="color: {html.escape(heading_color, quote=True)}; '
            'text-shadow: 0 0 10px rgba(103,232,249,0.28), 0 1px 0 rgba(230,241,255,0.18);"'
        )
    else:
        heading_style = ""
    return (
        '<div class="exo-section">'
        f'<div class="exo-eyebrow">{html.escape(eyebrow)}</div>'
        f"<h3{heading_style}>{html.escape(title)}</h3>"
        f"<p>{html.escape(subtitle)}</p>"
        "</div>"
    )


def _model_insights_html() -> str:
    model_frame, metrics_frame, _ = model_status()
    values = _load_model_metadata() or {}
    status = "Not trained"
    if not model_frame.empty and "status" in model_frame.columns:
        status = str(model_frame.iloc[0].get("status", "Not trained"))
    status_badge = "ok" if str(status).lower() == "trained" else "warn"
    metrics_text = "Available" if not metrics_frame.empty else "Not available"
    metrics_badge = "ok" if not metrics_frame.empty else "warn"
    sample_counts = values.get("sample_counts") or {}
    test_samples = sample_counts.get("test_samples")
    test_text = f"{test_samples} held-out samples" if test_samples is not None else "Shown only from genuine held-out evaluation"
    explanation = (
        '<div class="exo-callout"><strong>Explanation:</strong> Genuine labelled training data is required. '
        "This project does not fabricate accuracy, precision, recall, F1-score, confusion matrices, or ROC-AUC. "
        "When a real labelled dataset is trained and evaluated on held-out data, checkpoint artifacts populate this view.</div>"
    )
    if values and str(status).lower() == "trained":
        explanation = (
            '<div class="exo-callout"><strong>Evaluation basis:</strong> Metrics below are loaded from saved artifacts '
            "produced on genuine held-out Kepler KOI-labelled light curves (processed through the project's existing "
            "preprocessing and feature extraction pipeline). No synthetic demo data is used for training.</div>"
        )
    return (
        '<div class="exo-grid">'
        + _metric_card("Model 1", values.get("model_name", "Random Forest"), "Baseline classifier for transit-feature screening")
        + _metric_card("Model status", status, "Read from checkpoint metadata when present")
        + _metric_card("Evaluation metrics", metrics_text, test_text)
        + "</div>"
        f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin:8px 0 6px 0">'
        f'<span class="exo-badge {status_badge}">Status: {html.escape(str(status))}</span>'
        f'<span class="exo-badge {metrics_badge}">Evaluation metrics: {html.escape(metrics_text)}</span>'
        "</div>"
        + explanation
    )


def _model_details_markdown() -> str:
    values = _load_model_metadata()
    if not values:
        return (
            "### Evaluation details\n"
            "- **Training status:** Not trained.\n"
            "- **Genuine labelled dataset:** not available in project artifacts.\n"
            "- **Required for training/evaluation:** genuine labelled light curves (positive/negative), "
            "preprocessing + feature extraction, and a leakage-safe held-out split.\n"
            "- **Current policy:** no fabricated labels, samples, or metrics."
        )
    provenance = values.get("dataset_provenance") or {}
    labels = values.get("label_definition") or {}
    split = values.get("split") or {}
    counts = values.get("sample_counts") or {}
    dist = values.get("class_distribution") or {}
    metrics = values.get("metrics") or {}
    roc = metrics.get("roc_auc")
    roc_text = f"{roc:.6f}" if isinstance(roc, (int, float)) and not (isinstance(roc, float) and math.isnan(roc)) else "not available"
    lines = [
        "### Genuine evaluation details",
        f"- **Training status:** {values.get('training_status', 'trained')} (trained at {values.get('trained_at', 'unknown')})",
        f"- **Label source:** {provenance.get('label_source', 'NASA Exoplanet Archive KOI dispositions')}",
        f"- **Light-curve source:** {provenance.get('light_curve_source', 'Kepler mission light curves via Lightkurve')}",
        f"- **Positive label:** CONFIRMED (1); **Negative label:** FALSE POSITIVE (0). Excluded: CANDIDATE / NOT DISPOSITIONED / uncertain or conflicting star labels.",
        f"- **Preprocessing:** {provenance.get('preprocessing', 'existing preprocessing pipeline')}",
        f"- **Feature extraction:** {provenance.get('feature_extraction', 'existing feature extraction pipeline')}",
        f"- **Leakage prevention:** split at star/target group level (`kepid`), group overlap in train/test = {split.get('leakage_group_overlap_count', 'unknown')} (method: {split.get('method', 'unknown')}, seed {split.get('seed', 'n/a')}).",
        f"- **Samples:** total {counts.get('total_samples', 'n/a')}, train {counts.get('train_samples', 'n/a')} ({counts.get('train_groups', 'n/a')} groups), test {counts.get('test_samples', 'n/a')} ({counts.get('test_groups', 'n/a')} groups).",
        f"- **Class distribution:** overall {dist.get('overall', {})}, train {dist.get('train', {})}, test {dist.get('test', {})}.",
        f"- **Held-out metrics (genuine):** accuracy/precision/recall/F1 and confusion matrix from saved artifacts; ROC-AUC: {roc_text} (reported where appropriate).",
        "- **Integrity:** no synthetic demo data used for training; no fabricated labels, samples, or metrics.",
    ]
    report_path = Path("models/evaluation/kepler_koi_random_forest_evaluation.json")
    if report_path.exists():
        lines.append(f"- **Evaluation report artifact:** `{report_path.as_posix()}`")
    return "\n".join(lines)


def refresh_dashboard() -> tuple[str, pd.DataFrame]:
    return _dashboard_html(), _recent_activity_frame()


def _curve_cache_key(curve: LightCurve) -> bytes:
    digest = hashlib.blake2b(digest_size=20)
    digest.update(curve.tic_id.encode("utf-8"))
    digest.update(curve.source.encode("utf-8"))
    for values in (curve.time, curve.flux, curve.flux_err):
        array = np.ascontiguousarray(values, dtype=np.float64)
        digest.update(memoryview(array).cast("B"))
    return digest.digest()


def _prepare_analysis(
    raw: LightCurve,
    progress: gr.Progress,
) -> tuple[LightCurve, dict[str, float]]:
    key = _curve_cache_key(raw)
    with _analysis_cache_lock:
        cached = _analysis_cache.get(key)
        if cached is not None:
            _analysis_cache.move_to_end(key)
            logger.info("Using cached preprocessing/features for %s", raw.tic_id)
            progress(0.35, desc="Using cached preprocessing and features")
            return cached[1], cached[2]

    progress(0.15, desc="Cleaning and preprocessing light curve")
    processed = preprocess_light_curve(raw)
    progress(0.4, desc="Extracting scientific features")
    features = extract_features(processed)
    with _analysis_cache_lock:
        _analysis_cache[key] = (raw, processed, features)
        _analysis_cache.move_to_end(key)
        while len(_analysis_cache) > 2:
            _analysis_cache.popitem(last=False)
    return processed, features


def _load(
    file: object | None,
    use_demo: bool,
    downloaded_curve: object | None = None,
    target: str = "",
    mission: str = "TESS",
) -> tuple[Any, str]:
    if downloaded_curve is not None:
        logger.info("Using downloaded %s light curve for target %s", mission, target)
        return downloaded_curve, f"{mission} archive download: {target}"
    if use_demo or file is None:
        logger.info("Using synthetic demo light curve")
        return create_demo_light_curve(), "DEMO DATA - synthetic demonstration"
    path = (
        os.fspath(file)
        if isinstance(file, os.PathLike)
        else getattr(file, "name", str(file))
    )
    curve = load_light_curve(path)
    return curve, f"Uploaded data: {curve.metadata.get('filename', path)}"


def run_analysis(file: object | None, use_demo: bool) -> tuple[
    object, object, pd.DataFrame, pd.DataFrame, str, str
]:
    return run_analysis_with_state(file, use_demo, None, "", "TESS")


def run_analysis_with_state(
    file: object | None,
    use_demo: bool,
    downloaded_curve: object | None,
    target: str,
    mission: str,
    progress: gr.Progress = _GRADIO_PROGRESS,
) -> tuple[object, object, pd.DataFrame, pd.DataFrame, str, str]:
    fig: Figure | None = None
    try:
        progress(0.05, desc="Loading selected light curve")
        raw, source = _load(file, use_demo, downloaded_curve, target, mission)
        logger.info("Starting transit analysis for %s (%s)", raw.tic_id, source)
        processed, features = _prepare_analysis(raw, progress)
        progress(0.58, desc="Searching transit periods with Box Least Squares")
        candidates = detect_transits(
            processed,
            min_snr=0.5 if raw.metadata.get("demo") else 3.0,
            confidence_threshold=0.1 if raw.metadata.get("demo") else 0.7,
            feature_values=features,
        )

        progress(0.82, desc="Preparing plots and candidate results")
        raw_plot = downsample_light_curve(raw, MAX_PLOT_CADENCES)
        processed_plot = downsample_light_curve(processed, MAX_PLOT_CADENCES)
        fig, axes = plt.subplots(3, 1, figsize=(11.5, 9.5), constrained_layout=True)
        axes[0].plot(raw_plot.time, raw_plot.flux, color="#38bdf8", lw=0.9, alpha=0.95)
        axes[0].fill_between(raw_plot.time, raw_plot.flux, alpha=0.08, color="#22d3ee")
        axes[0].set_ylabel("Flux")
        axes[1].plot(
            processed_plot.time,
            processed_plot.flux,
            color="#5eead4",
            lw=1.0,
            alpha=0.98,
        )
        axes[1].fill_between(
            processed_plot.time,
            processed_plot.flux,
            alpha=0.08,
            color="#34d399",
        )
        axes[1].set_ylabel("Normalized flux")
        plot_duration = max(float(processed_plot.time[-1] - processed_plot.time[0]), 1e-6)
        frequency = np.linspace(
            1.0 / plot_duration,
            min(10.0, 0.5 * len(processed_plot) / plot_duration),
            2_000,
        )
        power = LombScargle(processed_plot.time, processed_plot.flux).power(frequency)
        axes[2].plot(frequency, power, color="#a78bfa", lw=1.0)
        axes[2].fill_between(frequency, power, alpha=0.12, color="#8b5cf6")
        axes[2].set_ylabel("Power")
        _style_axes(
            fig,
            list(axes),
            [
                "Raw light curve",
                "Processed and normalized light curve",
                "Lomb–Scargle periodogram",
            ],
            ["Time", "Time", "Frequency (cycles/day)"],
        )

        feature_frame = pd.DataFrame(
            [{"feature": name, "value": value} for name, value in features.items()]
        )
        rows: list[dict[str, Any]] = []
        for candidate in candidates:
            transit = candidate.best_transit
            row = {
                "target_id": candidate.tic_id,
                "candidate_name": candidate.candidate_name,
                "classification": "AI_PREDICTED_CANDIDATE",
                "confirmed_exoplanet": False,
                "probability": candidate.exoplanet_probability,
                "period_days": candidate.estimated_period_days,
                "transit_depth_ppm": transit.transit_depth_ppm if transit else None,
                "transit_duration_hours": transit.duration_hours if transit else None,
                "snr": transit.snr if transit else None,
                "confidence": candidate.exoplanet_probability,
                "notes": candidate.notes,
            }
            rows.append(row)
        store.add_analysis(
            raw.tic_id,
            source,
            "analysis completed",
            len(processed),
            {"synthetic": bool(raw.metadata.get("demo", False)), "candidates": len(rows)},
        )
        store.add_candidates(rows)
        candidate_frame = pd.DataFrame(rows)
        if candidate_frame.empty:
            candidate_frame = pd.DataFrame(
                [{"message": "No candidate exceeded the configured transit confidence threshold."}]
            )
            status = "Analysis completed: no transit candidates detected."
        else:
            status = f"Transit detection completed: {len(rows)} candidate(s) found."
        logger.info(status)
        progress(1.0, desc=status)
        plt.close(fig)
        return fig, fig, feature_frame, candidate_frame, source, status
    except Exception as exc:
        logger.exception("Light-curve analysis failed")
        status = f"Analysis failed: {type(exc).__name__}: {exc}"
        progress(1.0, desc=status)
        if fig is not None:
            plt.close(fig)
        empty_fig, empty_axes = plt.subplots(figsize=(8, 3.5))
        empty_fig.patch.set_facecolor("#07111d")
        empty_axes.set_facecolor("#0b1628")
        empty_axes.text(
            0.5,
            0.5,
            str(exc),
            ha="center",
            va="center",
            wrap=True,
            color="#fca5a5",
            fontsize=11,
        )
        empty_axes.set_axis_off()
        plt.close(empty_fig)
        return (
            empty_fig,
            empty_fig,
            pd.DataFrame([{"error": str(exc)}]),
            pd.DataFrame([{"error": str(exc)}]),
            "Unavailable",
            status,
        )


def archive_search(target: str, mission: str) -> pd.DataFrame:
    if not target.strip():
        return pd.DataFrame([{"message": "Enter a target name or ID."}])
    try:
        results = search_targets(target.strip(), mission)  # type: ignore[arg-type]
        if not results:
            return pd.DataFrame(
                [{"message": f"No {mission} light-curve products found for '{target.strip()}'."}]
            )
        return pd.DataFrame(results)
    except Exception as exc:
        return pd.DataFrame([{"error": str(exc)}])


def archive_download(
    target: str,
    mission: str,
    sector: str,
    progress: gr.Progress = _GRADIO_PROGRESS,
) -> tuple[pd.DataFrame, str, LightCurve | None]:
    if not target.strip():
        return (
            pd.DataFrame([{"message": "Enter a target name or ID."}]),
            "No target selected",
            None,
        )
    sector_value = int(sector) if sector.strip().isdigit() else None
    try:
        progress(0.05, desc="Connecting to NASA MAST")
        curve = download_light_curve(
            target.strip(),
            mission,
            sector=sector_value,
            cache_dir=archive_cache_dir(),
        )  # type: ignore[arg-type]
        progress(0.85, desc="Preparing light-curve preview")
        preview = downsample_light_curve(curve, MAX_PLOT_CADENCES)
        frame = pd.DataFrame(
            {
                "time": preview.time,
                "flux": preview.flux,
                "flux_err": preview.flux_err,
            }
        )
        logger.info(
            "Downloaded %s target %s with %d cadences; preview contains %d",
            mission,
            target,
            len(curve),
            len(preview),
        )
        progress(1.0, desc="Preview ready")
        return (
            frame,
            f"Downloaded {mission} data for {target}; {len(curve)} cadences "
            f"({len(preview)} shown in preview)",
            curve,
        )
    except Exception as exc:
        progress(1.0, desc="NASA archive request failed")
        logger.exception("NASA archive download failed for %s", target)
        return pd.DataFrame([{"error": str(exc)}]), f"NASA archive unavailable: {exc}", None


def candidate_filter(text: str) -> pd.DataFrame:
    frame = pd.DataFrame(store.candidates())
    if frame.empty:
        return frame
    if text.strip():
        mask = frame.astype(str).apply(
            lambda column: column.str.contains(text.strip(), case=False, regex=False)
        ).any(axis=1)
        frame = frame.loc[mask]
    return frame


def export_candidates() -> str:
    path = Path("data/exports/gradio_candidate_results.csv")
    store.candidate_csv(path)
    return str(path)


def _load_model_metadata() -> dict[str, Any] | None:
    metadata_path = Path("models/checkpoints/baseline_random_forest.json")
    if not metadata_path.exists():
        return None
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _confusion_matrix_frame(values: dict[str, Any] | None) -> pd.DataFrame:
    if not values:
        return pd.DataFrame()
    cm = values.get("confusion_matrix") or {}
    matrix = cm.get("matrix")
    if isinstance(matrix, list) and len(matrix) == 2:
        try:
            return pd.DataFrame(matrix, index=["Actual 0 (FP)", "Actual 1 (CONFIRMED)"], columns=["Pred 0 (FP)", "Pred 1 (CONFIRMED)"])
        except Exception:
            return pd.DataFrame(matrix)
    metrics = values.get("metrics") or {}
    tn, fp, fn, tp = metrics.get("tn"), metrics.get("fp"), metrics.get("fn"), metrics.get("tp")
    if None not in (tn, fp, fn, tp):
        return pd.DataFrame(
            [[int(tn), int(fp)], [int(fn), int(tp)]],
            index=["Actual 0 (FP)", "Actual 1 (CONFIRMED)"],
            columns=["Pred 0 (FP)", "Pred 1 (CONFIRMED)"],
        )
    return pd.DataFrame()


def model_status() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    values = _load_model_metadata()
    if not values:
        return (
            pd.DataFrame(
                [
                    {
                        "model": "Random Forest",
                        "status": "Not trained",
                        "evaluation_metrics": "Not available",
                    }
                ]
            ),
            pd.DataFrame(),
            "Genuine labelled training data is required. No real labelled training dataset has been supplied, so no model metrics are claimed.",
        )
    metrics = values.get("metrics", {})
    comparison = pd.DataFrame([metrics]) if metrics else pd.DataFrame()
    sample_counts = values.get("sample_counts") or {}
    train_size = sample_counts.get("train_samples", values.get("train_size"))
    test_size = sample_counts.get("test_samples", values.get("test_size"))
    trained_at = values.get("trained_at", "unknown")
    status = values.get("training_status", "trained")
    note = values.get("evaluation_note") or "These metrics were produced by the baseline training pipeline on genuine held-out test data."
    return (
        pd.DataFrame(
            [
                {
                    "model": values.get("model_name", "Random Forest"),
                    "status": status,
                    "train_size": train_size,
                    "test_size": test_size,
                    "trained_at": trained_at,
                }
            ]
        ),
        comparison,
        note,
    )


def _build_app() -> gr.Blocks:
    with gr.Blocks(
        title="EXOPLANET DISCOVERY AI",
        theme=THEME,
        css=CUSTOM_CSS,
        fill_height=True,
    ) as demo:
        gr.HTML('<div id="exo-app-root"></div>')
        gr.HTML(_hero_html())
        with gr.Row():
            analysis_status = gr.Textbox(
                label="Analysis status",
                value="Ready. Load data, then run preprocessing or transit detection.",
                interactive=False,
                scale=4,
            )
            refresh_overview = gr.Button("Refresh dashboard data", variant="secondary", scale=1)

        with gr.Tabs() as tabs:
            with gr.Tab("Home Dashboard", id="home"):
                gr.HTML(_module_header("MISSION CONTROL", "Home Dashboard", "Live overview from actual project data: analyses, candidate records, and model status."))
                dashboard_cards = gr.HTML(value=_dashboard_html())
                with gr.Row():
                    gr.HTML(
                        '<div class="exo-section"><div class="exo-eyebrow">WORKFLOW</div>'
                        "<h3>Discovery pipeline</h3>"
                        "<p>1) Ingest NASA or local light curves → 2) Preprocess and normalize → "
                        "3) Extract scientific features → 4) Run BLS screening → 5) Review AI-predicted candidates "
                        "and model readiness. All outputs are evidence-based.</p></div>"
                    )
                gr.HTML(
                    '<div class="exo-section"><div class="exo-eyebrow">RECENT ACTIVITY</div>'
                    "<h3>Analysis history</h3><p>Most recent runs recorded in the local results store.</p></div>"
                )
                activity_table = gr.Dataframe(
                    value=_recent_activity_frame(),
                    label="Recent analysis activity",
                    interactive=False,
                    wrap=True,
                )

            with gr.Tab("NASA Data Explorer", id="explorer"):
                gr.HTML(_module_header("DATA INGESTION", "NASA Data Explorer", "Search TESS/Kepler mission products, download light curves, or upload local CSV/FITS files."))
                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Group():
                            gr.Markdown("### Archive search & retrieval")
                            with gr.Row():
                                target = gr.Textbox(label="Target name or ID", value="Kepler-90")
                                mission = gr.Dropdown(["TESS", "Kepler"], value="TESS", label="Mission")
                            with gr.Row():
                                sector = gr.Textbox(label="Optional sector/quarter")
                                search_button = gr.Button("Search NASA archive", variant="secondary")
                                download_button = gr.Button("Download selected target", variant="primary")
                            archive_results = gr.Dataframe(
                                label="Available archive products",
                                interactive=False,
                                wrap=True,
                            )
                            downloaded = gr.Dataframe(
                                label="Downloaded light curve preview",
                                interactive=False,
                                wrap=True,
                            )
                            download_status = gr.Textbox(
                                label="Download status",
                                value="No download requested yet.",
                                interactive=False,
                            )
                            downloaded_curve = gr.State(value=None)
                    with gr.Column(scale=2):
                        with gr.Group():
                            gr.Markdown("### Local input & demo mode")
                            upload = gr.File(label="Upload CSV or FITS light curve")
                            use_demo = gr.Checkbox(
                                value=True,
                                label="Use synthetic DEMO DATA when no file/download is selected",
                            )
                            gr.Markdown(
                                "_Demo data is explicitly synthetic and must not be interpreted as an exoplanet result. "
                                "Downloaded/uploaded observational data is processed as provided._"
                            )

                search_button.click(archive_search, [target, mission], archive_results)
                download_button.click(
                    archive_download,
                    [target, mission, sector],
                    [downloaded, download_status, downloaded_curve],
                    trigger_mode="once",
                    concurrency_limit=1,
                    concurrency_id="nasa-archive-download",
                )

            with gr.Tab("Light Curve Analysis", id="analysis"):
                gr.HTML(_module_header("SIGNAL PROCESSING", "Light Curve Analysis", "Run preprocessing, normalization, and feature extraction with scientific visualizations."))
                with gr.Row():
                    run = gr.Button("Run preprocessing and feature extraction", variant="primary", scale=2)
                    source = gr.Textbox(label="Data source", interactive=False, scale=3)
                with gr.Row():
                    light_chart = gr.Plot(label="Light-curve visualizations (raw, processed, periodogram)")
                    periodogram = gr.Plot(label="Periodogram view", visible=False)
                features = gr.Dataframe(
                    label="Extracted astronomical features",
                    interactive=False,
                    wrap=True,
                )

            with gr.Tab("Exoplanet Detection", id="detection"):
                gr.HTML(_module_header("TRANSIT SCREENING", "Exoplanet Detection", "Box Least Squares screening produces AI-predicted candidate signals for review, not confirmations.", heading_color="#111827"))
                detect_button = gr.Button("Run transit detection", variant="primary")
                candidates = gr.Dataframe(
                    label="AI-predicted candidates",
                    interactive=False,
                    wrap=True,
                )
                gr.Markdown(
                    "**Interpretation note:** Candidates are model-assisted predictions based on available signals and thresholds. "
                    "Independent follow-up is required before any confirmation claim."
                )

            with gr.Tab("AI Model Insights", id="model"):
                gr.HTML(_module_header("MODEL TRANSPARENCY", "AI Model Insights", "Honest, artifact-backed model status and evaluation reporting.", heading_color="#111827"))
                _init_model_frame, _init_metrics, _init_model_msg = model_status()
                insights_html = gr.HTML(value=_model_insights_html())
                model_frame = gr.Dataframe(value=_init_model_frame, label="Model status", interactive=False, wrap=True)
                metrics_frame = gr.Dataframe(value=_init_metrics, label="Evaluation metrics (actual, held-out)", interactive=False, wrap=True)
                confusion_frame = gr.Dataframe(value=_confusion_matrix_frame(_load_model_metadata()), label="Confusion matrix (actual, held-out)", interactive=False, wrap=True)
                model_details = gr.Markdown(value=_model_details_markdown())
                model_message = gr.Textbox(value=_init_model_msg, label="Evaluation status", interactive=False, lines=3)
                model_button = gr.Button("Refresh model status", variant="secondary")

                def _refresh_model_insights() -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
                    frame, metrics, msg = model_status()
                    return (
                        _model_insights_html(),
                        frame,
                        metrics,
                        _confusion_matrix_frame(_load_model_metadata()),
                        _model_details_markdown(),
                        msg,
                    )

                model_button.click(
                    _refresh_model_insights,
                    [],
                    [insights_html, model_frame, metrics_frame, confusion_frame, model_details, model_message],
                )

            with gr.Tab("Candidate Explorer", id="candidates"):
                gr.HTML(_module_header("DISCOVERY REGISTER", "Candidate Explorer", "Browse, filter, and export AI-predicted candidate records stored locally.", heading_color="#111827"))
                with gr.Row():
                    search = gr.Textbox(label="Search and filter candidates", placeholder="Filter by target, notes, classification...")
                    export_button = gr.Button("Export candidate CSV", variant="secondary")
                candidate_table = gr.Dataframe(
                    value=pd.DataFrame(store.candidates()),
                    label="Candidate register",
                    interactive=False,
                    wrap=True,
                )
                export_file = gr.File(label="Download CSV export")
                search.submit(candidate_filter, search, candidate_table)
                search.change(candidate_filter, search, candidate_table)
                export_button.click(export_candidates, [], export_file)

            with gr.Tab("About & Methodology", id="about"):
                gr.HTML(_module_header("SCIENTIFIC CONTEXT", "About & Methodology", "Methods, integrity guarantees, and known limitations of this research workspace.", heading_color="#111827"))
                gr.Markdown(
                    "### Methodology\n"
                    "- **Ingestion:** NASA TESS/Kepler archive retrieval (network-dependent) and local CSV/FITS loading; synthetic demo data is explicitly labelled.\n"
                    "- **Preprocessing:** quality filtering, robust normalization, detrending/smoothing, and bounded sampling for large light curves.\n"
                    "- **Feature extraction:** statistical, variability, Lomb–Scargle periodogram, autocorrelation, and frequency-domain features.\n"
                    "- **Detection:** Box Least Squares (BLS) transit screening; optional trained Random Forest probability blending when a genuine trained checkpoint is available.\n"
                    "- **Training & evaluation:** when genuine labelled Kepler KOI light curves (CONFIRMED vs FALSE POSITIVE, excluding CANDIDATE/NOT DISPOSITIONED and conflicting labels) are acquired and processed through the existing pipeline, the Random Forest can be trained with a star/target-group (`kepid`) held-out split and evaluated with real metrics. Check the **AI Model Insights** tab for current artifact-backed status, sample counts, confusion matrix, ROC-AUC (where appropriate), provenance, and methodology.\n\n"
                    "### Limitations\n"
                    "- AI predictions are **not** confirmed exoplanets and require independent scientific follow-up.\n"
                    "- NASA archive availability and completeness depend on network/service access.\n"
                    "- Performance metrics can only be reported from real, held-out evaluation on genuine labelled data; none are fabricated.\n"
                    "- Large curves are downsampled/bounded for visualization and screening efficiency, which can affect sensitivity.\n\n"
                    "### Responsible reporting\n"
                    "This interface distinguishes synthetic demos, uploaded/downloaded observations, and AI-predicted candidates. "
                    "Model status and metrics reflect actual artifacts only."
                )

        def _analysis_and_refresh(*args: Any) -> tuple[Any, Any, pd.DataFrame, pd.DataFrame, str, str, str, pd.DataFrame]:
            chart, period, feats, cands, src, status = run_analysis_with_state(*args)
            dash, activity = refresh_dashboard()
            return chart, period, feats, cands, src, status, dash, activity

        run.click(
            _analysis_and_refresh,
            [upload, use_demo, downloaded_curve, target, mission],
            [light_chart, periodogram, features, candidates, source, analysis_status, dashboard_cards, activity_table],
        )
        detect_button.click(
            _analysis_and_refresh,
            [upload, use_demo, downloaded_curve, target, mission],
            [light_chart, periodogram, features, candidates, source, analysis_status, dashboard_cards, activity_table],
        )
        refresh_overview.click(refresh_dashboard, [], [dashboard_cards, activity_table])

        demo.load(refresh_dashboard, [], [dashboard_cards, activity_table])
        demo.load(_refresh_model_insights, [], [insights_html, model_frame, metrics_frame, model_message])

    return demo


app = _build_app()

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_error=True,
    )
