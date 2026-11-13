"""
app.py  -  Streamlit GUI for Digital Image Tampering & Forgery Detection.

Run with:
    streamlit run app.py

Color scheme (40 : 40 : 20):
    Cream:     #F1EEE8 (40%)  -  main background, cards, inputs
    Deep Teal: #0E4F4A (40%)  -  sidebar, headers, text, tabs
    Coral:     #FF6F5E (20%)  -  CTA buttons, alerts, slider accents
"""

import base64
import os
import tempfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from main import analyze_image


# ──────────────────────────────────────────────────────────────────────
# Logo Helper
# ──────────────────────────────────────────────────────────────────────
def get_logo_b64() -> str:
    """Load the generated PixelGuard icon as base64 for embedding."""
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
    if os.path.exists(logo_path):
        try:
            with open(logo_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return ""
    return ""


# ──────────────────────────────────────────────────────────────────────
# Custom CSS  (Cream 40% / Deep Teal 40% / Coral 20%)
# ──────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:wght@600;700;800&display=swap');

/* ── Global ───────────────────────────────────────────────────── */
html, body, .stApp {
    background-color: #F1EEE8 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.block-container {
    padding-top: 1.5rem !important;
    max-width: 1100px !important;
}

/* ── Typography ───────────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Playfair Display', Georgia, serif !important;
    color: #0E4F4A !important;
}
p, label, li {
    font-family: 'Inter', sans-serif !important;
    color: #0A3B37;
}

/* Restore Material Symbols font explicitly (prevents ligature text bleed) */
[data-testid="stIconMaterial"],
.material-symbols-rounded,
[class*="material-symbols"] {
    font-family: "Material Symbols Rounded", "Material Icons", sans-serif !important;
    font-weight: normal !important;
    font-style: normal !important;
    text-transform: none !important;
    letter-spacing: normal !important;
    font-feature-settings: 'liga' 1 !important;
    -webkit-font-feature-settings: 'liga' 1 !important;
    display: inline-block !important;
    line-height: 1 !important;
}

/* ── Sidebar (Deep Teal - 40%) ────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(195deg, #0A3B37 0%, #0E4F4A 55%, #136B63 100%) !important;
    border-right: none !important;
    box-shadow: 4px 0 24px rgba(14,79,74,0.18);
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] div {
    color: #F1EEE8 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(241,238,232,0.12) !important;
    margin: 1rem 0;
}

/* ── Sidebar sliders: White Dragging Line & High-Contrast Thumb ── */
/* 1. Track Bar (line of dragging) -> Crisp WHITE */
section[data-testid="stSidebar"] [data-testid="stSlider"] .react-aria-SliderTrack > div,
section[data-testid="stSidebar"] [data-testid="stSlider"] div[class*="efbyxod5"],
section[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] > div > div,
section[data-testid="stSidebar"] [data-testid="stSlider"] div[style*="background: linear-gradient"],
section[data-testid="stSidebar"] [data-testid="stSlider"] div[style*="background:linear-gradient"] {
    background: #FFFFFF !important;
    height: 4px !important;
    border-radius: 4px !important;
}

/* Base track container */
section[data-testid="stSidebar"] [data-testid="stSlider"] .react-aria-SliderTrack,
section[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] {
    background: rgba(255, 255, 255, 0.25) !important;
    border-radius: 4px !important;
}

/* 2. Dragging Thumb (the round handle) */
section[data-testid="stSidebar"] [data-testid="stSlider"] .react-aria-SliderThumb,
section[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"],
section[data-testid="stSidebar"] [data-testid="stSlider"] div[class*="efbyxod3"] {
    background-color: #FFFFFF !important;
    border: 3px solid #FF6F5E !important;
    width: 18px !important;
    height: 18px !important;
    border-radius: 50% !important;
    box-shadow: 0 0 10px rgba(255, 255, 255, 0.8), 0 2px 8px rgba(0, 0, 0, 0.35) !important;
    cursor: grab !important;
}

section[data-testid="stSidebar"] [data-testid="stSlider"] .react-aria-SliderThumb:active,
section[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"]:active {
    cursor: grabbing !important;
    transform: scale(1.15) !important;
}

/* 3. Slider Value Numbers (e.g. 90, 0.60, 5, 0.08) */
section[data-testid="stSidebar"] [data-testid="stSlider"] div[class*="efbyxod4"],
section[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stThumbValue"] {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    font-family: 'Inter', sans-serif !important;
}

/* 4. Slider Labels (ELA Quality, Match Ratio, etc.) */
section[data-testid="stSidebar"] [data-testid="stSlider"] label,
section[data-testid="stSidebar"] [data-testid="stSlider"] label p {
    color: #F1EEE8 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

section[data-testid="stSidebar"] .stTooltipIcon svg {
    fill: rgba(241, 238, 232, 0.7) !important;
}

/* ── Tabs (Active tab white text, Inactive tab teal text) ──────── */
[data-testid="stTabs"] [data-baseweb="tab-list"],
.stTabs [data-baseweb="tab-list"] {
    background-color: #E5E0D8;
    border-radius: 14px;
    padding: 5px;
    gap: 4px;
    border: 1px solid rgba(14, 79, 74, 0.05);
}

/* Inactive tabs */
[data-testid="stTab"],
.stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    transition: all 0.25s ease !important;
    border: none !important;
}

[data-testid="stTab"] *,
[data-testid="stTab"] p,
[data-testid="stTab"] span,
.stTabs [data-baseweb="tab"] *,
.stTabs [data-baseweb="tab"] p,
.stTabs [data-baseweb="tab"] span {
    color: #0E4F4A !important;
    font-weight: 500 !important;
}

[data-testid="stTab"]:hover,
.stTabs [data-baseweb="tab"]:hover {
    background-color: rgba(14, 79, 74, 0.08) !important;
}

/* Active tab button background */
[data-testid="stTab"][aria-selected="true"],
[data-testid="stTab"][data-selected="true"],
.stTabs [aria-selected="true"],
.stTabs [data-selected="true"] {
    background-color: #0E4F4A !important;
    box-shadow: 0 2px 10px rgba(14, 79, 74, 0.28) !important;
    border-radius: 10px !important;
}

/* Active tab text - FORCE PURE WHITE on all children */
[data-testid="stTab"][aria-selected="true"] *,
[data-testid="stTab"][data-selected="true"] *,
.stTabs [aria-selected="true"] *,
.stTabs [data-selected="true"] *,
[data-testid="stTab"][aria-selected="true"] p,
[data-testid="stTab"][data-selected="true"] p,
.stTabs [aria-selected="true"] p,
.stTabs [data-selected="true"] p,
[data-testid="stTab"][aria-selected="true"] span,
[data-testid="stTab"][data-selected="true"] span,
.stTabs [aria-selected="true"] span,
.stTabs [data-selected="true"] span {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* Remove default Streamlit tab underlines and highlights */
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}

/* ── Buttons (Coral accent) ───────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #FF6F5E 0%, #E5584A 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    font-family: 'Inter', sans-serif !important;
    box-shadow: 0 4px 14px rgba(255,111,94,0.3);
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #E5584A 0%, #D14D3E 100%) !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(255,111,94,0.4) !important;
}

/* ── File Uploader Styling (Clean & Modern) ────────────────────── */
[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

[data-testid="stFileUploader"] > label {
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    color: #0E4F4A !important;
    margin-bottom: 8px !important;
}

/* Dropzone container */
[data-testid="stFileUploaderDropzone"] {
    background-color: #FAF8F5 !important;
    border: 2px dashed rgba(14, 79, 74, 0.25) !important;
    border-radius: 16px !important;
    padding: 26px 24px !important;
    transition: all 0.25s ease !important;
    text-align: center !important;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #0E4F4A !important;
    background-color: rgba(14, 79, 74, 0.03) !important;
}

/* Browse / Upload button inside dropzone */
[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(135deg, #0E4F4A 0%, #14655E 100%) !important;
    color: #F1EEE8 !important;
    border: none !important;
    border-radius: 9px !important;
    padding: 10px 22px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 2px 8px rgba(14, 79, 74, 0.2) !important;
    transition: all 0.2s ease !important;
    min-height: 40px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 8px !important;
}

[data-testid="stFileUploaderDropzone"] button:hover {
    background: linear-gradient(135deg, #14655E 0%, #0A3B37 100%) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(14, 79, 74, 0.3) !important;
}

[data-testid="stFileUploaderDropzone"] button * {
    color: #F1EEE8 !important;
    fill: #F1EEE8 !important;
}

/* File format instructions */
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: #0A3B37 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    opacity: 0.7 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] * {
    color: #0A3B37 !important;
}

/* ── Metric Cards ─────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background-color: #FAF8F5;
    border: 1px solid #E5E0D8;
    border-radius: 14px !important;
    padding: 20px !important;
    box-shadow: 0 1px 4px rgba(14,79,74,0.06);
    transition: all 0.25s ease;
}
[data-testid="stMetric"]:hover {
    box-shadow: 0 4px 16px rgba(14,79,74,0.1);
    transform: translateY(-2px);
}
[data-testid="stMetricValue"] {
    color: #FF6F5E !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: #0E4F4A !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    font-size: 0.72rem !important;
    letter-spacing: 0.06em;
}

/* ── Slider (main area) ───────────────────────────────────────── */
.stSlider [role="slider"] {
    background-color: #FF6F5E !important;
    border: 2px solid white !important;
}

/* ── Divider ──────────────────────────────────────────────────── */
hr { border-color: #E5E0D8 !important; }

/* ── Images ───────────────────────────────────────────────────── */
[data-testid="stImage"] {
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(14,79,74,0.06);
    transition: box-shadow 0.3s ease;
}
[data-testid="stImage"]:hover {
    box-shadow: 0 4px 16px rgba(14,79,74,0.12);
}
[data-testid="stImage"] > div > p {
    color: #1A6B64 !important;
    text-align: center;
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* ── Scrollbar ────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #F1EEE8; }
::-webkit-scrollbar-thumb { background: #E5E0D8; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #1A6B64; }

/* ── Alerts ───────────────────────────────────────────────────── */
.stAlert {
    border-radius: 10px !important;
    border: none !important;
}

/* ── Animations ───────────────────────────────────────────────── */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
"""


# ──────────────────────────────────────────────────────────────────────
# HTML Components  (NO leading indentation — avoids Markdown code block)
# ──────────────────────────────────────────────────────────────────────

def hero_header(logo_b64: str = "") -> str:
    """Top hero section featuring the real PixelGuard app icon."""
    if logo_b64:
        icon_html = (
            f'<img src="data:image/png;base64,{logo_b64}" alt="PixelGuard Icon" '
            f'style="width:76px;height:76px;border-radius:18px;margin-bottom:14px;'
            f'box-shadow:0 8px 24px rgba(14,79,74,0.22);display:inline-block;" />'
        )
    else:
        icon_html = (
            '<div style="display:inline-flex;align-items:center;justify-content:center;width:60px;height:60px;'
            'background:linear-gradient(135deg,#0E4F4A,#14655E);border-radius:16px;margin-bottom:14px;'
            'box-shadow:0 4px 20px rgba(14,79,74,0.25)">'
            '<svg xmlns="http://www.w3.org/2000/svg" width="30" height="30" viewBox="0 0 24 24" fill="none" '
            'stroke="#F1EEE8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>'
            '<path d="M9 12l2 2 4-4"/></svg></div>'
        )

    return (
f'<div style="text-align:center;padding:28px 20px 20px 20px;animation:fadeIn .6s ease-out">'
f'{icon_html}'
f'<h1 style="font-family:Playfair Display,serif!important;color:#0E4F4A!important;'
f'font-size:2.2rem!important;font-weight:700!important;margin:0 0 8px 0;letter-spacing:-0.02em">'
f'PixelGuard</h1>'
f'<p style="color:#1A6B64;font-size:15px;font-weight:400;max-width:540px;margin:0 auto;'
f'line-height:1.6;opacity:0.8">'
f'Analyze images for signs of digital tampering using '
f'<strong style="color:#0E4F4A">Error Level Analysis</strong>, '
f'<strong style="color:#0E4F4A">Copy-Move Detection</strong>, and '
f'<strong style="color:#0E4F4A">Segmentation</strong>.</p>'
f'</div>'
    )


def verdict_html(verdict: str, ratio: float, matches: int) -> str:
    """Styled verdict card."""
    if verdict == "TAMPERED":
        accent = "#FF6F5E"
        bg = "rgba(255,111,94,0.05)"
        border = "rgba(255,111,94,0.2)"
        badge_bg = "linear-gradient(135deg,#FF6F5E,#E5584A)"
        label = "Tampering Detected"
        desc = "Suspicious regions were found in this image."
        icon_svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" '
                    'fill="none" stroke="#fff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">'
                    '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>'
                    '<line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>')
    else:
        accent = "#0E4F4A"
        bg = "rgba(14,79,74,0.03)"
        border = "rgba(14,79,74,0.12)"
        badge_bg = "linear-gradient(135deg,#0E4F4A,#14655E)"
        label = "Image Appears Authentic"
        desc = "No significant signs of tampering were detected."
        icon_svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" '
                    'fill="none" stroke="#fff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">'
                    '<polyline points="20 6 9 17 4 12"/></svg>')

    return (
f'<div style="background:{bg};border:1.5px solid {border};border-radius:18px;'
f'padding:32px;margin:20px 0;text-align:center;animation:fadeIn .5s ease-out">'
f'<div style="display:inline-flex;align-items:center;gap:8px;background:{badge_bg};'
f'color:#fff;padding:8px 22px;border-radius:50px;font-family:Inter,sans-serif;'
f'font-weight:600;font-size:14px;letter-spacing:0.02em;margin-bottom:14px;'
f'box-shadow:0 4px 14px {accent}33">{icon_svg} {label}</div>'
f'<p style="color:#0A3B37;font-size:14px;margin:0 0 20px 0;opacity:0.65">{desc}</p>'
f'<div style="display:flex;justify-content:center;gap:40px;flex-wrap:wrap">'
f'<div style="text-align:center">'
f'<div style="color:{accent};font-size:28px;font-weight:700;font-family:Inter,sans-serif;line-height:1">'
f'{ratio*100:.2f}%</div>'
f'<div style="color:#0A3B37;font-size:11px;font-weight:600;text-transform:uppercase;'
f'letter-spacing:0.08em;margin-top:6px;opacity:0.45">Tampered Area</div></div>'
f'<div style="width:1px;background:{border};align-self:stretch"></div>'
f'<div style="text-align:center">'
f'<div style="color:{accent};font-size:28px;font-weight:700;font-family:Inter,sans-serif;line-height:1">'
f'{matches}</div>'
f'<div style="color:#0A3B37;font-size:11px;font-weight:600;text-transform:uppercase;'
f'letter-spacing:0.08em;margin-top:6px;opacity:0.45">Copy-Move Matches</div></div>'
f'</div></div>'
    )


def upload_placeholder() -> str:
    """Clean hint card beneath the uploader."""
    return (
'<div style="text-align:center;padding:26px 20px;border-radius:14px;'
'margin:16px auto;max-width:480px;background:rgba(14,79,74,0.02);'
'border:1px solid rgba(14,79,74,0.08);animation:fadeIn .4s ease-out">'
'<p style="color:#0E4F4A;font-size:14px;font-weight:600;margin:0 0 4px 0">'
'Ready for Inspection</p>'
'<p style="color:#1A6B64;font-size:13px;margin:0;opacity:0.7">'
'Upload a photo above to run forensics and detect modifications</p>'
'</div>'
    )


def section_header(title: str, subtitle: str = "") -> str:
    """Section header with coral accent bar."""
    sub = ""
    if subtitle:
        sub = (f'<p style="color:#0A3B37;font-size:13px;margin:4px 0 0 0;'
               f'opacity:0.45;font-weight:400">{subtitle}</p>')
    return (
f'<div style="margin-bottom:16px;animation:fadeIn .4s ease-out">'
f'<h3 style="font-family:Playfair Display,serif!important;color:#0E4F4A!important;'
f'font-size:1.25rem!important;font-weight:600!important;margin:0;'
f'display:flex;align-items:center;gap:10px">'
f'<span style="display:inline-block;width:4px;height:22px;'
f'background:linear-gradient(180deg,#FF6F5E,#E5584A);border-radius:2px"></span>'
f'{title}</h3>{sub}</div>'
    )


# ──────────────────────────────────────────────────────────────────────
# Main Streamlit app
# ──────────────────────────────────────────────────────────────────────
def main():
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
    page_icon = Image.open(logo_path) if os.path.exists(logo_path) else "🛡️"

    st.set_page_config(
        page_title="PixelGuard - Digital Forensics",
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    logo_b64 = get_logo_b64()

    # Inject CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Header ───────────────────────────────────────────────────────
    st.markdown(hero_header(logo_b64), unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

        # Sidebar brand with real app icon
        if logo_b64:
            sidebar_icon_html = (
                f'<img src="data:image/png;base64,{logo_b64}" alt="PixelGuard" '
                f'style="width:52px;height:52px;border-radius:13px;margin-bottom:10px;'
                f'box-shadow:0 4px 14px rgba(0,0,0,0.3);display:inline-block;" />'
            )
        else:
            sidebar_icon_html = (
                '<div style="display:inline-flex;align-items:center;justify-content:center;'
                'width:46px;height:46px;background:rgba(255,111,94,0.12);border-radius:13px;margin-bottom:10px">'
                '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" '
                'stroke="#F1EEE8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>'
                '<path d="M9 12l2 2 4-4"/></svg></div>'
            )

        st.markdown(
f'<div style="text-align:center;margin-bottom:24px;padding-bottom:20px;'
f'border-bottom:1px solid rgba(241,238,232,0.1)">'
f'{sidebar_icon_html}'
f'<p style="color:#F1EEE8!important;font-family:Playfair Display,serif!important;'
f'font-size:17px!important;font-weight:700!important;margin:0;letter-spacing:-0.01em">'
f'PixelGuard</p>'
f'<p style="color:rgba(241,238,232,0.4)!important;font-size:11px!important;'
f'margin:4px 0 0 0!important;font-weight:400!important;letter-spacing:0.04em">'
f'Digital Image Forensics</p>'
f'</div>', unsafe_allow_html=True)

        # Section title
        st.markdown(
'<div style="margin-bottom:16px;padding-bottom:10px;'
'border-bottom:1px solid rgba(241,238,232,0.08)">'
'<p style="font-family:Inter,sans-serif!important;color:rgba(241,238,232,0.5)!important;'
'font-size:10px!important;font-weight:600!important;text-transform:uppercase;'
'letter-spacing:0.14em;margin:0">'
'<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" '
'stroke="rgba(241,238,232,0.5)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
'style="vertical-align:-1px;margin-right:5px">'
'<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 '
'2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 '
'2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 '
'2 2 0 0 1 0-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 '
'2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 '
'2 2 0 0 1 2.83 0l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 '
'2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 '
'2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 '
'2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>'
'Analysis Parameters</p></div>', unsafe_allow_html=True)

        ela_quality = st.slider(
            "ELA Quality", min_value=50, max_value=100, value=90,
            help="JPEG re-save quality for ELA. Lower = more sensitive."
        )
        cm_ratio = st.slider(
            "Match Ratio", min_value=0.3, max_value=0.9, value=0.6, step=0.05,
            help="Lowe's ratio test. Lower = stricter matching."
        )
        min_matches = st.slider(
            "Min Copy-Move Matches", min_value=3, max_value=30, value=5,
            help="Minimum matched pairs to flag copy-move."
        )
        ratio_threshold = st.slider(
            "Tampered Area Threshold", min_value=0.010, max_value=0.200,
            value=0.075, step=0.005, format="%.3f",
            help="Min fraction of tampered pixels to flag image."
        )

        # Footer
        st.markdown(
'<div style="text-align:center;padding:16px 0 4px 0;margin-top:20px;'
'border-top:1px solid rgba(241,238,232,0.06)">'
'<p style="color:rgba(241,238,232,0.25)!important;font-size:11px!important;'
'margin:0;font-weight:400;letter-spacing:0.03em">'
'Built with PixelGuard v1.0</p></div>', unsafe_allow_html=True)

    # ── File uploader ────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload Image for Analysis",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Drag and drop or browse from your device",
    )

    if uploaded is None:
        st.markdown(upload_placeholder(), unsafe_allow_html=True)
        return

    # ── Save uploaded file to temp location ──────────────────────────
    suffix = os.path.splitext(uploaded.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    # ── Format check / warnings ──────────────────────────────────────
    if suffix.lower() == ".png":
        st.warning("ELA works best on JPEG images with compression history. Results on PNG files may be less reliable.")

    # ── Run analysis ─────────────────────────────────────────────────
    try:
        with st.spinner("Analyzing image..."):
            result = analyze_image(
                tmp_path,
                ela_quality=ela_quality,
                ratio=cm_ratio,
                min_matches=min_matches,
                ratio_threshold=ratio_threshold,
            )
    except Exception as e:
        st.error(f"Could not analyze the uploaded file: {e}")
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return

    # Clean up temp file
    try:
        os.unlink(tmp_path)
    except OSError:
        pass

    # ── Verdict ──────────────────────────────────────────────────────
    st.markdown(
        verdict_html(result["verdict"], result["tampered_ratio"], result["num_matches"]),
        unsafe_allow_html=True,
    )

    # ── Metrics row ──────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Verdict", result["verdict"])
    with col2:
        st.metric("Tampered Area", f"{result['tampered_ratio']*100:.2f}%")
    with col3:
        st.metric("Copy-Move Matches", result["num_matches"])

    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

    # ── Tabs for detailed results ────────────────────────────────────
    tab_ela, tab_cm, tab_seg, tab_final = st.tabs([
        "ELA Analysis",
        "Copy-Move Detection",
        "Segmentation",
        "Final Result",
    ])

    # ── Tab 1: ELA ───────────────────────────────────────────────────
    with tab_ela:
        st.markdown(
            section_header("Error Level Analysis",
                           "Highlights regions with different compression levels"),
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(result["original"], caption="Original", use_container_width=True)
        with c2:
            st.image(result["ela_image"], caption="ELA (Color)", use_container_width=True)
        with c3:
            st.image(result["ela_gray"], caption="ELA (Grayscale)",
                     use_container_width=True, clamp=True)

    # ── Tab 2: Copy-Move ─────────────────────────────────────────────
    with tab_cm:
        st.markdown(
            section_header("Copy-Move Detection",
                           "Identifies regions that have been cloned within the image"),
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            st.image(result["cm_vis"],
                     caption=f"Matched Pairs ({result['num_matches']})",
                     use_container_width=True)
        with c2:
            st.image(result["cm_mask"], caption="Copy-Move Mask",
                     use_container_width=True, clamp=True)

    # ── Tab 3: Segmentation ──────────────────────────────────────────
    with tab_seg:
        st.markdown(
            section_header("Segmentation Masks",
                           "Binary masks isolating suspected tampered regions"),
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(result["ela_mask"], caption="ELA Mask",
                     use_container_width=True, clamp=True)
        with c2:
            st.image(result["cm_mask"], caption="Copy-Move Mask",
                     use_container_width=True, clamp=True)
        with c3:
            st.image(result["final_mask"], caption="Combined Mask",
                     use_container_width=True, clamp=True)

    # ── Tab 4: Final ─────────────────────────────────────────────────
    with tab_final:
        st.markdown(
            section_header("Final Result",
                           "Side-by-side comparison with tampered region highlights"),
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(result["original"], caption="Original",
                     use_container_width=True)
        with c2:
            st.image(result["overlay"], caption="Overlay (Tampered Regions)",
                     use_container_width=True)
        with c3:
            st.image(result["bbox_image"], caption="Bounding Boxes",
                     use_container_width=True)


if __name__ == "__main__":
    main()
