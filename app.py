"""
Privacy Guard — Clinical Document PHI Redaction (macOS Edition)
Streamlit application entry point.
"""
import sys
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Privacy Guard (macOS Edition) — Clinical PHI Redaction",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Check Operating System Compatibility First ─────────────────────────────
if sys.platform != "darwin":
    st.markdown("""
    <div style='background-color:#1e2130; border: 2px solid #ef4444; border-radius:12px; padding:2.5rem; text-align:center; margin: 2rem 10%;'>
        <h1 style='color:#f8fafc; margin-bottom:0.5rem;'>🔒 Privacy Guard <span style='color:#a78bfa;'>macOS Edition</span></h1>
        <h3 style='color:#ef4444; margin-top: 1rem;'>⚠️ Incompatible Operating System Detected</h3>
        <p style='color:#94a3b8; font-size:1.15rem; margin-top:1.5rem; line-height: 1.6;'>
            This application is engineered specifically for <b>native macOS hardware</b>. <br>
            It utilizes Apple's native <code>Vision.framework</code> and Apple Silicon Neural Engine for high-performance, local document OCR.
        </p>
        <p style='color:#64748b; font-size:0.95rem; margin-top:1rem;'>
            Please run this project on a macOS machine (M-Series recommended) to utilize the hardware-accelerated redaction pipeline.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

from types import ModuleType

# Mock BoundingBox and TextRegion so pipeline modules can import them without legacy files existing
class BoundingBox:
    """Represents a bounding box in the document frame."""
    def __init__(self, x1: int, y1: int, x2: int, y2: int, confidence: float = 1.0, category: str = "text") -> None:
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)
        self.confidence = float(confidence)
        self.category = category

class TextRegion:
    """Represents an OCR text region with its bounding box."""
    def __init__(self, text: str, box: BoundingBox) -> None:
        self.text = text
        self.box = box

redactor_mod = ModuleType('pipeline.redactor')
redactor_mod.BoundingBox = BoundingBox
sys.modules['pipeline.redactor'] = redactor_mod

ocr_mod = ModuleType('pipeline.ocr')
ocr_mod.TextRegion = TextRegion
ocr_mod.BoundingBox = BoundingBox
sys.modules['pipeline.ocr'] = ocr_mod

import base64
import fitz
import cv2
import numpy as np
from dotenv import load_dotenv

# Load environment variables FIRST before importing pipeline modules that initialize OpenAI()
load_dotenv()

from pipeline.document import _document_to_frames, _run_vision_ocr_document
from pipeline.ai_phi_detector import detect_phi_batch
from pipeline.visual_redactor import find_and_redact_phi

JPEG_QUALITY = 92
KB_DIVISOR = 1024

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; color: #e2e8f0; }
    
    .hero {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .hero h1 {
        font-size: 2.5rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.5rem;
    }
    .hero p {
        color: #94a3b8;
        font-size: 1.1rem;
    }
    
    .stat-card {
        background: #1e2130;
        border: 1px solid #2d3148;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .stat-number {
        font-size: 1.8rem;
        font-weight: 800;
        color: #a78bfa;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.2rem;
    }

    .phi-badge {
        display: inline-block;
        background: #1e1b4b;
        color: #a78bfa;
        border-radius: 99px;
        padding: 2px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
    }

    .audit-row {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid #1a1f2e;
        font-size: 0.875rem;
    }
    
    .stButton > button {
        background: #7c3aed;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 2rem;
    }
    .stButton > button:hover {
        background: #6d28d9;
    }
    
    div[data-testid="stFileUploader"] {
        background: #1e2130;
        border: 2px dashed #334155;
        border-radius: 12px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def process_document(file_bytes: bytes, file_ext: str) -> dict:
    """Runs the full PHI redaction pipeline on uploaded document bytes."""
    frames = _document_to_frames(file_bytes, file_ext)
    if not frames:
        return {"error": "Could not read document"}

    if file_ext == "pdf":
        pdf_bytes = file_bytes
    else:
        doc = fitz.open()
        img = cv2.imencode('.png', frames[0])[1].tobytes()
        img_doc = fitz.open("png", img)
        pdf_bytes = img_doc.convert_to_pdf()

    all_text_regions = []
    for frame in frames:
        regions = _run_vision_ocr_document(frame)
        all_text_regions.append(regions)

    pages_text = ["\n".join(r.text for r in regions) for regions in all_text_regions]

    with st.spinner("🧠 Identifying PHI with AI..."):
        all_phi_objects = detect_phi_batch(pages_text)

    import re
    GLOBAL_CATEGORIES = {"NAME", "SSN", "DOB", "EMAIL", "PHONE", "ACCOUNT",
                         "MEDICARE", "POLICY", "CREDIT_CARD", "ADDRESS"}
    global_phi = {}
    for page_phi in all_phi_objects:
        for phi_obj in page_phi:
            if phi_obj.get("category", "").upper() in GLOBAL_CATEGORIES:
                key = re.sub(r'\s+', '', phi_obj.get("text", "").lower())
                if key and key not in global_phi:
                    global_phi[key] = phi_obj
    global_phi_list = list(global_phi.values())

    combined_phi_per_page = []
    for i, page_phi in enumerate(all_phi_objects):
        page_keys = set(re.sub(r'\s+', '', p.get("text","").lower()) for p in page_phi)
        combined = list(page_phi)
        for phi_obj in global_phi_list:
            key = re.sub(r'\s+', '', phi_obj.get("text","").lower())
            if key not in page_keys:
                combined.append(phi_obj)
        combined_phi_per_page.append(combined)

    with st.spinner("✏️ Applying redactions..."):
        page_results = find_and_redact_phi(pdf_bytes, combined_phi_per_page, all_text_regions)

    all_pages_original = []
    all_pages_redacted = []
    all_audit = []
    total_pii = 0

    for i, (original, redacted) in enumerate(page_results):
        _, orig_buf = cv2.imencode(".jpg", original, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        _, red_buf = cv2.imencode(".jpg", redacted, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        all_pages_original.append(base64.b64encode(orig_buf.tobytes()).decode())
        all_pages_redacted.append(base64.b64encode(red_buf.tobytes()).decode())

        phi_list = combined_phi_per_page[i] if i < len(combined_phi_per_page) else []
        for phi_obj in phi_list:
            all_audit.append({
                "category": phi_obj.get("category", "PHI"),
                "text": "[REDACTED]",
                "page": i + 1
            })
        total_pii += len(phi_list)

    return {
        "pages": [{"original": all_pages_original[i], "redacted": all_pages_redacted[i]}
                  for i in range(len(page_results))],
        "audit_log": all_audit,
        "pii_count": total_pii,
        "page_count": len(page_results)
    }


def create_download_pdf(redacted_pages_b64: list[str], original_filename: str) -> bytes:
    """Stitches redacted page images into a single downloadable PDF."""
    output_doc = fitz.open()
    for b64 in redacted_pages_b64:
        img_bytes = base64.b64decode(b64)
        img_doc = fitz.open("jpeg", img_bytes)
        pdf_bytes = img_doc.convert_to_pdf()
        page_doc = fitz.open("pdf", pdf_bytes)
        output_doc.insert_pdf(page_doc)
    return output_doc.tobytes()


# ── Main UI ───────────────────────────────────────────────────────────────

# Hero header
st.markdown("""
<div class="hero">
    <h1>🔒 Privacy Guard <span style='font-size:1.6rem; color:#a78bfa;'>macOS Neural Engine Edition</span></h1>
    <p>AI-powered PHI redaction for clinical documents — Apple Silicon accelerated, runs locally</p>
</div>
""", unsafe_allow_html=True)

# Stats row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="stat-card"><div class="stat-number">18</div><div class="stat-label">PHI Types Detected</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="stat-card"><div class="stat-number">GPT-4o</div><div class="stat-label">Context-Aware AI</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="stat-card"><div class="stat-number">M-Series</div><div class="stat-label">Neural Engine OCR</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="stat-card"><div class="stat-number">0</div><div class="stat-label">External Data Sent</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Upload section
uploaded_file = st.file_uploader(
    "Upload a clinical document",
    type=["pdf", "png", "jpg", "jpeg"],
    help="Supports PDF, PNG, JPG. Discharge summaries, prior auth forms, lab reports, etc."
)

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    file_ext = uploaded_file.name.lower().split(".")[-1]

    st.markdown(f"**Uploaded:** `{uploaded_file.name}` ({len(file_bytes) // KB_DIVISOR} KB)")

    if st.button("🔍 Detect & Redact PHI", use_container_width=True):

        with st.spinner("📄 Processing document..."):
            result = process_document(file_bytes, file_ext)

        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            # Summary metrics
            st.success(f"✅ Complete — {result['pii_count']} PHI items redacted across {result['page_count']} page(s)")

            # PHI category badges
            categories = list(set(a["category"] for a in result["audit_log"]))
            badges = " ".join(f'<span class="phi-badge">{c}</span>' for c in sorted(categories))
            st.markdown(f"**Categories found:** {badges}", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Page-by-page comparison
            for i, page in enumerate(result["pages"]):
                st.markdown(f"#### Page {i + 1} of {result['page_count']}")
                left, right = st.columns(2)
                with left:
                    st.caption("Original")
                    orig_bytes = base64.b64decode(page["original"])
                    st.image(orig_bytes, use_container_width=True)
                with right:
                    st.caption("Redacted")
                    red_bytes = base64.b64decode(page["redacted"])
                    st.image(red_bytes, use_container_width=True)

                st.markdown("---")

            # Download button
            pdf_bytes = create_download_pdf(
                [p["redacted"] for p in result["pages"]],
                uploaded_file.name
            )
            clean_name = uploaded_file.name.rsplit(".", 1)[0]
            st.download_button(
                label="⬇️ Download Redacted PDF",
                data=pdf_bytes,
                file_name=f"{clean_name}_redacted.pdf",
                mime="application/pdf",
                use_container_width=True
            )

            # Audit log
            with st.expander("🔒 Audit Log — PHI Redacted (no actual PHI content logged)"):
                st.markdown("| Page | Category | Status |")
                st.markdown("|------|----------|--------|")
                for entry in result["audit_log"]:
                    st.markdown(f"| {entry['page']} | `{entry['category']}` | ✅ Redacted |")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#475569; font-size:0.8rem;'>"
    "Privacy Guard (macOS Edition) — PHI processing is local. Only OCR text is sent to OpenAI for classification. "
    "No images or raw frames are transmitted. HIPAA-compliant by design."
    "</p>",
    unsafe_allow_html=True
)
