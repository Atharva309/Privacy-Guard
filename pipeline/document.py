"""
Clinical document processing and Apple Vision OCR module.
Converts uploaded document bytes to frames and executes native macOS Neural Engine OCR.
"""
import typing
import cv2
import numpy as np
from Foundation import NSData
import Vision
from pipeline.redactor import BoundingBox
from pipeline.ocr import TextRegion

DEFAULT_DPI = 200
MIN_TEXT_HEIGHT = 0.005
MIN_CONFIDENCE_DOC = 0.2
MIN_CONFIDENCE_WORD = 0.3
VERT_ADJ_MIN = 5
VERT_ADJ_MAX = 35
HORIZ_TOLERANCE = 20
PARAGRAPH_RATIO_THRESHOLD = 0.6
WORD_CROP_SCALE = 3.0
REGION_PAD = 4
BAR_PAD = 2
FONT_SCALE_DIVISOR = 30.0


def _document_to_frames(file_bytes: bytes, ext: str) -> list[np.ndarray]:
    """Converts uploaded document bytes to list of numpy image arrays."""
    if ext == "pdf":
        try:
            import fitz
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            frames = []
            for page in doc:
                pix = page.get_pixmap(dpi=DEFAULT_DPI)
                img_array = np.frombuffer(pix.tobytes("png"), dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if frame is not None:
                    frames.append(frame)
            return frames
        except Exception as error:
            print(f"[REDACT] PDF conversion failed: {error}")
            return []
    else:
        np_arr = np.frombuffer(file_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return [frame] if frame is not None else []


def _exec_vision_request(buf: np.ndarray) -> typing.Any:
    """Initializes and executes the Apple Vision OCR request."""
    ns_data = NSData.dataWithBytes_length_(buf.tobytes(), len(buf.tobytes()))
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(ns_data, {})
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(True)
    request.setMinimumTextHeight_(MIN_TEXT_HEIGHT)
    try:
        handler.performRequests_error_([request], None)
        return request
    except Exception as error:
        print(f"[REDACT] Vision OCR document scan failed: {error}")
        return None


def _run_vision_ocr_document(frame: np.ndarray) -> list[TextRegion]:
    """Runs Apple Vision OCR on a static document image with high accuracy."""
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    success, buf = cv2.imencode('.png', rgb)
    if not success:
        return []

    request = _exec_vision_request(buf)
    if not request:
        return []

    results = []
    for obs in (request.results() or []):
        candidates = obs.topCandidates_(1)
        if not candidates:
            continue
        text = str(candidates[0].string())
        conf = float(candidates[0].confidence())
        if conf < MIN_CONFIDENCE_DOC or not text.strip():
            continue

        bbox = obs.boundingBox()
        x1 = int(bbox.origin.x * w)
        y1 = int((1.0 - bbox.origin.y - bbox.size.height) * h)
        x2 = int((bbox.origin.x + bbox.size.width) * w)
        y2 = int((1.0 - bbox.origin.y) * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 > x1 and y2 > y1:
            results.append(TextRegion(text, BoundingBox(x1, y1, x2, y2, conf, "text")))

    return results


def _expand_multiline_boxes(pii_boxes: list[BoundingBox], all_regions: list[TextRegion]) -> list[BoundingBox]:
    """Expands redaction to cover multi-line text continuations with category preservation."""
    expanded = list(pii_boxes)
    redacted_ids = set(id(b) for b in pii_boxes)

    for box in pii_boxes:
        for region in all_regions:
            rb = region.box
            vertically_adjacent = (
                rb.y1 >= box.y2 - VERT_ADJ_MIN and
                rb.y1 <= box.y2 + VERT_ADJ_MAX and
                rb.x1 <= box.x2 + HORIZ_TOLERANCE and
                rb.x2 >= box.x1 - HORIZ_TOLERANCE
            )
            if vertically_adjacent and id(rb) not in redacted_ids:
                rb.category = getattr(box, "category", "PHI")
                expanded.append(rb)
                redacted_ids.add(id(rb))

    return expanded


def _is_paragraph_line(region_text: str, phi_text: str) -> bool:
    """Returns True if OCR region is a paragraph line containing PHI as a small part."""
    region_stripped = region_text.strip()
    phi_stripped = phi_text.strip()

    if len(phi_stripped) >= len(region_stripped) * PARAGRAPH_RATIO_THRESHOLD:
        return False

    return True


def _run_crop_vision_request(crop: np.ndarray, scale: float) -> typing.Any:
    """Encodes cropped region and executes targeted word-level Vision OCR request."""
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    enlarged = cv2.resize(rgb, (int(rgb.shape[1] * scale), int(rgb.shape[0] * scale)))
    success, buf = cv2.imencode('.png', enlarged)
    if not success:
        return None, 0, 0

    ns_data = NSData.dataWithBytes_length_(buf.tobytes(), len(buf.tobytes()))
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(ns_data, {})
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(False)

    try:
        handler.performRequests_error_([request], None)
        return request, enlarged.shape[0], enlarged.shape[1]
    except Exception as err:
        print(f"[REDACT] Word OCR Failed: {err}")
        return None, 0, 0


def _get_word_boxes_for_region(frame: np.ndarray, box: BoundingBox, phi_text: str) -> list[BoundingBox]:
    """Runs targeted word-level OCR on a specific region to find tight boxes for PHI words."""
    h, w = frame.shape[:2]
    y1, y2 = max(0, box.y1 - REGION_PAD), min(h, box.y2 + REGION_PAD)
    x1, x2 = max(0, box.x1), min(w, box.x2)
    if x2 <= x1 or y2 <= y1:
        return []

    request, crop_h, crop_w = _run_crop_vision_request(frame[y1:y2, x1:x2], WORD_CROP_SCALE)
    if not request:
        return []

    phi_lower = phi_text.lower().strip()
    phi_words = [w.strip('.,;:()') for w in phi_lower.split() if len(w.strip('.,;:()')) > 1]
    word_boxes = []

    for obs in (request.results() or []):
        candidates = obs.topCandidates_(1)
        if not candidates:
            continue
        word_text = str(candidates[0].string()).strip().lower()
        conf = float(candidates[0].confidence())
        if conf < MIN_CONFIDENCE_WORD:
            continue

        word_clean = word_text.strip('.,;:() ')
        if not any(word_clean in phi_w or phi_w in word_clean for phi_w in phi_words if len(phi_w) > 1):
            continue

        bbox = obs.boundingBox()
        wx1 = int((bbox.origin.x * crop_w) / WORD_CROP_SCALE) + x1
        wy1 = int(((1.0 - bbox.origin.y - bbox.size.height) * crop_h) / WORD_CROP_SCALE) + y1
        wx2 = int(((bbox.origin.x + bbox.size.width) * crop_w) / WORD_CROP_SCALE) + x1
        wy2 = int(((1.0 - bbox.origin.y) * crop_h) / WORD_CROP_SCALE) + y1
        wx1, wy1 = max(0, wx1 - BAR_PAD), max(0, wy1 - BAR_PAD)
        wx2, wy2 = min(w, wx2 + BAR_PAD), min(h, wy2 + BAR_PAD)

        if wx2 > wx1 and wy2 > wy1:
            word_boxes.append(BoundingBox(wx1, wy1, wx2, wy2, conf, getattr(box, 'category', 'PHI')))

    return word_boxes


def _draw_redaction_bar(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, category: str) -> None:
    """Draws solid black redaction bar with white category label overlay."""
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    box_h = y2 - y1
    font_scale = max(0.25, min(0.4, box_h / FONT_SCALE_DIVISOR))
    thickness = 1
    (tw, th), _ = cv2.getTextSize(category, font, font_scale, thickness)
    tx = x1 + 4
    ty = min(y1 + th + max(2, (box_h - th) // 2), y2 - 2)
    cv2.putText(frame, category, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _apply_document_redaction(
    frame: np.ndarray, box: BoundingBox, phi_text: str = "", source_frame: np.ndarray = None
) -> None:
    """Applies solid black redaction bar, using targeted word OCR for paragraph lines."""
    h, w = frame.shape[:2]
    category = getattr(box, "category", "PHI").upper()
    region_text = getattr(box, "_source_text", "")

    if phi_text and region_text and _is_paragraph_line(region_text, phi_text) and source_frame is not None:
        word_boxes = _get_word_boxes_for_region(source_frame, box, phi_text)
        if word_boxes:
            for wb in word_boxes:
                _draw_redaction_bar(frame, wb.x1, wb.y1, wb.x2, wb.y2, category)
            return

    x1 = max(0, box.x1 - BAR_PAD)
    y1 = max(0, box.y1 - BAR_PAD)
    x2 = min(w, box.x2 + BAR_PAD)
    y2 = min(h, box.y2 + BAR_PAD)
    _draw_redaction_bar(frame, x1, y1, x2, y2, category)
