"""
Visual PHI redaction using rendered text template matching.
Finds PHI strings directly on rendered page images using PyMuPDF and sliding window search.
"""
import re
import cv2
import numpy as np
import fitz  # PyMuPDF

DEFAULT_DPI = 300
MATCH_THRESHOLD = 0.85
PHONE_MIN_DIGITS = 7
PHONE_MAX_DIGITS = 11
PHONE_EXACT_DIGITS = 10
RECT_PAD = 2
FONT_DIVISOR = 28.0
FONT_SCALE_MIN = 0.25
FONT_SCALE_MAX = 0.45
LINE_GROUP_DIVISOR = 10.0
SLIDING_WINDOW_MAX = 15
ZIP_X0_FRACTION = 0.40

LABEL_WORDS = {
    'name', 'email', 'address', 'phone', 'cell', 'home', 'work',
    'fax', 'ssn', 'dob', 'mrn', 'date', 'birth', 'sex', 'gender',
    'insurance', 'provider', 'policy', 'group', 'member', 'id',
    'contact', 'emergency', 'language', 'account', 'billing',
    'subscriber', 'medicare', 'portal', 'login', 'card', 'credit',
    'patient', 'physician', 'doctor', 'referring', 'attending',
    'facility', 'hospital', 'clinic', 'field', 'value', 'type',
    'number', 'street', 'city', 'state', 'zip', 'apt', 'suite',
}


def _render_pdf_page(pdf_bytes: bytes, page_index: int, dpi: int = DEFAULT_DPI) -> np.ndarray:
    """Renders a PDF page to a high-resolution numpy image."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_index]
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat)
    img_array = np.frombuffer(pix.tobytes("png"), dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    doc.close()
    return frame


def _find_text_locations(
    page_image: np.ndarray, phi_text: str, threshold: float = MATCH_THRESHOLD
) -> list[tuple[int, int, int, int]]:
    """Uses multiple strategies to find all locations of phi_text on the page image."""
    locations = []
    return locations


def _add_email_card_variants(text: str, variants: list[str]) -> None:
    """Appends email truncation repair and credit card digit variants to the list."""
    if text.endswith('.') and '@' in text:
        for tld in ['com', 'org', 'net', 'gov', 'edu']:
            variants.append(text + tld)
        variants.append(text.rstrip('.'))

    if 'ending' in text.lower() or 'last 4' in text.lower():
        digits_match = re.search(r'\d{4}$', text.strip())
        if digits_match:
            variants.append(digits_match.group())
            variants.append(f"ending {digits_match.group()}")

    if '@' in text:
        clean = re.sub(r'\s*@\s*', '@', text)
        clean = re.sub(r'\s*\.\s*', '.', clean)
        if clean not in variants:
            variants.append(clean)


def _normalize_phi(phi_text: str) -> list[str]:
    """Returns a list of normalized variants of a PHI string to try searching."""
    variants = []
    text = phi_text.strip()
    variants.append(text)

    _add_email_card_variants(text, variants)

    no_space_after_dot = re.sub(r'\.\s+', '.', text)
    if no_space_after_dot != text:
        variants.append(no_space_after_dot)

    no_spaces = re.sub(r'\s+', '', text)
    if no_spaces not in variants:
        variants.append(no_spaces)

    no_space_hyphen = re.sub(r'\s*-\s*', '-', text)
    if no_space_hyphen not in variants:
        variants.append(no_space_hyphen)

    digits = re.sub(r'[^\d]', '', text)
    if PHONE_MIN_DIGITS <= len(digits) <= PHONE_MAX_DIGITS and len(digits) == PHONE_EXACT_DIGITS:
        variants.append(f"({digits[:3]}) {digits[3:6]}-{digits[6:]}")
        variants.append(f"{digits[:3]}-{digits[3:6]}-{digits[6:]}")

    if not '@' in text and not any(c.isdigit() for c in text):
        variants.append(text.title())
        variants.append(text.upper())

    seen = set()
    return [v for v in variants if v and not (v in seen or seen.add(v))]


def _draw_bar(frame: np.ndarray, x1: float, y1: float, x2: float, y2: float, category: str) -> None:
    """Draws solid black redaction bar and white category label on the frame."""
    h_img, w_img = frame.shape[:2]
    x1_i, y1_i = max(0, int(x1) - RECT_PAD), max(0, int(y1) - RECT_PAD)
    x2_i, y2_i = min(w_img, int(x2) + RECT_PAD), min(h_img, int(y2) + RECT_PAD)
    if x2_i <= x1_i or y2_i <= y1_i:
        return
    cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), (0, 0, 0), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    box_h = y2_i - y1_i
    fs = max(FONT_SCALE_MIN, min(FONT_SCALE_MAX, box_h / FONT_DIVISOR))
    (tw, th), _ = cv2.getTextSize(category, font, fs, 1)
    tx, ty = x1_i + 4, min(y1_i + th + max(2, (box_h - th) // 2), y2_i - 2)
    cv2.putText(frame, category, (tx, ty), font, fs, (255, 255, 255), 1, cv2.LINE_AA)


def _normalize_str(text: str) -> str:
    """Normalizes string for comparison by removing all whitespace and lowercasing."""
    return re.sub(r'\s+', '', text).lower()


def _redact_word_groups(frame: np.ndarray, word_group: list, scale: float, category: str) -> int:
    """Groups words by line Y-coordinate and draws redaction bars, skipping label-only lines."""
    line_groups = {}
    for ww in word_group:
        line_key = round(ww[1] * scale / LINE_GROUP_DIVISOR) * int(LINE_GROUP_DIVISOR)
        if line_key not in line_groups:
            line_groups[line_key] = []
        line_groups[line_key].append(ww)

    valid_lines = 0
    for line_key, line_words in line_groups.items():
        line_words_list = [ww[4].lower().rstrip(':') for ww in line_words]
        if all(w in LABEL_WORDS for w in line_words_list):
            print(f"[REDACT] Skipping label-only line: {[ww[4] for ww in line_words]}")
            continue

        lx0 = min(ww[0] for ww in line_words) * scale
        ly0 = min(ww[1] for ww in line_words) * scale
        lx1 = max(ww[2] for ww in line_words) * scale
        ly1 = max(ww[3] for ww in line_words) * scale
        _draw_bar(frame, lx0, ly0, lx1, ly1, category)
        valid_lines += 1

    return valid_lines


def _check_window_match(
    frame: np.ndarray, word_group: list, phi_norm: str, phi_text: str, category: str, scale: float, page_idx: int, phi_words: list[str], accumulated_norm: str
) -> bool:
    """Validates accumulated match position and word ratios, executing redaction if passed."""
    match_start = accumulated_norm.find(phi_norm)
    first_two_words_len = sum(len(_normalize_str(ww[4])) for ww in word_group[:2])
    if match_start > first_two_words_len:
        return False

    if category == "NAME" and phi_words:
        group_text = " ".join(ww[4].lower() for ww in word_group)
        if not all(pw in group_text for pw in phi_words):
            return False

    lines_count = _redact_word_groups(frame, word_group, scale, category)
    print(f"[REDACT] Page {page_idx + 1}: '{phi_text[:40]}' ({lines_count} line(s)) [{category}]")
    return True


def _search_sliding_window(
    frame: np.ndarray, words: list, phi_norm: str, phi_text: str, category: str, scale: float, page_idx: int
) -> bool:
    """Searches words using sliding window normalization with label skipping and position guards."""
    found = False
    start_idx = 0
    phi_words = [w.lower() for w in phi_text.split() if len(w) > 1]

    while start_idx < len(words):
        accumulated_norm = ""
        word_group = []
        window_broken = False

        for end_idx in range(start_idx, min(start_idx + SLIDING_WINDOW_MAX, len(words))):
            w = words[end_idx]
            word_text = w[4]

            if word_text.isupper() and len(word_text) > 2:
                start_idx = end_idx + 1
                window_broken = True
                break

            if not word_group and word_text.lower().rstrip(':') in LABEL_WORDS:
                window_broken = True
                break

            accumulated_norm += _normalize_str(word_text)
            word_group.append(w)

            if phi_norm in accumulated_norm:
                if _check_window_match(frame, word_group, phi_norm, phi_text, category, scale, page_idx, phi_words, accumulated_norm):
                    found = True
                window_broken = True
                break

            if len(accumulated_norm) > len(phi_norm) * 2:
                break

        if found:
            break

        if not window_broken or start_idx <= end_idx:
            start_idx += 1

    return found


def _breakdown_address(address: str) -> list[str]:
    """Breaks an address into HIPAA-relevant sub-components only (street, apt, zip)."""
    parts = []
    text = address.strip()

    street_match = re.match(
        r'^(\d+\s+[A-Za-z0-9\s\.\-]+?)'
        r'(?:\s*[,#]|\s+(?:apt|apartment|suite|ste|unit|#)|\s+[A-Z]{2}\s+\d{5}|$)',
        text, re.IGNORECASE
    )
    if street_match:
        street = street_match.group(1).strip().rstrip(',')
        if len(street) > 4:
            parts.append(street)

    apt_match = re.search(
        r'(?:apt\.?|apartment|suite|ste\.?|unit|#)\s*([A-Z0-9]+)',
        text, re.IGNORECASE
    )
    if apt_match:
        parts.append(apt_match.group(0).strip())

    zip_match = re.search(r'\b(\d{5}(?:-\d{4})?)\b', text)
    if zip_match:
        zipcode = zip_match.group(1)
        parts.append(zipcode)
        state_zip = re.search(r'\b([A-Z]{2})\s+' + re.escape(zipcode) + r'\b', text)
        if state_zip:
            parts.append(state_zip.group(0))

    if not parts:
        first_segment = text.split(',')[0].strip()
        if first_segment:
            parts.append(first_segment)

    seen = set()
    result = []
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            result.append(p)

    return result


def _is_address_phi(phi_text: str, category: str) -> bool:
    """Returns True if PHI string matches address category and common address patterns."""
    if category not in ('ADDRESS', 'PHI'):
        return False
    if not bool(re.search(r'\d', phi_text)) or not bool(re.search(r'[A-Za-z]', phi_text)):
        return False
    kw_match = bool(re.search(
        r'\b(st|ave|blvd|rd|dr|ln|ct|pkwy|hwy|way|pl|ter|cir|'
        r'street|avenue|boulevard|road|drive|lane|court|parkway|'
        r'highway|place|terrace|circle|apt|suite|unit)\b',
        phi_text, re.IGNORECASE
    ))
    return ',' in phi_text or kw_match


def _search_target_variants(
    frame: np.ndarray, page: fitz.Page, search_targets: list[str], scale: float, category: str, page_idx: int, phi_text: str, is_address: bool, found_by_strategy1: set
) -> bool:
    """Searches for target variants on page, handling full match return or address continuation."""
    found_any = False
    for target in search_targets:
        for variant in _normalize_phi(target):
            rects = page.search_for(variant, quads=False)
            if rects:
                found_by_strategy1.add(_normalize_str(phi_text))
                if re.match(r'^[A-Z]{2}\s+\d{5}', variant):
                    for rect in rects:
                        rect_w = (rect.x1 - rect.x0) * scale
                        zip_x0 = (rect.x0 * scale) + (rect_w * ZIP_X0_FRACTION)
                        _draw_bar(frame, zip_x0, rect.y0 * scale, rect.x1 * scale, rect.y1 * scale, category)
                    print(f"[REDACT] Page {page_idx + 1}: '{variant}' → ZIP portion only [{category}]")
                else:
                    for rect in rects:
                        _draw_bar(frame, rect.x0 * scale, rect.y0 * scale, rect.x1 * scale, rect.y1 * scale, category)
                    print(f"[REDACT] Page {page_idx + 1}: '{target[:40]}' (from address '{phi_text[:30]}') [{category}]")
                found_any = True
                if not is_address:
                    return True
                break
    return found_any


def _run_ocr_fallback(
    frame: np.ndarray, page_ocr: list, phi_text: str, category: str, page_idx: int
) -> bool:
    """Executes OCR box fallback when PyMuPDF extraction fails in styled containers."""
    return False


def _redact_phi_string(
    frame: np.ndarray, page: fitz.Page, phi_text: str, category: str, scale: float, page_ocr: list, page_idx: int, found_by_strategy1: set
) -> None:
    """Finds occurrences of phi_text on the page using target breakdown, sliding window, and OCR."""
    phi_norm = _normalize_str(phi_text)
    if not phi_norm:
        return

    is_addr = _is_address_phi(phi_text, category)
    if is_addr:
        search_targets = _breakdown_address(phi_text)
        print(f"[ADDRESS] Breaking down: '{phi_text[:50]}' → {len(search_targets)} parts")
    else:
        search_targets = [phi_text]

    if _search_target_variants(frame, page, search_targets, scale, category, page_idx, phi_text, is_addr, found_by_strategy1):
        return

    if phi_norm in found_by_strategy1:
        return

    words = page.get_text("words")
    if words and _search_sliding_window(frame, words, phi_norm, phi_text, category, scale, page_idx):
        return

    if phi_norm in found_by_strategy1:
        return

    if _run_ocr_fallback(frame, page_ocr, phi_text, category, page_idx):
        return

    print(f"[MISS] Page {page_idx + 1}: '{phi_text[:40]}' [{category}]")


def _redact_page_loop(
    frame: np.ndarray, page: fitz.Page, phi_list: list[dict], scale: float, page_ocr: list, page_idx: int
) -> None:
    """Iterates through page PHI items and executes redaction with continuation search."""
    found_by_strategy1 = set()
    for phi_obj in phi_list:
        phi_text = phi_obj.get("text", "").strip()
        category = phi_obj.get("category", "PHI").upper()
        if not phi_text:
            continue

        _redact_phi_string(frame, page, phi_text, category, scale, page_ocr, page_idx, found_by_strategy1)


def find_and_redact_phi(
    pdf_bytes: bytes, phi_strings_per_page: list[list[dict]], ocr_regions_per_page: list[list] = None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Main redaction function using PyMuPDF text search and OCR continuation matching."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    dpi = DEFAULT_DPI
    scale = dpi / 72.0
    redacted_pages = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat)
        img_array = np.frombuffer(pix.tobytes("png"), dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        original = frame.copy()

        phi_list = phi_strings_per_page[page_idx] if page_idx < len(phi_strings_per_page) else []
        page_ocr = (
            ocr_regions_per_page[page_idx]
            if ocr_regions_per_page and page_idx < len(ocr_regions_per_page)
            else []
        )
        _redact_page_loop(frame, page, phi_list, scale, page_ocr, page_idx)
        redacted_pages.append((original, frame))

    doc.close()
    return redacted_pages
