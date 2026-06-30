"""
AI-powered PHI detection using OpenAI API.
Identifies patient PII in clinical document text with full clinical context.
"""
import json
from openai import OpenAI
from pipeline.redactor import BoundingBox

_client = OpenAI()

MAX_TOKENS = 1024
TEMPERATURE = 0.0
MAX_PAGE_SAMPLE_LEN = 2000

SCORE_EXACT_MATCH = 100.0
SCORE_SUBSTRING_MATCH = 80.0
SCORE_REVERSE_MATCH = 60.0
SCORE_MIN_THRESHOLD = 40.0
WORD_MIN_MATCH_RATIO = 0.5
MIN_REVERSE_MATCH_LEN = 4
MIN_WORD_LEN = 2

_SHARED_PROMPT_RULES = """CRITICAL FORMATTING RULE: Return PHI strings EXACTLY as they appear in the document.
Do not add spaces, punctuation, or modify formatting in any way.
If the document shows "d.washington1981@yahoo.com" return that exact string.

CRITICAL: Never truncate PHI values. Always return the complete string.
If an email ends in ".com", ".org", ".net" etc — always include the full TLD.
Never return a string ending with a bare period like "example@domain."

For addresses, always return TWO separate entries:
1. The street address (house number + street name + apt if present)
2. The ZIP/postal code on its own as a separate object

Example:
{"text": "3312 Peachtree Blvd NE", "category": "ADDRESS"},
{"text": "30305", "category": "ADDRESS"},
{"text": "#4B", "category": "ADDRESS"},
{"text": "Visa ending 7734", "category": "CREDIT_CARD"},
{"text": "UHC-88-004419922", "category": "ACCOUNT"}

Never return the full address as one string including city, state, and ZIP together."""

SYSTEM_PROMPT = f"""You are a HIPAA-compliant PHI detection system for clinical documents.

Your job is to identify ONLY patient-identifying information (PHI) in the provided text.

REDACT these (patient PII only):
- Patient full name, first name, last name (Category: Name)
- Patient date of birth (Category: DOB)
- Patient SSN (Category: SSN)
- Patient home address, street, city, state, ZIP (Category: Address)
- Patient ZIP code / postal code (redact separately from the street address)
- Patient apartment, unit, suite number (e.g. "#4B", "Apt 3C", "Suite 400") — return as a separate ADDRESS entry
- Patient personal phone numbers (Category: Phone)
- Patient personal email addresses (Category: Email)
- Patient insurance policy number and group number (Category: Policy)
- Patient insurance member ID / subscriber ID (distinct from policy/group number)
- Patient Medicare or Medicaid beneficiary ID (Category: Medicare ID)
- Patient medical record number (Category: MRN)
- Patient billing account numbers and portal account IDs (Category: Account)
- Patient portal login credentials and portal account ID
- Patient credit/debit card details including "Visa ending XXXX", "Mastercard ending XXXX", "card on file" references
- Emergency contact names and their phone numbers (Category: Contact)

DO NOT REDACT:
- Physician names and credentials
- Hospital or clinic names and addresses
- NPI numbers
- Physician phone numbers, pager numbers, direct lines
- Physician email addresses
- Dates of service, admission dates, discharge dates
- ICD-10, CPT, DRG, NDC codes
- Medication names, dosages, instructions
- Lab test names and values
- Diagnosis descriptions
- Insurance company names
- Billing charges and totals
- EIN or Tax ID numbers
- Pre-authorization numbers
- Consent form IDs
- Any clinical or administrative codes

{_SHARED_PROMPT_RULES}

Return ONLY a valid JSON array of objects. Each object has "text" (exact string to redact)
and "category" (one of: NAME, DOB, SSN, ADDRESS, PHONE, EMAIL, ACCOUNT, POLICY, MEDICARE,
CREDIT_CARD, EMPLOYER). No explanation, no markdown, no code blocks.
Example: [
  {{"text": "Robert Alan Hargrove", "category": "NAME"}},
  {{"text": "07/22/1964", "category": "DOB"}},
  {{"text": "523-67-4891", "category": "SSN"}},
  {{"text": "r.hargrove1964@gmail.com", "category": "EMAIL"}},
  {{"text": "3312 Peachtree Blvd NE", "category": "ADDRESS"}},
  {{"text": "30305", "category": "ADDRESS"}},
  {{"text": "#4B", "category": "ADDRESS"}},
  {{"text": "Visa ending 7734", "category": "CREDIT_CARD"}},
  {{"text": "UHC-88-004419922", "category": "ACCOUNT"}}
]
If no PHI found return: []"""


def detect_phi(text: str) -> list[dict]:
    """Sends OCR text to OpenAI and returns list of PHI dicts with text and category."""
    if not text or not text.strip():
        return []

    try:
        response = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Identify all patient PHI in this clinical document text:\n\n{text}"}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"}
        )
        response_text = response.choices[0].message.content.strip()
        parsed = json.loads(response_text)
        phi_list = parsed if isinstance(parsed, list) else parsed.get("phi", [])
        print(f"[AI PHI] Detected {len(phi_list)} PHI items: {phi_list}")
        return [item for item in phi_list if isinstance(item, dict) and item.get("text")]

    except json.JSONDecodeError as err:
        print(f"[AI PHI] JSON parse error: {err}")
        return []
    except Exception as err:
        print(f"[AI PHI] Error calling OpenAI: {err}")
        return []


def _extract_document_context(full_doc_sample: str) -> dict:
    """Extracts global document context (patient, facility, doctors, phones) via OpenAI."""
    prompt = """Analyze the clinical document sample and extract global context as JSON.
Return a JSON object with exact keys:
- document_type (string, e.g. inpatient discharge summary, claims form, clinic note)
- facility_name (string, e.g. Oregon Regional Health System)
- patient_name (string, e.g. Robert Alan Hargrove)
- non_patient_names (list of strings, e.g. ['Dr. Sarah Chen', 'Dr. James Tompkins', 'Linda Park'])
- facility_phones (list of strings, e.g. ['(503) 922-4400', '(503) 922-4499', '1-800-422-4299'])
If any field is missing, use empty string or empty list."""
    try:
        response = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Extract context from this text:\n\n{full_doc_sample}"}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as err:
        print(f"[AI PHI] Context extraction error: {err}")
        return {}


def _build_redaction_prompt(context: dict) -> str:
    """Builds a customized system prompt incorporating global document context."""
    doc_type = context.get("document_type", "clinical document")
    facility = context.get("facility_name", "medical facility")
    patient = context.get("patient_name", "the patient")
    non_patient = ", ".join(context.get("non_patient_names", []))
    phones = ", ".join(context.get("facility_phones", []))

    return f"""You are a HIPAA-compliant PHI detection system for a {doc_type} from {facility}.

Your job is to identify ONLY patient-identifying information (PHI) for patient {patient}.

CRITICAL SHARED CONTEXT RULES:
1. Patient Name: {patient}. Redact this name and any variations (e.g. Robert Hargrove, Mr. Hargrove) and family/spouse names (e.g. Margaret Hargrove).
2. DO NOT REDACT non-patient healthcare personnel: {non_patient}. (e.g. Dr. Sarah Chen, Dr. James Tompkins, Linda Park, etc.).
3. DO NOT REDACT facility/clinic/physician phone numbers: {phones} or physician direct lines/emails (e.g. billing@oregonregional.org).
4. DO NOT REDACT dates of service, admission/discharge dates, clinical codes (ICD-10, CPT, DRG, NDC, NPI), lab values, medications, insurance companies, EINs, or billing totals.
5. REDACT patient PII: DOB, SSN, patient home address (e.g. 1847 Oak Street), patient personal/work phones, patient emails, patient account/policy numbers, credit card info, ZIP codes, apt/unit numbers, member IDs, and portal logins.

{_SHARED_PROMPT_RULES}

Return ONLY a valid JSON array of objects. Each object has "text" (exact string to redact)
and "category" (one of: NAME, DOB, SSN, ADDRESS, PHONE, EMAIL, ACCOUNT, POLICY, MEDICARE,
CREDIT_CARD, EMPLOYER). No explanation, no markdown, no code blocks.
Example: [
  {{"text": "{patient}", "category": "NAME"}},
  {{"text": "07/22/1964", "category": "DOB"}},
  {{"text": "523-67-4891", "category": "SSN"}},
  {{"text": "r.hargrove1964@gmail.com", "category": "EMAIL"}},
  {{"text": "3312 Peachtree Blvd NE", "category": "ADDRESS"}},
  {{"text": "30305", "category": "ADDRESS"}},
  {{"text": "#4B", "category": "ADDRESS"}},
  {{"text": "Visa ending 7734", "category": "CREDIT_CARD"}},
  {{"text": "UHC-88-004419922", "category": "ACCOUNT"}}
]
If no PHI found return: []"""


def detect_phi_batch(pages_text: list[str]) -> list[list[dict]]:
    """Two-stage PHI detection across a multi-page document using shared context."""
    if not pages_text:
        return []

    full_doc_sample = "\n\n--- PAGE BREAK ---\n\n".join(
        page[:MAX_PAGE_SAMPLE_LEN] for page in pages_text
    )
    context = _extract_document_context(full_doc_sample)

    print(f"[AI PHI] Document: {context.get('document_type')}")
    print(f"[AI PHI] Facility: {context.get('facility_name')}")
    print(f"[AI PHI] Patient: {context.get('patient_name')}")
    print(f"[AI PHI] Non-patient names: {context.get('non_patient_names')}")
    print(f"[AI PHI] Facility phones: {context.get('facility_phones')}")

    system_prompt = _build_redaction_prompt(context)
    results = []

    for i, page_text in enumerate(pages_text):
        if not page_text.strip():
            results.append([])
            continue

        try:
            response = _client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Identify all patient PHI on page {i + 1}:\n\n{page_text}"}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)

            if isinstance(parsed, list):
                phi_list = parsed
            elif isinstance(parsed, dict):
                phi_list = next((v for v in parsed.values() if isinstance(v, list)), [])
            else:
                phi_list = []

            normalised = []
            for item in phi_list:
                if isinstance(item, dict) and "text" in item:
                    normalised.append({
                        "text": str(item["text"]),
                        "category": str(item.get("category", "PHI")).upper()
                    })
                elif isinstance(item, str):
                    normalised.append({"text": item, "category": "PHI"})

            print(f"[AI PHI] Page {i + 1}: {len(normalised)} PHI items: {[(x['text'], x['category']) for x in normalised]}")
            results.append(normalised)

        except Exception as err:
            print(f"[AI PHI] Page {i + 1} detection error: {err}")
            results.append([])

    return results


def _score_region_match(phi_lower: str, phi_words: list[str], region_lower: str) -> float:
    """Calculates match score between PHI string and OCR text region."""
    if phi_lower == region_lower:
        return SCORE_EXACT_MATCH
    if phi_lower in region_lower:
        return SCORE_SUBSTRING_MATCH * (len(phi_lower) / max(len(region_lower), 1))
    if region_lower in phi_lower and len(region_lower) > MIN_REVERSE_MATCH_LEN:
        return SCORE_REVERSE_MATCH
    if phi_words:
        matching = sum(1 for w in phi_words if w in region_lower)
        ratio = matching / len(phi_words)
        if ratio >= WORD_MIN_MATCH_RATIO:
            return SCORE_MIN_THRESHOLD * ratio
    return 0.0


def find_phi_boxes(phi_objects: list[dict], text_regions: list) -> list:
    """Maps PHI objects {text, category} to bounding boxes with source text attributes."""
    boxes_to_redact = []
    seen_ids = set()

    for phi in phi_objects:
        phi_text = phi.get("text", "")
        phi_category = phi.get("category", "PHI")
        phi_lower = phi_text.lower().strip()
        phi_words = [w for w in phi_lower.split() if len(w) > MIN_WORD_LEN]
        best_match, best_score = None, 0.0

        for region in text_regions:
            if id(region.box) in seen_ids:
                continue
            region_lower = region.text.lower().strip()
            score = _score_region_match(phi_lower, phi_words, region_lower)
            if score > best_score:
                best_score = score
                best_match = region

        if best_match is not None and best_score >= SCORE_MIN_THRESHOLD:
            best_match.box.category = phi_category
            best_match.box._source_text = best_match.text
            best_match.box._phi_text = phi_text
            boxes_to_redact.append(best_match.box)
            seen_ids.add(id(best_match.box))

    return boxes_to_redact
