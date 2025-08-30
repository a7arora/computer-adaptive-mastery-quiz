import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import requests
import json
import re
import random
import io
from typing import List, Tuple

# --- NEW: OCR dependencies ---
try:
    from PIL import Image, ImageOps, ImageFilter
    import pytesseract
    TESS_AVAILABLE = True
except Exception:
    TESS_AVAILABLE = False

# =============================
# Config & API
# =============================
API_KEY = st.secrets["ANTHROPIC_API_KEY_1"]

headers = {
    "x-api-key": f"{API_KEY}",
    "anthropic-version": "2023-06-01",
    "Content-Type": "application/json"
}
CLAUDE_URL = "https://api.anthropic.com/v1/messages"
MODEL_NAME = "claude-sonnet-4-20250514"

# =============================
# OCR Helpers
# =============================

def page_to_image(page: fitz.Page, zoom: float = 2.0) -> Image.Image:
    """Render a PDF page to a PIL Image at the given zoom (scaling)."""
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_bytes = pix.tobytes("png")
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def preprocess_for_ocr(img: Image.Image, mode: str = "auto") -> Image.Image:
    """Lightweight preprocessing to improve OCR. Modes: auto|grayscale|binarize|sharpen|none"""
    if mode == "none":
        return img

    if mode == "grayscale" or mode == "auto":
        img = ImageOps.grayscale(img)

    if mode in ("binarize", "auto"):
        # simple adaptive-like threshold: blur + point
        img = img.filter(ImageFilter.MedianFilter(size=3))
        img = img.point(lambda x: 255 if x > 160 else 0)

    if mode == "sharpen":
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

    return img


def ocr_image(img: Image.Image, lang: str = "eng") -> str:
    if not TESS_AVAILABLE:
        return ""
    try:
        return pytesseract.image_to_string(img, lang=lang)
    except Exception:
        return ""


# =============================
# PDF Text Extraction (Text + OCR fallback)
# =============================

def extract_text_from_pdf_or_ocr(
    pdf_file,
    enable_ocr: bool = True,
    text_min_len_for_page: int = 60,
    zoom: float = 2.0,
    ocr_lang: str = "eng",
    preprocess_mode: str = "auto",
) -> List[str]:
    """
    Extract text from a PDF. For each page, first try live text from the PDF.
    If the page has too little text and OCR is enabled, rasterize and OCR it.
    Returns a list of non-empty page strings.
    """
    # Open once using bytes buffer (Streamlit uploaded file is a BytesIO-like object)
    file_bytes = pdf_file.read()
    doc = fitz.open(stream=file_bytes, filetype="pdf")

    pages_text: List[str] = []

    progress = st.progress(0, text="Reading PDF…")
    total_pages = len(doc)

    for i, page in enumerate(doc):
        raw_text = page.get_text().strip()
        page_text = raw_text

        # If little or no extractable text and OCR is enabled, try OCR
        if enable_ocr and (len(raw_text) < text_min_len_for_page):
            img = page_to_image(page, zoom=zoom)
            img = preprocess_for_ocr(img, mode=preprocess_mode)
            ocr_text = ocr_image(img, lang=ocr_lang).strip()

            # If both exist, prefer the longer one (helps with mixed content)
            if len(ocr_text) > len(raw_text):
                page_text = ocr_text

        if page_text:
            pages_text.append(page_text)

        progress.progress((i + 1) / max(total_pages, 1), text=f"Processed page {i+1}/{total_pages}")

    progress.empty()

    # Final sanity pass: drop empty strings
    return [t for t in pages_text if t and t.strip()]


# =============================
# Prompting & LLM
# =============================

def generate_prompt(text_chunk):
    return f"""
You are a teacher who is designing a test with multiple choice questions (each with 4 answer choices) to test content from a passage.

Given the following passage or notes, generate exactly 20 multiple choice questions that test comprehension and critical thinking. The questions must vary in difficulty. If there is not enough content to write 20 good questions, repeat or expand the material, or create additional plausible questions that still test content that is similar to what is in the passage.

**CRITICAL REQUIREMENT - NO TEXT REFERENCES:**
- Questions must be COMPLETELY SELF-CONTAINED and not reference the original text
- DO NOT use phrases like "according to the passage," "the text states," "the first example," "as mentioned," "the author discusses," etc.
- DO NOT reference specific figures, tables, pages, or sections from the passage
- Present all necessary context within the question itself
- Students should be able to answer based on their understanding of the concepts, not memory of where things appeared in the text
- Frame questions as direct concept tests, not reading comprehension
- If there is information about ISBN or ebook distribution consequences or copyrights, do not ask questions about these things. Only ask questions about academic content

**Example of what NOT to do:**
❌ "According to the passage, what does the first example demonstrate?"
❌ "The text mentions three types of X. Which one is described as Y?"

**Example of what TO do:**
✅ "In SAS programming, dates are stored as the number of days from which reference point?"
✅ "What happens when you use correlated features in a Random Forest model?"

**Requirements**:
- 5 easy (≥85%), 5 medium (60–84%), 5 medium-hard (40-60%), 5 hard(<40%)
You tend to make the questions easier than the respective labels(for instance, you make hard questions that 60% of students answer correctly or medium-hard questions that 70% of students answer correctly), so please try to make the questions more significantly more challenging than you would think they should be for medium-hard and hard questions

**CRITICAL JSON FORMAT REQUIREMENTS:**
Return ONLY a valid JSON array. Each question object must have this EXACT structure:

{{
  "question": "Your question text here",
  "options": ["A option text", "B option text", "C option text", "D option text"],
  "correct_answer": "A",
  "explanation": "Your explanation here",
  "cognitive_level": "Remember",
  "estimated_correct_pct": 85,
  "reasoning": "Your reasoning here"
}}

**IMPORTANT JSON RULES:**
- Use ARRAY format for options: ["option1", "option2", "option3", "option4"]
- Do NOT use object format like {{"A": "...", "B": "..."}}
- Do NOT include A), B), C), D) labels in the option text - just the content
- Ensure all JSON is properly escaped (quotes, backslashes, etc.)
- Return ONLY the JSON array, no additional text or markdown formatting
- No trailing commas
- All strings must be properly quoted

**Each question must include the following fields:**

- "question": A clear, concise, and unambiguous question that tests understanding of concepts from the passage. The question should be COMPLETELY SELF-CONTAINED with all necessary context included. Never reference "the passage," "the text," specific examples by position (first, second, etc.), or figures/tables. Ask about the concept directly. Make the question slightly more difficult than typical for the specified difficulty level. Ensure it tests conceptual understanding that would be valuable for learning, not memorization of text structure.

- "options": An array of exactly 4 answer choices (without A, B, C, D labels - just the content). For medium/hard questions, create wrong answers that reflect common misconceptions. Ensure only one answer is clearly correct.

- "correct_answer": The letter ("A", "B", "C", or "D") corresponding to the correct option position in the array.

- "explanation": A deep, pedagogically useful explanation that teaches the concept behind the correct answer. The explanation must:
    1. Start by stating the correct letter and full answer
    2. Explain WHY that answer is correct using conceptual reasoning - explain mechanisms, properties, or principles
    3. For each incorrect answer, explain why it's wrong and what misconception might lead to choosing it
    4. Focus on teaching the underlying concept, not referencing where information appeared in the text
    5. Use the tone of a tutor helping a student understand the concept

- "cognitive_level": Choose from "Remember", "Understand", "Apply", "Analyze", "Evaluate", or "Create" based on the cognitive skill actually tested.

- "estimated_correct_pct": Numeric estimate of percentage of students expected to answer correctly. Consider complexity, inference required, and common misconceptions. You tend to underestimate difficulty, so evaluate thoroughly.

- "reasoning": Brief rationale for the percentage assignment considering complexity, inference required, and detail recall.

All math expressions must use valid LaTeX format with $...$ for inline math and $$...$$ for display math.

Return ONLY a valid JSON array of 20 questions. Focus on testing conceptual understanding rather than text memorization.

If the passage contains code or table output, generate questions about how the code works and what outputs mean - but present these as general programming/analysis questions, not as references to "the code shown" or "the table above."

Passage:
{text_chunk}
"""


def call_claude_api(prompt):
    data = {
        "model": MODEL_NAME,
        "max_tokens": 4500,
        "temperature": 0.7,
        "system": "You are a helpful educational assistant. Always return properly formatted JSON arrays without any additional text or markdown formatting.",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(CLAUDE_URL, headers=headers, json=data)
    if response.status_code != 200:
        return None, response.text
    return response.json()["content"][0]["text"], None


# =============================
# JSON Cleaning / Parsing (unchanged)
# =============================

def clean_response_text(text: str) -> str:
    text = text.strip()
    fence_patterns = [
        r"```json\s*(.*?)```",
        r"```\s*(.*?)```",
        r"`{3,}\s*json\s*(.*?)`{3,}",
        r"`{3,}\s*(.*?)`{3,}"
    ]
    for pattern in fence_patterns:
        fence_match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if fence_match:
            text = fence_match.group(1).strip()
            break
    start_idx = text.find('[')
    end_idx = text.rfind(']')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_text = text[start_idx:end_idx + 1]
        return json_text.strip()
    return text


def repair_json(text: str) -> str:
    text = re.sub(r',\s*([\]}])', r'\1', text)
    text = re.sub(r'}\s*{', r'}, {', text)
    text = re.sub(r']\s*\[', r'], [', text)
    text = re.sub(r'(\d+)\s*%', r'\1', text)
    text = text.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')

    def convert_options_object_to_array(match):
        options_content = match.group(1)
        option_pattern = r'"[A-D]":\s*"([^"]*)"'
        options = re.findall(option_pattern, options_content)
        if len(options) == 4:
            options_array = json.dumps(options)
            return f'"options": {options_array}'
        return match.group(0)

    text = re.sub(r'"options":\s*\{([^}]+)\}', convert_options_object_to_array, text)
    return text


def validate_question_structure(question, index):
    required_fields = ["question", "options", "correct_answer", "explanation", "cognitive_level", "estimated_correct_pct", "reasoning"]
    if not isinstance(question, dict):
        return False
    for field in required_fields:
        if field not in question:
            return False
    if not isinstance(question["options"], list) or len(question["options"]) != 4:
        return False
    if str(question["correct_answer"]).upper() not in ["A", "B", "C", "D"]:
        return False
    return True


def extract_questions_manually(text):
    questions = []
    question_pattern = r'\{\s*"question":[^}]*?"reasoning":[^}]*?\}'
    potential_questions = re.findall(question_pattern, text, re.DOTALL)
    for q_text in potential_questions:
        try:
            repaired = repair_json(q_text)
            q_obj = json.loads(repaired)
            if validate_question_structure(q_obj, len(questions)):
                questions.append(q_obj)
        except Exception:
            continue
    return questions


def parse_question_json(text: str):
    cleaned = clean_response_text(text)
    cleaned = repair_json(cleaned)
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            valid_questions = []
            for i, q in enumerate(result):
                if validate_question_structure(q, i):
                    valid_questions.append(q)
            return valid_questions
        else:
            return []
    except json.JSONDecodeError:
        try:
            questions = extract_questions_manually(cleaned)
            if questions:
                return questions
        except Exception:
            pass
        st.error("⚠️ JSON parsing failed. Please try uploading the PDF again.")
        return []


# =============================
# Difficulty Binning (unchanged)
# =============================

def filter_invalid_difficulty_alignment(questions):
    bloom_difficulty_ranges = {
        "Remember": (70, 100),
        "Understand": (50, 85),
        "Apply": (45, 80),
        "Analyze": (25, 65),
        "Evaluate": (0, 60),
        "Create": (0, 50)
    }
    valid = []
    invalid = []
    for q in questions:
        if not isinstance(q, dict):
            invalid.append(q)
            continue
        cog = str(q.get("cognitive_level", "")).strip().capitalize()
        try:
            pct = int(q.get("estimated_correct_pct", -1))
        except Exception:
            pct = -1
        if cog in bloom_difficulty_ranges and 0 <= pct <= 100:
            low, high = bloom_difficulty_ranges[cog]
            (valid if (low <= pct <= high) else invalid).append(q)
        else:
            invalid.append(q)
    return valid, invalid


def assign_difficulty_label(estimated_pct):
    try:
        pct = int(estimated_pct)
    except Exception:
        return None
    if pct < 30: return 8
    elif pct < 40: return 7
    elif pct < 50: return 6
    elif pct < 65: return 5
    elif pct < 75: return 4
    elif pct < 85: return 3
    elif pct < 90: return 2
    else: return 1


def group_by_difficulty(questions):
    groups = {i: [] for i in range(1, 9)}
    for q in questions:
        pct = q.get("estimated_correct_pct", 0)
        label = assign_difficulty_label(pct)
        if label:
            q["difficulty_label"] = label
            groups[label].append(q)
    return groups


def pick_question(diff, asked, all_qs):
    pool = all_qs.get(diff, [])
    return [(i, q) for i, q in enumerate(pool) if (diff, i) not in asked]


def find_next_difficulty(current_diff, going_up, asked, all_qs):
    next_diff = current_diff + 1 if going_up else current_diff - 1
    if 1 <= next_diff <= 8 and pick_question(next_diff, asked, all_qs):
        return next_diff
    search_range = (range(next_diff + 1, 9) if going_up else range(next_diff - 1, 0, -1))
    for d in search_range:
        if pick_question(d, asked, all_qs):
            return d
    return current_diff


def get_next_question(current_diff, asked, all_qs):
    available = pick_question(current_diff, asked, all_qs)
    if not available:
        return current_diff, None, None
    idx, q = random.choice(available)
    return current_diff, idx, q


def compute_mastery_score(answers: List[Tuple[int, bool]]):
    mastery_bands = {
        (1, 2): 25,
        (3, 4): 65,
        (5, 6): 85,
        (7, 8): 100
    }
    min_attempts_required = 3
    band_scores = []
    for levels, weight in mastery_bands.items():
        relevant = [correct for d, correct in answers if d in levels]
        attempts = len(relevant)
        if attempts == 0:
            continue
        acc = sum(relevant) / attempts
        normalized_score = max((acc - 0.25) / 0.75, 0)
        if attempts < min_attempts_required:
            scaled_score = normalized_score * weight * (attempts / min_attempts_required)
            band_scores.append(scaled_score)
        else:
            band_scores.append(normalized_score * weight)
    if not band_scores:
        return 0
    return int(round(max(band_scores)))


# =============================
# UI helpers
# =============================

def render_mastery_bar(score):
    if score < 30:
        color = "red"; text_color = "white"
    elif score < 50:
        color = "yellow"; text_color = "black"
    else:
        color = "green"; text_color = "white"
    st.markdown(f"""
    <style>
        .mastery-bar-wrapper {{ position: fixed; top: 0; left: 0; width: 100%; z-index: 9999; background-color: white; padding: 8px 16px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); }}
        .mastery-bar {{ border: 1px solid #ccc; border-radius: 8px; overflow: hidden; height: 24px; width: 100%; background-color: #eee; }}
        .mastery-bar-fill {{ height: 100%; width: {score}%; background-color: {color}; text-align: center; color: {text_color}; font-weight: bold; line-height: 24px; }}
        .spacer {{ height: 60px; }}
    </style>
    <div class="mastery-bar-wrapper"><div class="mastery-bar"><div class="mastery-bar-fill">{score}%</div></div></div><div class="spacer"></div>
    """, unsafe_allow_html=True)


# =============================
# App
# =============================

st.title("AscendQuiz (OCR-enabled)")

# Sidebar OCR controls
with st.sidebar:
    st.subheader("OCR Settings")
    enable_ocr = st.checkbox("Enable OCR for image-only or low-text pages", value=True)
    ocr_lang = st.text_input("Tesseract language codes (comma-separated)", value="eng")
    zoom = st.slider("Rasterization scale (affects OCR quality)", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
    text_thresh = st.number_input("Min extracted characters before using OCR", min_value=0, max_value=2000, value=60, step=10)
    preprocess_mode = st.selectbox("Preprocessing", options=["auto", "grayscale", "binarize", "sharpen", "none"], index=0)
    if not TESS_AVAILABLE:
        st.info("pytesseract/Pillow not detected. Install Tesseract OCR and the pytesseract Python package on your host.")
        st.caption("Ubuntu example: `sudo apt-get update && sudo apt-get install -y tesseract-ocr libtesseract-dev && pip install pytesseract pillow`\nAdd extra language packs like `tesseract-ocr-spa` for Spanish, etc.")

if "all_questions" not in st.session_state:
    st.markdown("""
Welcome to your personalized learning assistant — an AI-powered tool that transforms any PDF into a mastery-based, computer-adaptive quiz.

**New:** This version can read scanned/image-only PDFs using OCR. Use the sidebar to configure OCR language, scaling, and preprocessing.

---
""")

score = compute_mastery_score(st.session_state.get("quiz_state", {}).get("answers", []))
render_mastery_bar(score)

uploaded_pdf = st.file_uploader("Upload class notes (PDF)", type="pdf")

if uploaded_pdf:
    with st.spinner("Extracting text (with OCR fallback)…"):
        # IMPORTANT: Streamlit uploads are one-shot read; pass a copy to our function
        uploaded_pdf.seek(0)
        chunks = extract_text_from_pdf_or_ocr(
            uploaded_pdf,
            enable_ocr=enable_ocr,
            text_min_len_for_page=text_thresh,
            zoom=zoom,
            ocr_lang="+".join([c.strip() for c in ocr_lang.split(",") if c.strip()]) or "eng",
            preprocess_mode=preprocess_mode,
        )

    if not chunks:
        st.error("Couldn't extract any text (even with OCR). Try increasing the rasterization scale or enabling OCR.")
        st.stop()

    with st.expander("Preview extracted text (first 2 pages)"):
        preview = "\n\n----\n\n".join(chunks[:2])
        st.text_area("Text preview", preview, height=250)

    with st.spinner("Generating questions…"):
        # Adaptive chunking
        if len(chunks) <= 2:
            grouped_chunks = ["\n\n".join(chunks)]
        else:
            grouped_chunks = ["\n\n".join(chunks[i:i+4]) for i in range(0, len(chunks), 4)]

        all_questions = []
        chunks_to_use = grouped_chunks[:2] if len(grouped_chunks) >= 2 else [grouped_chunks[0], grouped_chunks[0]]

        for chunk in chunks_to_use:
            prompt = generate_prompt(chunk)
            response_text, error = call_claude_api(prompt)
            if error:
                st.error("API error: " + error)
                continue
            parsed = parse_question_json(response_text)
            valid, invalid = filter_invalid_difficulty_alignment(parsed)
            all_questions.extend(valid)
            if "filtered_questions" not in st.session_state:
                st.session_state.filtered_questions = []
            st.session_state.filtered_questions.extend(invalid)

        if all_questions:
            st.session_state.all_questions = all_questions
            st.session_state.questions_by_difficulty = group_by_difficulty(all_questions)
            st.session_state.quiz_state = {
                "current_difficulty": 4,
                "asked": set(),
                "answers": [],
                "quiz_end": False,
                "current_q_idx": None,
                "current_q": None,
                "show_explanation": False,
                "last_correct": None,
                "last_explanation": None,
            }
            st.success(f"✅ Questions generated! Starting the quiz with {len(all_questions)} questions…")
            st.session_state.quiz_ready = True
            st.rerun()
        else:
            st.error("No questions were generated. Please try again with a different PDF or tweak OCR settings.")

elif "quiz_ready" in st.session_state and st.session_state.quiz_ready:
    all_qs = st.session_state.questions_by_difficulty
    state = st.session_state.get("quiz_state", None)

    if state is None:
        st.warning("Quiz state not found. Please restart the app or re-upload a PDF.")
        st.stop()

    score = compute_mastery_score(state.get("answers", []))

    if not state["quiz_end"]:
        if state["current_q"] is None and not state.get("show_explanation", False):
            diff, idx, q = get_next_question(state["current_difficulty"], state["asked"], all_qs)
            if q is None:
                state["quiz_end"] = True
            else:
                state["current_q"] = q
                state["current_q_idx"] = idx
                state["current_difficulty"] = diff

    if not state["quiz_end"] and state["current_q"]:
        q = state["current_q"]
        idx = state["current_q_idx"]

        st.markdown(f"### Question (Difficulty {state['current_difficulty']})")
        st.markdown(q["question"], unsafe_allow_html=True)

        def strip_leading_label(text):
            return re.sub(r"^[A-Da-d][\).:\-]?\s+", "", str(text)).strip()

        option_labels = ["A", "B", "C", "D"]
        cleaned_options = [strip_leading_label(str(opt)) for opt in q["options"]]
        rendered_options = [f"{label}. {text}" for label, text in zip(option_labels, cleaned_options)]

        selected = st.radio("Select your answer:", options=rendered_options, key=f"radio_{idx}", index=None)

        if st.button("Submit Answer", key=f"submit_{idx}") and not state.get("show_explanation", False):
            if selected is None:
                st.warning("Please select an answer before submitting.")
                st.stop()

            selected_letter = selected.split(".")[0].strip().upper()
            letter_to_index = {"A": 0, "B": 1, "C": 2, "D": 3}
            correct_letter = str(q["correct_answer"]).strip().upper()
            correct_index = letter_to_index.get(correct_letter, None)

            if correct_index is None:
                st.error("⚠️ Question error: Correct answer letter invalid.")
                state["quiz_end"] = True
                st.stop()

            correct = (selected_letter == correct_letter)
            state["asked"].add((state["current_difficulty"], idx))
            state["answers"].append((state["current_difficulty"], correct))
            state["last_correct"] = correct
            state["last_explanation"] = q["explanation"]
            state["show_explanation"] = True

            score = compute_mastery_score(state["answers"])
            if score >= 50:
                state["quiz_end"] = True

        if state.get("show_explanation", False):
            st.success("✅ Correct!") if state["last_correct"] else st.error("❌ Incorrect.")
            st.markdown("**Explanation:**")
            st.markdown(state["last_explanation"], unsafe_allow_html=True)

            if st.button("Next Question"):
                if state["last_correct"]:
                    state["current_difficulty"] = find_next_difficulty(
                        state["current_difficulty"], going_up=True, asked=state["asked"], all_qs=all_qs
                    )
                else:
                    state["current_difficulty"] = find_next_difficulty(
                        state["current_difficulty"], going_up=False, asked=state["asked"], all_qs=all_qs
                    )
                state["current_q"] = None
                state["current_q_idx"] = None
                state["show_explanation"] = False
                state["last_correct"] = None
                state["last_explanation"] = None
                st.rerun()

    elif state["quiz_end"]:
        st.markdown("## Quiz Completed 🎉")
        if score >= 50:
            st.success(f"🎉 You have mastered the content! Your mastery score is {score}%. Great job!")
        else:
            st.warning(f"Mastery not yet achieved. Your mastery score is {score}%. Review the material and try again.")

        if "all_questions" in st.session_state:
            df = pd.DataFrame(st.session_state.all_questions)
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download All Quiz Questions (CSV)",
                data=csv_data,
                file_name="ascendquiz_questions.csv",
                mime="text/csv"
            )

