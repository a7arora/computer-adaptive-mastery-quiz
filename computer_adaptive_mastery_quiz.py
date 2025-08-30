import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import requests
import json
import re
import random

# Multiple API keys
API_KEYS = [
    st.secrets["ANTHROPIC_API_KEY_1"],
    st.secrets["ANTHROPIC_API_KEY_2"],
    st.secrets["ANTHROPIC_API_KEY_3"],
]

CLAUDE_URL = "https://api.anthropic.com/v1/messages"
MODEL_NAME = "claude-sonnet-4-20250514"


# ---------------------------
# PDF Processing
# ---------------------------
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return [page.get_text() for page in doc if page.get_text().strip()]


# ---------------------------
# Prompt Generation
# ---------------------------
def generate_prompt(text_chunk):
    return f"""
You are a teacher who is designing a test with multiple choice questions (each with 4 answer choices) to test content from a passage.

Given the following passage or notes, generate exactly 12 multiple choice questions that test comprehension and critical thinking. The questions must vary in difficulty. If there is not enough content to write 20 good questions, repeat or expand the material, or create additional plausible questions that still test content that is similar to what is in the passage.

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
- 3 easy (≥85%), 3 medium (60–84%), 3 medium-hard (40-60%), 3 hard(<40%)
You tend to make the questions easier than the respective labels(for instance, you make hard questions that 60% of students answer correctly or medium-hard questions that 70% of students answer correctly), so please try to make the questions more significantly more challenging than you would think they should be for medium-hard and hard questions. Please try to make the medium-hard and hard questions apply the concepts covered in the lecture material in unique ways. 

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

Return ONLY a valid JSON array of 12 questions. Focus on testing conceptual understanding rather than text memorization.

If the passage contains code or table output, generate questions about how the code works and what outputs mean - but present these as general programming/analysis questions, not as references to "the code shown" or "the table above."

Passage:
{text_chunk}
"""


# ---------------------------
# Claude API
# ---------------------------
def call_claude_api(prompt, api_key):
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL_NAME,
        "max_tokens": 4500,
        "temperature": 0.7,
        "system": "You are a helpful educational assistant. Always return properly formatted JSON arrays without any additional text or markdown formatting.",
        "messages": [{"role": "user", "content": prompt}],
    }
    response = requests.post(CLAUDE_URL, headers=headers, json=data)
    if response.status_code != 200:
        return None, response.text
    return response.json()["content"][0]["text"], None


# ---------------------------
# JSON Cleaning & Parsing
# ---------------------------
def clean_response_text(text: str) -> str:
    text = text.strip()
    # Remove ```json fences
    fence_patterns = [
        r"```json\s*(.*?)```",
        r"```\s*(.*?)```",
    ]
    for pattern in fence_patterns:
        fence_match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if fence_match:
            text = fence_match.group(1).strip()
            break
    # Extract array
    start_idx, end_idx = text.find("["), text.rfind("]")
    return text[start_idx:end_idx + 1] if start_idx != -1 and end_idx != -1 else text


def repair_json(text: str) -> str:
    text = re.sub(r",\s*([\]}])", r"\1", text)  # remove trailing commas
    text = re.sub(r"}\s*{", r"}, {", text)
    text = re.sub(r"]\s*\[", r"], [", text)
    text = re.sub(r"(\d+)\s*%", r"\1", text)
    return text


def parse_question_json(text: str):
    cleaned = clean_response_text(text)
    repaired = repair_json(cleaned)
    try:
        result = json.loads(repaired)
        return [q for q in result if validate_question_structure(q)]
    except Exception as e:
        st.error(f"⚠️ JSON parse error: {e}")
        return []


def validate_question_structure(question):
    required = ["question", "options", "correct_answer", "explanation", "cognitive_level", "estimated_correct_pct", "reasoning"]
    if not isinstance(question, dict):
        return False
    if not all(f in question for f in required):
        return False
    if not isinstance(question["options"], list) or len(question["options"]) != 4:
        return False
    if question["correct_answer"].upper() not in ["A", "B", "C", "D"]:
        return False
    return True


# ---------------------------
# Difficulty Grouping
# ---------------------------
def assign_difficulty_label(estimated_pct):
    try:
        pct = int(estimated_pct)
    except:
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
        label = assign_difficulty_label(q.get("estimated_correct_pct", 0))
        if label:
            q["difficulty_label"] = label
            groups[label].append(q)
    return groups


# ---------------------------
# Quiz Engine
# ---------------------------
def pick_question(diff, asked, all_qs):
    pool = all_qs.get(diff, [])
    return [(i, q) for i, q in enumerate(pool) if (diff, i) not in asked]


def find_next_difficulty(current_diff, going_up, asked, all_qs):
    next_diff = current_diff + 1 if going_up else current_diff - 1
    if 1 <= next_diff <= 8 and pick_question(next_diff, asked, all_qs):
        return next_diff
    search_range = range(next_diff + 1, 9) if going_up else range(next_diff - 1, 0, -1)
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


def compute_mastery_score(answers):
    mastery_bands = {(1, 2): 25, (3, 4): 65, (5, 6): 85, (7, 8): 100}
    min_attempts = 3
    band_scores = []
    for levels, weight in mastery_bands.items():
        relevant = [c for d, c in answers if d in levels]
        if not relevant:
            continue
        acc = sum(relevant) / len(relevant)
        normalized = max((acc - 0.25) / 0.75, 0)
        if len(relevant) < min_attempts:
            band_scores.append(normalized * weight * (len(relevant) / min_attempts))
        else:
            band_scores.append(normalized * weight)
    return int(round(max(band_scores))) if band_scores else 0


def render_mastery_bar(score):
    color = "red" if score < 30 else "yellow" if score < 50 else "green"
    text_color = "white" if color in ["red", "green"] else "black"
    st.markdown(f"""
    <style>
        .mastery-bar-wrapper {{position: fixed;top: 0;left: 0;width: 100%;z-index: 9999;background-color: white;padding: 8px 16px;box-shadow: 0 2px 6px rgba(0,0,0,0.1);}}
        .mastery-bar {{border: 1px solid #ccc;border-radius: 8px;overflow: hidden;height: 24px;width: 100%;background-color: #eee;}}
        .mastery-bar-fill {{height: 100%;width: {score}%;background-color: {color};text-align: center;color: {text_color};font-weight: bold;line-height: 24px;}}
        .spacer {{height: 60px;}}
    </style>
    <div class="mastery-bar-wrapper"><div class="mastery-bar"><div class="mastery-bar-fill">{score}%</div></div></div><div class="spacer"></div>
    """, unsafe_allow_html=True)


# ---------------------------
# App UI
# ---------------------------
st.title("AscendQuiz")

if "all_questions" not in st.session_state:
    st.markdown("Welcome to AscendQuiz! Upload a PDF to generate adaptive questions.")

score = compute_mastery_score(st.session_state.get("quiz_state", {}).get("answers", []))
render_mastery_bar(score)

uploaded_pdf = st.file_uploader("Upload class notes (PDF)", type="pdf")
if uploaded_pdf:
    with st.spinner("Generating questions from multiple API keys..."):
        chunks = extract_text_from_pdf(uploaded_pdf)
        if len(chunks) <= 2:
            grouped_chunks = ["\n\n".join(chunks)]
        else:
            grouped_chunks = ["\n\n".join(chunks[i:i+4]) for i in range(0, len(chunks), 4)]
        chunks_to_use = grouped_chunks[:3] if len(grouped_chunks) >= 3 else [grouped_chunks[0]] * 3

        all_questions = []
        for key, chunk in zip(API_KEYS, chunks_to_use):
            prompt = generate_prompt(chunk)
            response_text, error = call_claude_api(prompt, key)
            if error:
                st.error(f"API error with one key: {error}")
                continue
            parsed = parse_question_json(response_text)
            all_questions.extend(parsed)

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
            st.success(f"✅ Generated {len(all_questions)} questions from 3 API calls!")
            st.session_state.quiz_ready = True
            st.rerun()
        else:
            st.error("No questions generated. Try another PDF.")

elif "quiz_ready" in st.session_state and st.session_state.quiz_ready:
    all_qs = st.session_state.questions_by_difficulty
    state = st.session_state.get("quiz_state")

    score = compute_mastery_score(state.get("answers", []))

    if not state["quiz_end"]:
        if state["current_q"] is None and not state.get("show_explanation", False):
            diff, idx, q = get_next_question(state["current_difficulty"], state["asked"], all_qs)
            if q is None:
                state["quiz_end"] = True
            else:
                state["current_q"], state["current_q_idx"], state["current_difficulty"] = q, idx, diff

    if not state["quiz_end"] and state["current_q"]:
        q = state["current_q"]
        idx = state["current_q_idx"]

        st.markdown(f"### Question (Difficulty {state['current_difficulty']})")
        st.markdown(q["question"], unsafe_allow_html=True)

        def strip_label(text): return re.sub(r"^[A-Da-d][\).:\-]?\s+", "", text).strip()
        options = [strip_label(str(opt)) for opt in q["options"]]
        selected = st.radio("Select your answer:", [f"{l}. {o}" for l, o in zip("ABCD", options)], index=None, key=f"radio_{idx}")

        if st.button("Submit Answer", key=f"submit_{idx}") and not state.get("show_explanation", False):
            if not selected:
                st.warning("Please select an answer before submitting.")
                st.stop()
            selected_letter = selected.split(".")[0].strip().upper()
            correct_letter = q["correct_answer"].strip().upper()
            correct = selected_letter == correct_letter
            state["asked"].add((state["current_difficulty"], idx))
            state["answers"].append((state["current_difficulty"], correct))
            state["last_correct"], state["last_explanation"], state["show_explanation"] = correct, q["explanation"], True
            score = compute_mastery_score(state["answers"])
            if score >= 50:
                state["quiz_end"] = True

        if state.get("show_explanation", False):
            st.success("✅ Correct!") if state["last_correct"] else st.error("❌ Incorrect.")
            st.markdown("**Explanation:**")
            st.markdown(state["last_explanation"], unsafe_allow_html=True)
            if st.button("Next Question"):
                state["current_difficulty"] = find_next_difficulty(state["current_difficulty"], state["last_correct"], state["asked"], all_qs)
                state.update({"current_q": None, "current_q_idx": None, "show_explanation": False, "last_correct": None, "last_explanation": None})
                st.rerun()

    elif state["quiz_end"]:
        st.markdown("## Quiz Completed 🎉")
        if score >= 50:
            st.success(f"🎉 Mastery achieved! Score {score}%.")
        else:
            st.warning(f"Not yet mastered. Score {score}%. Review and retry.")
        if "all_questions" in st.session_state:
            df = pd.DataFrame(st.session_state.all_questions)
            st.download_button("📥 Download All Questions (CSV)", df.to_csv(index=False), "ascendquiz_questions.csv", "text/csv")




