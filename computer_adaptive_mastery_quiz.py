import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import requests
import json
import re
import random

API_KEY = st.secrets["DEEPSEEK_API_KEY"]

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME = "deepseek-chat"

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return [page.get_text() for page in doc if page.get_text().strip()]

def generate_prompt(text_chunk):
    return f"""
IMPORTANT: Return exactly one JSON array and nothing else. The JSON must be valid (no markdown, no commentary, no backticks). Example: [ {{ "question": "...", "options": ["A...", "B...", "C...", "D..."], "correct_answer": "A", "explanation": "...", "cognitive_level": "Understand", "estimated_correct_pct": 85, "reasoning": "..." }}, ... ]

You are a teacher who is designing a test with multiple choice questions (each with 4 answer choices) to test content from a passage.

Given the following passage or notes, generate exactly 20 multiple choice questions that test comprehension and critical thinking. The questions must vary in difficulty. If there is not enough content to write 20 good questions, repeat or expand the material, or create additional plausible questions that still test content that is similar to what is in the passage. You can repeat the same questions if there is very little content, even though this is not preferred, so that there are 20 total questions. Please make sure that the question is something that relates to the material in the passage or an application/extension of it. Please do not use specialized vocabulary that is not explicitly stated in the passage without definition (such as mentioning photosynthesis when a passage only talks about flowers — in that case please define pollination before asking a question about pollination).

**Requirements**:
- 5 easy (≥85%), 5 medium (60–84%), 5 medium-hard (40-60%), 5 hard (<40%)

**Each question must include the following fields:**

- "question": A clear, concise, and unambiguous question directly related or an application of content from the passage that aligns with key learning objectives. The question should be designed to test understanding of material covered in the passage and should be made so it could show up on an educational assessment testing material from this passage. The question should be cognitively appropriate for the specified difficulty level, encouraging critical thinking, application, analysis, or synthesis rather than rote recall if not at the easiest difficulty level. Please make the question at a slightly higher difficulty than you would expect a question of this difficulty to be at (i.e. creating a medium hard question when a medium question is requested). Avoid overly complex wording or ambiguity to ensure students understand exactly what is being asked. Furthermore, make sure that the question has all the context in itself and does not reference specific figures or pages in the passage, as the question is designed for the user to do independently without the passage, even though it tests knowledge of content from the passage. You tend to make questions at an easier difficulty than requested, so make it slightly more difficult than you would expect from the given difficulty level. 
- "options": A list of 4 plausible answer choices labeled "A", "B", "C", and "D" (with one of them being the correct answer). If the question is of medium or hard difficulty, please come up with wrong answers that are ones that a user who does not know the concept well or makes an error would select. Please make sure that only one answer choice is correct by solving the problem and checking all of the answer choices carefully and thoroughly. It should not be ambiguous as to whether an answer choice is correct or not.
- "correct_answer": The letter ("A", "B", "C", or "D") corresponding to the correct option. Please make sure that this answer is correct and is clearly the only correct answer.
- "explanation": A deep, pedagogically useful explanation that **teaches the concept** behind the correct answer and analyzes the flaws in the others. The explanation must:
    1. Start by stating the correct letter and full answer.
    2. Teach **why** that answer is correct using **conceptual reasoning** — including how the mechanism works, or why the property matters — not just restating facts.
       - For example, if the correct answer is "correlation between features degrades performance," then explain **why correlated features reduce tree diversity in Random Forests**, which is the core reason performance drops.
       - Use step-by-step reasoning, examples, or analogies when helpful.
    3. For each incorrect answer, state its letter and text, and **explain why it's wrong**, including what misconception a student might have that could lead them to choose it.
    4. The tone should be that of a **tutor or explainer**, helping a confused student understand both the correct idea and the traps in the wrong ones.
- "cognitive_level": The Bloom's Taxonomy level required to answer the question correctly. Choose from: "Remember", "Understand", "Apply", "Analyze", "Evaluate", or "Create". Think deeply about which cognitive skill is actually tested.
- "estimated_correct_pct": A numeric estimate of the percentage of students expected to answer correctly (consistent with the difficulty category). Make it based on factors such as complexity, question context, inference required, distractors, and detail recall. Put yourself in the shoes of a student who is taking a quiz or test based on this unit, and consider how likely a student is to pick both the correct and incorrect answer, simulating the student's thinking process. You tend to think a question is harder than expected (putting into a lower percentage correct category), so please evaluate the question thoroughly and based on real evidence from student abilities on similar topics/questions if possible.
- "reasoning": A brief rationale explaining why the question fits its percentage correct assignment, considering factors such as complexity, inference required, answer or detail recall.

Avoid vague phrases like “According to the passage.” Don’t just repeat the answer. Your goal is to help the student learn the concept by explaining it clearly and thoroughly.
All math expressions, formulas, variables, and symbols in the questions, answer choices, and explanations must be written in valid LaTeX format using $...$ for inline math and $$...$$ for display math when appropriate. This ensures proper rendering in LaTeX-supported environments.
Return a valid JSON list of 20 questions. If there is insufficient content, as previously stated, duplicate existing questions to create a total of 20 questions at the appropriate difficulty levels.

Passage:
{text_chunk}
"""

def call_deepseek_api(prompt):
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful educational assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 4500
    }
    response = requests.post(DEEPSEEK_URL, headers=headers, json=data)
    if response.status_code != 200:
        return None, response.text
    return response.json()["choices"][0]["message"]["content"], None

def parse_question_json(text):
    def clean_response_text(raw):
        match = re.search(r"```(?:json)?\s*(.*?)```", raw.strip(), re.DOTALL)
        return match.group(1).strip() if match else raw.strip()

    cleaned = clean_response_text(text)
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, list):
            return obj
    except Exception:
        pass

    if cleaned.startswith("[") and not cleaned.endswith("]"):
        try:
            fixed = cleaned + "]"
            return json.loads(fixed)
        except Exception:
            pass

    objects = re.findall(r"\{.*?\}", cleaned, re.DOTALL)
    parsed_objects = []
    for obj_text in objects:
        try:
            parsed_objects.append(json.loads(obj_text))
        except Exception:
            continue
    if parsed_objects:
        return parsed_objects

    try:
        import json_repair
        repaired = json_repair.loads(cleaned)
        if isinstance(repaired, dict):
            return [repaired]
        if isinstance(repaired, list):
            return repaired
    except Exception:
        pass

    return []

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
        cog = q.get("cognitive_level", "").strip().capitalize()
        try:
            pct = int(q.get("estimated_correct_pct", -1))
        except:
            pct = -1

        if cog in bloom_difficulty_ranges and 0 <= pct <= 100:
            low, high = bloom_difficulty_ranges[cog]
            if low <= pct <= high:
                valid.append(q)
            else:
                invalid.append(q)
        else:
            invalid.append(q)

    return valid, invalid

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
    search_range = (
        range(next_diff + 1, 9) if going_up else range(next_diff - 1, 0, -1)
    )
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

def accuracy_on_levels(answers, levels):
    filtered = [c for d, c in answers if d in levels]
    return sum(filtered) / len(filtered) if filtered else 0

def compute_mastery_score(answers):
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
            band_score = normalized_score * weight
            band_scores.append(band_score)
    if not band_scores:
        return 0  
    return int(round(max(band_scores)))

# --- Mastery Bar Function ---
def render_mastery_bar(score):
    if score < 30:
        color = "red"
        text_color = "white"
    elif score < 70:
        color = "yellow"
        text_color = "black"
    else:
        color = "green"
        text_color = "white"

    st.markdown(f"""
    <style>
        .mastery-bar-wrapper {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            z-index: 9999;
            background-color: white;
            padding: 8px 16px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        }}
        .mastery-bar {{
            border: 1px solid #ccc;
            border-radius: 8px;
            overflow: hidden;
            height: 24px;
            width: 100%;
            background-color: #eee;
        }}
        .mastery-bar-fill {{
            height: 100%;
            width: {score}%;
            background-color: {color};
            text-align: center;
            color: {text_color};
            font-weight: bold;
            line-height: 24px;
        }}
        .spacer {{
            height: 60px;
        }}
    </style>

    <div class="mastery-bar-wrapper">
        <div class="mastery-bar">
            <div class="mastery-bar-fill">{score}%</div>
        </div>
    </div>
    <div class="spacer"></div>
    """, unsafe_allow_html=True)

# --- 🔥 FIX: Always render mastery bar on every rerun ---
if "quiz_state" in st.session_state:
    score = compute_mastery_score(st.session_state["quiz_state"].get("answers", []))
else:
    score = 0
render_mastery_bar(score)

# --- Main App ---
st.title("AscendQuiz")

if "all_questions" not in st.session_state:
    st.markdown("""
Welcome to your personalized learning assistant — an AI-powered tool that transforms any PDF into a mastery-based, computer-adaptive quiz.

**How it works:**
... (rest of text unchanged)
""")

    uploaded_pdf = st.file_uploader("Upload class notes (PDF)", type="pdf")
    if uploaded_pdf:
        with st.spinner("Generating questions..."):
            chunks = extract_text_from_pdf(uploaded_pdf)
            if len(chunks) <= 2:
                grouped_chunks = ["\n\n".join(chunks)]
            else:
                grouped_chunks = ["\n\n".join(chunks[i:i+4]) for i in range(0, len(chunks), 4)]

            all_questions = []
            chunks_to_use = grouped_chunks[:2] if len(grouped_chunks) >= 2 else [grouped_chunks[0], grouped_chunks[0]]

            for chunk in chunks_to_use:
                prompt = generate_prompt(chunk)
                response_text, error = call_deepseek_api(prompt)
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
                st.success("✅ Questions generated! Starting the quiz...")
                st.session_state.quiz_ready = True
                st.rerun()
            else:
                st.error("No questions were generated.")

elif "quiz_ready" in st.session_state and st.session_state.quiz_ready:
    all_qs = st.session_state.questions_by_difficulty
    state = st.session_state.get("quiz_state", None)

    if state is None:
        st.warning("Quiz state not found. Please restart the app or re-upload a PDF.")
        st.stop()

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
            return re.sub(r"^[A-Da-d][\).:\-]?\s+", "", text).strip()

        option_labels = ["A", "B", "C", "D"]
        cleaned_options = [strip_leading_label(opt) for opt in q["options"]]
        rendered_options = []
        for label, text in zip(option_labels, cleaned_options):
            if "$" in text or "\\" in text:
                rendered_text = f"{label}. $${text}$$"
            else:
                rendered_text = f"{label}. {text}"
            rendered_options.append(rendered_text)

        selected = st.radio("Select your answer:", options=rendered_options, key=f"radio_{idx}", index=None)

        if st.button("Submit Answer", key=f"submit_{idx}") and not state.get("show_explanation", False):
            selected_letter = selected.split(".")[0].strip().upper()
            letter_to_index = {"A": 0, "B": 1, "C": 2, "D": 3}
            correct_letter = q["correct_answer"].strip().upper()
            correct = (selected_letter == correct_letter)
            state["answers"].append((state["current_difficulty"], int(correct)))
            state["asked"].add((state["current_difficulty"], idx))
            state["last_correct"] = correct
            state["last_explanation"] = q["explanation"]
            state["show_explanation"] = True
            st.session_state.quiz_state = state
            st.rerun()

        if state.get("show_explanation", False):
            correct_letter = q["correct_answer"].strip().upper()
            correct_opt_text = cleaned_options[option_labels.index(correct_letter)]
            correctness_msg = "✅ Correct!" if state["last_correct"] else f"❌ Incorrect. The correct answer was {correct_letter}: {correct_opt_text}"
            st.markdown(f"### {correctness_msg}")
            st.markdown("### Explanation")
            st.markdown(state["last_explanation"], unsafe_allow_html=True)

            if st.button("Next Question"):
                going_up = state["last_correct"]
                new_diff = find_next_difficulty(state["current_difficulty"], going_up, state["asked"], all_qs)
                state["current_difficulty"] = new_diff
                diff, idx, q = get_next_question(new_diff, state["asked"], all_qs)
                if q is None:
                    state["quiz_end"] = True
                else:
                    state["current_q"], state["current_q_idx"] = q, idx
                state["show_explanation"] = False
                st.session_state.quiz_state = state
                st.rerun()

    if state["quiz_end"]:
        st.markdown("### 🏁 Quiz Complete")
        score = compute_mastery_score(state["answers"])
        st.markdown(f"**Your mastery score: {score}%**")
