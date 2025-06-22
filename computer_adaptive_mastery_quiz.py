import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import requests
import json
import re
import random
import html

# === CONFIGURATION ===
API_KEY = st.secrets["GROQ_API_KEY"]
HEADERS_JSON = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

# === FUNCTION DEFINITIONS ===
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return [page.get_text() for page in doc if page.get_text().strip()]

def generate_prompt(text_chunk):
    return f"""
You are an expert educational content designer with experience in assessment development, instructional design, and psychometrics. Your task is to generate high-quality multiple choice questions from the provided academic text. Prioritize clarity, validity, and pedagogical value.

Passage:
{text_chunk}

**Your task:**
- Generate exactly 15 multiple choice questions based on the above passage.
- The questions must test comprehension, reasoning, and higher-order thinking — not just factual recall.
- Use Bloom's Taxonomy: include a mix of knowledge, application, analysis, evaluation, and synthesis levels.
- Avoid ambiguous, trivial, or overly literal questions.
- When writing mathematical expressions or formulas, format them properly using LaTeX syntax, enclosed inside dollar signs $...$ for inline expressions.

**Difficulty Distribution (based on estimated correct %):**
- 5 easy (≥85% correct)
- 5 medium (60–84% correct)
- 5 hard (<60% correct)

**For each question, return a dictionary with the following fields:**
- "question": The question text (can include LaTeX inside $...$)
- "options": A list of exactly 4 plausible, distinct answer choices (can include LaTeX inside $...$)
- "correct_answer": The exact text of the correct option (must match one of the options verbatim)
- "explanation": A concise, clear explanation of why the correct answer is correct.
- "estimated_correct_pct": Estimated % of students who would correctly answer this question.
- "estimated_correct_pct_reasoning": Data-driven justification for that estimated percentage.
- "question_quality_rating": Score out of 10 indicating how well this question assesses learning objectives.
- "question_quality_reasoning": Explain why you gave that quality rating.
- "reasoning": Rationale for why this question was written.
- "distractor_rationales": A dictionary with keys 'A', 'B', 'C', and 'D'. For each key, explain in 1-2 sentences **why a student might plausibly choose that option**, even though it is incorrect. Each explanation must be **specific to that option** and **grounded in the passage content**.

**Critical constraints:**
- Avoid vague wording.
- Ensure correct answer is unambiguously correct.
- Ensure distractors are plausible but clearly incorrect upon reflection.
- Return only valid JSON list of exactly 15 dictionaries. Do not include commentary or markdown.
"""

def call_groq_api(prompt):
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 4000
    }
    response = requests.post(GROQ_URL, headers=HEADERS_JSON, json=data)
    if response.status_code != 200:
        return None, response.text
    return response.json()["choices"][0]["message"]["content"], None

def clean_response_text(text):
    match = re.search(r"```(?:json)?\s*(.*?)```", text.strip(), re.DOTALL)
    return match.group(1).strip() if match else text.strip()

def parse_question_json(text):
    try:
        raw_data = json.loads(clean_response_text(text))
        for q in raw_data:
            if "distractor_rationales" not in q or not isinstance(q["distractor_rationales"], dict):
                q["distractor_rationales"] = {}
            for i, label in enumerate(["A", "B", "C", "D"]):
                if label not in q["distractor_rationales"]:
                    q["distractor_rationales"][label] = "No rationale provided for this option."
        return raw_data
    except Exception as e:
        print(f"JSON parsing error: {e}")
        return []

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

def get_next_question(curr_diff, asked, all_qs):
    for offset in range(0, 8):
        for direction in [-1, 1]:
            d = curr_diff + direction * offset
            if 1 <= d <= 8:
                candidates = pick_question(d, asked, all_qs)
                if candidates:
                    return d, *random.choice(candidates)
    return None, None, None

def normalize_answer(ans):
    return ans.replace(" ", "").replace("\\_", "_").lower()

def accuracy_on_levels(answers, levels):
    filtered = [c for d, c in answers if d in levels]
    return sum(filtered) / len(filtered) if filtered else 0

def render_text_with_latex(text):
    if "$" in text:
        st.latex(text)
    else:
        st.write(text)

# === STREAMLIT APP ===
st.title("AscendQuiz: LaTeX Adaptive Mastery Edition")

if "all_questions" not in st.session_state:
    st.markdown("""
## 🎓AscendQuiz — Now with LaTeX rendering!
Upload your class notes or textbook PDF, and this app will generate mastery-based adaptive quizzes with detailed explanations, distractor rationales, difficulty ratings, and full LaTeX support for formulas.
""")
    uploaded_pdf = st.file_uploader("Upload PDF Notes", type="pdf")
    if uploaded_pdf:
        with st.spinner("Generating questions..."):
            chunks = extract_text_from_pdf(uploaded_pdf)
            grouped_chunks = ["\n\n".join(chunks[i:i+4]) for i in range(0, len(chunks), 4)]
            all_questions = []
            for chunk in grouped_chunks[:5]:
                prompt = generate_prompt(chunk)
                response_text, error = call_groq_api(prompt)
                if error:
                    st.error("API error: " + error)
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
                st.success("✅ Questions generated! Starting quiz...")
                st.session_state.quiz_ready = True
                st.rerun()
            else:
                st.error("No questions generated.")

elif "quiz_ready" in st.session_state and st.session_state.quiz_ready:
    all_qs = st.session_state.questions_by_difficulty
    state = st.session_state.quiz_state

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
        render_text_with_latex(q["question"])

        option_labels = ["A", "B", "C", "D"]
        for label, text in zip(option_labels, q["options"]):
            st.write(f"{label}: ", unsafe_allow_html=True)
            render_text_with_latex(text)

        selected_letter = st.radio("Select your answer:", option_labels, key=f"radio_{idx}")

        if st.button("Submit Answer", key=f"submit_{idx}") and not state.get("show_explanation", False):
            state["selected_letter"] = selected_letter
            correct_index = next(i for i, opt in enumerate(q["options"]) if normalize_answer(opt) == normalize_answer(q["correct_answer"]))
            correct_letter = option_labels[correct_index]
            correct = (selected_letter == correct_letter)

            state["asked"].add((state["current_difficulty"], idx))
            state["answers"].append((state["current_difficulty"], correct))
            state["last_correct"] = correct
            state["last_explanation"] = q["explanation"]
            state["show_explanation"] = True

            hard_correct = [1 for d, c in state["answers"] if d >= 6 and c]
            if len(hard_correct) >= 5 and sum(hard_correct) / len(hard_correct) >= 0.75:
                state["quiz_end"] = True

        if state.get("show_explanation", False):
            if state["last_correct"]:
                st.success("✅ Correct!")
            else:
                st.error("❌ Incorrect.")
            st.write(f"**Explanation:** {state['last_explanation']}")

            distractors = q.get("distractor_rationales", {})
            rationale = distractors.get(state["selected_letter"], None)
            st.write(f"**Why you might have picked {state['selected_letter']}:** {rationale or 'No rationale provided.'}")

            if st.button("Next Question"):
                state["current_difficulty"] = min(8, state["current_difficulty"] + 1) if state["last_correct"] else max(1, state["current_difficulty"] - 1)
                state["current_q"] = None
                state["current_q_idx"] = None
                state["show_explanation"] = False
                st.rerun()

    elif state["quiz_end"]:
        acc = accuracy_on_levels(state["answers"], [6,7,8])
        hard_attempts = len([1 for d, _ in state["answers"] if d >= 6])
        st.markdown("## Quiz Completed 🎉")
        st.markdown(f"Accuracy on hard questions: {acc:.0%} ({hard_attempts} hard questions attempted)")
        if acc >= 0.75 and hard_attempts >= 5:
            st.success("🎉 Mastery Achieved! You scored 75%+ on hard questions.")
        else:
            st.warning("Mastery not achieved. Keep practicing!")
        if st.button("Restart Quiz"):
            for key in ["all_questions", "questions_by_difficulty", "quiz_state", "quiz_ready"]:
                del st.session_state[key]
            st.rerun()
