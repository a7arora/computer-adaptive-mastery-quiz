import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import requests
import json
import re
import random

# === CONFIGURATION ===
API_KEY = st.secrets["GROQ_API_KEY"]
headers = {"Authorization": f"Bearer {API_KEY}"}
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

# === FUNCTION DEFINITIONS ===
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return [page.get_text() for page in doc if page.get_text().strip()]

def generate_prompt(text_chunk):
    return f"""
You are an educational assistant helping teachers generate multiple choice questions from a passage.

Given the following passage or notes, generate exactly 15 multiple choice questions that test comprehension and critical thinking. The questions must vary in difficulty.

You may use LaTeX formatting for math content, using `$...$` for inline and `$$...$$` for display equations.

**Requirements**:
- 5 easy (≥85%), 5 medium (60–84%), 5 hard (<60%)

**Each question must include the following fields:**

- "question": A clear, concise, and unambiguous question directly related to the passage that aligns with key learning objectives. The question should be cognitively appropriate for the specified difficulty level, encouraging critical thinking, application, analysis, or synthesis rather than rote recall. Avoid overly complex wording or ambiguity to ensure students understand exactly what is being asked.
- "options": A list of 4 plausible answer choices labeled "A", "B", "C", and "D".
- "correct_answer": The letter ("A", "B", "C", or "D") corresponding to the correct option.
- "explanation": A detailed explanation that clearly states why the correct answer choice (including its letter and full text) is right, referencing specific content from the passage. For each incorrect option, explicitly mention its letter and full answer text, and explain why that choice is incorrect or less accurate based on the passage.
- "estimated_correct_pct": A numeric estimate of the percentage of students expected to answer correctly (consistent with the difficulty category). Make it based on factors such as complexity, inference required, or detail recall.
- "reasoning": A brief rationale explaining why the question fits its difficulty category, considering factors such as complexity, inference required, or detail recall.

Return only a valid JSON list of exactly 15 dictionaries.

Passage:
{text_chunk}
"""

def call_groq_api(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 3500
    }
    response = requests.post(GROQ_URL, headers=headers, json=data)
    if response.status_code != 200:
        return None, response.text
    return response.json()["choices"][0]["message"]["content"], None

def clean_response_text(text):
    match = re.search(r"```(?:json)?\s*(.*?)```", text.strip(), re.DOTALL)
    return match.group(1).strip() if match else text.strip()

def parse_question_json(text):
    try:
        return json.loads(clean_response_text(text))
    except Exception:
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

def accuracy_on_levels(answers, levels):
    filtered = [c for d, c in answers if d in levels]
    return sum(filtered) / len(filtered) if filtered else 0

# === STREAMLIT APP ===
st.title("AscendQuiz")

if "all_questions" not in st.session_state:
    st.markdown("""
## 🎓AscendQuiz: Computer Adaptive Mastery Quiz Generator

Welcome to your personalized learning assistant...
""")
    uploaded_pdf = st.file_uploader("Upload class notes (PDF)", type="pdf")
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
                st.success("✅ Questions generated! Starting the quiz...")
                st.session_state.quiz_ready = True
                st.rerun()
            else:
                st.error("No questions were generated.")

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
        st.markdown(q["question"], unsafe_allow_html=True)

        option_labels = ["A", "B", "C", "D"]
        for label, text in zip(option_labels, q["options"]):
            st.markdown(f"**{label}.** {text}", unsafe_allow_html=True)
        selected = st.radio("Select your answer:", options=option_labels, key=f"radio_{idx}")


        if st.button("Submit Answer", key=f"submit_{idx}") and not state.get("show_explanation", False):
            selected_letter = selected.split(".")[0].strip().upper()
            correct_letter = q["correct_answer"].strip().upper()
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
            st.markdown(state["last_explanation"], unsafe_allow_html=True)

            if st.button("Next Question"):
                if state["last_correct"]:
                    state["current_difficulty"] = min(8, state["current_difficulty"] + 1)
                else:
                    state["current_difficulty"] = max(1, state["current_difficulty"] - 1)

                state["current_q"] = None
                state["current_q_idx"] = None
                state["show_explanation"] = False
                st.rerun()

    elif state["quiz_end"]:
        acc = accuracy_on_levels(state["answers"], [6, 7, 8])
        hard_attempts = len([1 for d, _ in state["answers"] if d >= 6])
        st.markdown("## Quiz Completed 🎉")
        st.markdown(f"Accuracy on hard questions: {acc:.0%} ({hard_attempts} hard questions attempted)")

        if acc >= 0.75 and hard_attempts >= 5:
            st.success("🎉 You have mastered the content, achieving 75%+ accuracy on hard questions. Great job!")
        else:
            st.warning("Mastery was not achieved. Please review the material and try again.")

        if st.button("Restart Quiz"):
            del st.session_state.all_questions
            del st.session_state.questions_by_difficulty
            del st.session_state.quiz_state
            del st.session_state.quiz_ready
            st.rerun()
