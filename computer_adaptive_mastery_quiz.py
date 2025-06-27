import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import requests
import json
import re
import random

# === CONFIGURATION ===
# Get the API key securely from Streamlit secrets
API_KEY = st.secrets["GROQ_API_KEY"]

headers = {"Authorization": f"Bearer {API_KEY}"}
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

# === FUNCTION DEFINITIONS ===
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return [page.get_text() for page in doc if page.get_text().strip()]

def generate_prompt(text_chunk):
    return f""" ... (no changes in the prompt generation, same as before) ... """

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
        "max_tokens": 4500
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

# 🔹 NEW FUNCTION: Difficulty to label
def difficulty_text_label(level):
    if level in [1, 2]:
        return "easy"
    elif level == 3:
        return "easy-medium"
    elif level in [4, 5]:
        return "medium"
    elif level == 6:
        return "medium-hard"
    elif level in [7, 8]:
        return "hard"
    else:
        return "unknown"

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
    candidates = pick_question(curr_diff, asked, all_qs)
    if candidates:
        return curr_diff, *random.choice(candidates)
    for d in range(curr_diff - 1, 0, -1):
        candidates = pick_question(d, asked, all_qs)
        if candidates:
            return d, *random.choice(candidates)
    for d in range(curr_diff + 1, 9):
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
    st.markdown(""" ... (unchanged Streamlit markdown intro) ... """)
    uploaded_pdf = st.file_uploader("Upload class notes (PDF)", type="pdf")
    if uploaded_pdf:
        with st.spinner("Generating questions..."):
            chunks = extract_text_from_pdf(uploaded_pdf)
            if len(chunks) <= 2:
                grouped_chunks = ["\n\n".join(chunks)]
            else:
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

        question_number = len(state["asked"]) + 1
        st.markdown(f"### Question {question_number}")
        st.write(q["question"])

        def strip_leading_label(text):
            return re.sub(r"^[A-Da-d][\).:\-]?\s+", "", text).strip()

        option_labels = ["A", "B", "C", "D"]
        cleaned_options = [strip_leading_label(opt) for opt in q["options"]]
        options = [f"{label}. {text}" for label, text in zip(option_labels, cleaned_options)]
        selected = st.radio("Select your answer:", options=options, key=f"radio_{idx}", index=None)

        if st.button("Submit Answer", key=f"submit_{idx}") and not state.get("show_explanation", False):
            selected_letter = selected.split(".")[0].strip().upper()
            letter_to_index = {"A": 0, "B": 1, "C": 2, "D": 3}
            correct_letter = q["correct_answer"].strip().upper()
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

            hard_correct = [1 for d, c in state["answers"] if d >= 6 and c]
            if len(hard_correct) >= 5 and sum(hard_correct) / len(hard_correct) >= 0.75:
                state["quiz_end"] = True

    if state.get("show_explanation", False):
        difficulty_level = q.get("difficulty_label")
        diff_text = difficulty_text_label(difficulty_level) if difficulty_level else "unknown"

        if state["last_correct"]:
            st.success(f"✅ Correct! This was a **{diff_text}** difficulty problem.")
        else:
            st.error(f"❌ Incorrect. This was a **{diff_text}** difficulty problem.\n\n{state['last_explanation']}")

        if st.button("Next Question"):
            available_difficulties = [d for d in range(1, 9) if all_qs.get(d)]
            current = state["current_difficulty"]
            if state["last_correct"]:
                next_diffs = sorted([d for d in available_difficulties if d > current])
            else:
                next_diffs = sorted([d for d in available_difficulties if d < current], reverse=True)
            if next_diffs:
                state["current_difficulty"] = next_diffs[0]
            else:
                state["current_difficulty"] = current

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
