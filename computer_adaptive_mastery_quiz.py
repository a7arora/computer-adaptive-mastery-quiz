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
    return f"""..."""  # Unchanged prompt string — keep your original here

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

def generate_more_questions(pdf_file, existing_q_count=0):
    chunks = extract_text_from_pdf(pdf_file)
    grouped_chunks = ["\n\n".join(chunks[i:i+4]) for i in range(0, len(chunks), 4)]
    all_questions = []
    for chunk in grouped_chunks[existing_q_count // 15:]:
        prompt = generate_prompt(chunk)
        response_text, error = call_groq_api(prompt)
        if error:
            st.error("API error during regeneration: " + error)
            continue
        parsed = parse_question_json(response_text)
        all_questions.extend(parsed)
        if len(all_questions) >= 15:
            break
    return all_questions

def accuracy_on_levels(answers, levels):
    filtered = [c for d, c in answers if d in levels]
    return sum(filtered) / len(filtered) if filtered else 0

def difficulty_weight(d):
    return d ** 1.5

def calculate_mastery_score(answers):
    if not answers:
        return 0
    weighted_correct = sum(c * difficulty_weight(d) for d, c in answers)
    total_weight = sum(difficulty_weight(d) for d, _ in answers)
    return round(100 * weighted_correct / total_weight, 1)

def mastery_color(score):
    if score < 40:
        return "red"
    elif score < 70:
        return "orange"
    else:
        return "green" 

# === STREAMLIT APP ===
st.title("AscendQuiz")

if "all_questions" not in st.session_state:
    st.markdown("""
Welcome to your personalized learning assistant — an AI-powered tool that transforms any PDF into a mastery-based, computer-adaptive quiz.

**How it works:**
This app uses a large language model (LLM) and an adaptive difficulty engine to create multiple-choice questions from your uploaded notes or textbook excerpts. These questions are labeled with how likely students are to answer them correctly, allowing precise control over quiz difficulty.

The quiz adapts in real-time based on your performance. Starting at a medium level, each correct answer raises the difficulty, while incorrect answers lower it — just like the GRE or ALEKS. Once you get 5 hard questions (difficulty level 6 or above) correct at a 75%+ rate, the system considers you to have achieved **mastery** and ends the quiz.

Each question includes:
- Four answer options
- The correct answer
- An explanation
- A predicted correctness percentage

Unlike static tools like Khanmigo, this app uses generative AI to dynamically create the quiz from **your own content** — no rigid question banks required.

---

🧠 **Built using the meta-llama/llama-4-scout-17b model via Groq**, this app is a proof-of-concept showing what modern AI can do for personalized education. It blends mastery learning, real-time feedback, and adaptive testing into one clean experience.

---
""")
    uploaded_pdf = st.file_uploader("Upload class notes (PDF)", type="pdf")
    if uploaded_pdf:
        st.session_state["uploaded_pdf"] = uploaded_pdf
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
                    "regenerated": False
                }
                st.success("✅ Questions generated! Starting the quiz...")
                st.session_state.quiz_ready = True
                st.rerun()
            else:
                st.error("No questions were generated.")

elif "quiz_ready" in st.session_state and st.session_state.quiz_ready:
    all_qs = st.session_state.questions_by_difficulty
    state = st.session_state.quiz_state
    mastery_score = calculate_mastery_score(state["answers"])
    color = mastery_color(mastery_score)
    st.markdown(f"""
    <div style='margin-top:20px; margin-bottom:10px; font-weight:bold;'>📊 Mastery Progress</div>
    <div style='background-color:#ddd; border-radius:10px; height:24px; width:100%;'>
        <div style='background-color:{color}; width:{mastery_score}%; height:100%; border-radius:10px; text-align:center; color:white; font-weight:bold;'>
        {mastery_score}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not state["quiz_end"]:
        if state["current_q"] is None and not state.get("show_explanation", False):
            diff, idx, q = get_next_question(state["current_difficulty"], state["asked"], all_qs)
            if q is None and not state.get("regenerated", False):
                if "uploaded_pdf" in st.session_state:
                    st.info("🔄 Generating additional questions...")
                    new_qs = generate_more_questions(st.session_state["uploaded_pdf"], existing_q_count=len(st.session_state.all_questions))
                    if new_qs:
                        st.session_state.all_questions.extend(new_qs)
                        new_grouped = group_by_difficulty(new_qs)
                        for d, lst in new_grouped.items():
                            st.session_state.questions_by_difficulty[d].extend(lst)
                        state["regenerated"] = True
                        st.rerun()
                    else:
                        state["quiz_end"] = True
                else:
                    state["quiz_end"] = True
            elif q is None:
                state["quiz_end"] = True
            else:
                state["current_q"] = q
                state["current_q_idx"] = idx
                state["current_difficulty"] = diff

    if not state["quiz_end"] and state["current_q"]:
        ...
        # Leave rest of quiz logic here unchanged (question display, explanation, next question, etc.)

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
            del st.session_state.uploaded_pdf
            st.rerun()
