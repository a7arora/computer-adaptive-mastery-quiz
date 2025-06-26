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
    return f"""
You are an educational assistant helping teachers generate multiple choice questions from a passage.

Given the following passage or notes, generate exactly 15 multiple choice questions that test comprehension and critical thinking. The questions must vary in difficulty. If there is not enough content to write 15 good questions, repeat or expand the material, or create additional plausible questions that still test content that is similar to what is in the passage. If the passage is too short to reasonably support 15 distinct questions, generate as many high-quality questions as possible (minimum of 5), ensuring they reflect varying difficulty.

**Requirements**:
- 5 easy (≥85%), 5 medium (60–84%), 5 hard (<60%)

**Each question must include the following fields:**

- "question": A clear, concise, and unambiguous question directly related to the passage that aligns with key learning objectives. The question should be designed to test understanding of material covered in the passage and should be made so it could show up on an educational assessment testing material from this passage. The question should be cognitively appropriate for the specified difficulty level, encouraging critical thinking, application, analysis, or synthesis rather than rote recall if not at the easiest difficulty level. Avoid overly complex wording or ambiguity to ensure students understand exactly what is being asked. Furthermore, make sure that the question has all the context in itself and does not reference specific figures or pages in the passage, as the question is designed for the user to do independently without the passage.
- "options": A list of 4 plausible answer choices labeled "A", "B", "C", and "D"(with one of them being the correct answer). If the question is of medium or hard difficulty, please come up with wrong answers that are ones that a user who does not know the concept well or makes an error would select. Please make sure that only one answer choice is correct by solving the problem and checking all of the answer choices carefully and thoroughly.
- "correct_answer": The letter ("A", "B", "C", or "D") corresponding to the correct option.
- "explanation": A deep, pedagogically useful explanation that teaches the student **why** the correct answer is right and the others are wrong. It should: 1) Start by identifying the correct letter and its full text., 2) Then **explain the reasoning or concept** behind it in a way that would help a confused student understand — using analogies, examples, or logic as needed. Avoid simply saying “According to the passage” or restating the answer., 3) For each wrong option, name its letter and full text, then explain **why it’s wrong or misleading**, including what error a student might make if they chose it., 4) If relevant, clarify common misconceptions or subtle differences. This explanation should act like a mini-tutorial, helping students learn from their mistake or reinforce their understanding.
- "estimated_correct_pct": A numeric estimate of the percentage of students expected to answer correctly (consistent with the difficulty category). Make it based on factors such as complexity, inference required, or detail recall.
- "reasoning": A brief rationale explaining why the question fits its percentage correct assignment, considering factors such as complexity, inference required, or detail recall.

Return a valid JSON list of up to 15 questions. If there is insufficient content, generate as many high-quality questions as possible (minimum 5).

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

🧠 **Built using the `meta-llama/llama-4-scout-17b` model via Groq**, this app is a proof-of-concept showing what modern AI can do for personalized education. It blends mastery learning, real-time feedback, and adaptive testing into one clean experience.

---
""")

    uploaded_pdf = st.file_uploader("Upload class notes (PDF)", type="pdf")
    if uploaded_pdf:
        with st.spinner("Generating questions..."):
            chunks = extract_text_from_pdf(uploaded_pdf)
            # Adaptive chunking
            if len(chunks) <= 2:
                grouped_chunks = ["\n\n".join(chunks)]  # Treat as one full chunk
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

        st.markdown(f"### Question (Difficulty {state['current_difficulty']})")
        st.write(q["question"])

        # Display options as "A. Option text" (no duplicated letter)
        def strip_leading_label(text):
            # Removes A), A., A:, A - etc.
            return re.sub(r"^[A-Da-d][\).:\-]?\s+", "", text).strip()

        option_labels = ["A", "B", "C", "D"]
        cleaned_options = [strip_leading_label(opt) for opt in q["options"]]
        options = [f"{label}. {text}" for label, text in zip(option_labels, cleaned_options)]
        selected = st.radio("Select your answer:", options=options, key=f"radio_{idx}")

        if st.button("Submit Answer", key=f"submit_{idx}") and not state.get("show_explanation", False):
            # Extract selected letter (before the dot)
            selected_letter = selected.split(".")[0].strip().upper()

            # Map correct_answer letter to index
            letter_to_index = {"A": 0, "B": 1, "C": 2, "D": 3}
            correct_letter = q["correct_answer"].strip().upper()
            correct_index = letter_to_index.get(correct_letter, None)

            if correct_index is None:
                st.error("⚠️ Question error: Correct answer letter invalid.")
                state["quiz_end"] = True
                st.stop()

            correct = (selected_letter == correct_letter)

            # Record answer
            state["asked"].add((state["current_difficulty"], idx))
            state["answers"].append((state["current_difficulty"], correct))
            state["last_correct"] = correct
            state["last_explanation"] = q["explanation"]
            state["show_explanation"] = True

            # Check mastery
            hard_correct = [1 for d, c in state["answers"] if d >= 6 and c]
            if len(hard_correct) >= 5 and sum(hard_correct) / len(hard_correct) >= 0.75:
                state["quiz_end"] = True

        if state.get("show_explanation", False):
            if state["last_correct"]:
                st.success("✅ Correct!")
            else:
                st.error(f"❌ Incorrect. {state['last_explanation']}")

            if st.button("Next Question"):
                # Adjust difficulty based on correctness
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
