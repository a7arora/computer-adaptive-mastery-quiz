import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import requests
import json
import re
import random

# === CONFIGURATION ===
API_KEY = st.secrets["DEEPSEEK_API_KEY"]
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME = "deepseek-chat"

# === FUNCTION DEFINITIONS ===
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return [page.get_text() for page in doc if page.get_text().strip()]

def generate_prompt(text_chunk):
    return f"""
You are an educational assistant helping teachers generate multiple choice questions from a passage.

Given the following passage or notes, generate exactly 8 multiple choice questions that test comprehension and critical thinking. The questions must vary in difficulty. If there is not enough content to write 8 good questions, repeat or expand the material, or create additional plausible questions that still test content that is similar to what is in the passage. If the passage is too short to reasonably support 8 distinct questions, generate as many high-quality questions as possible (minimum of 4), ensuring they reflect varying difficulty.

**Requirements**:
- 2 easy (≥85%), 2 medium (60–84%), 2 medium-hard (40-60%), 2 hard(<40%)

**Each question must include the following fields:**

- "question": A clear, concise, and unambiguous question directly related to the passage that aligns with key learning objectives. The question should be designed to test understanding of material covered in the passage and should be made so it could show up on an educational assessment testing material from this passage. The question should be cognitively appropriate for the specified difficulty level, encouraging critical thinking, application, analysis, or synthesis rather than rote recall if not at the easiest difficulty level. Avoid overly complex wording or ambiguity to ensure students understand exactly what is being asked. Furthermore, make sure that the question has all the context in itself and does not reference specific figures or pages in the passage, as the question is designed for the user to do independently without the passage.
- "options": A list of 4 plausible answer choices labeled "A", "B", "C", and "D"(with one of them being the correct answer). If the question is of medium or hard difficulty, please come up with wrong answers that are ones that a user who does not know the concept well or makes an error would select. Please make sure that only one answer choice is correct by solving the problem and checking all of the answer choices carefully and thoroughly. It should not be ambigiuous which as to whether an answer choice is correct or not.
- "correct_answer": The letter ("A", "B", "C", or "D") corresponding to the correct option.
- "explanation": A deep, pedagogically useful explanation that **teaches the concept** behind the correct answer and analyzes the flaws in the others. The explanation must:
    1. Start by stating the correct letter and full answer.
    2. Teach **why** that answer is correct using **conceptual reasoning** — including how the mechanism works, or why the property matters — not just restating facts.
       - For example, if the correct answer is "correlation between features degrades performance," then explain **why correlated features reduce tree diversity in Random Forests**, which is the core reason performance drops.
       - Use step-by-step reasoning, examples, or analogies when helpful.
    3. For each incorrect answer, state its letter and text, and **explain why it's wrong**, including what misconception a student might have that could lead them to choose it.
    4. The tone should be that of a **tutor or explainer**, helping a confused student understand both the correct idea and the traps in the wrong ones.

Avoid vague phrases like “According to the passage.” Don’t just repeat the answer. Your goal is to help the student learn the concept by explaining it clearly and thoroughly. If any question, answer choice, or explanation involves math, format all equations using LaTeX syntax with `$...$` for inline and `$$...$$` for block-level math.
- "estimated_correct_pct": A numeric estimate of the percentage of students expected to answer correctly (consistent with the difficulty category). Make it based on factors such as complexity, inference required, or detail recall. Make sure that this estimate is grounded in evidence-based facts as to the skills that this question tests and the way that the question tests them. 

Return a valid JSON list of up to 8 questions. If there is insufficient content, generate as many high-quality questions as possible (minimum 4).

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
        "max_tokens": 2048
    }
    response = requests.post(DEEPSEEK_URL, headers=headers, json=data)
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


# === SILENT BACKGROUND CHUNK LOADING ===
def load_next_chunk_in_background():
    state = st.session_state.get("quiz_state", {})
    if (
        "chunk_queue" in st.session_state
        and st.session_state.generated_chunks < st.session_state.scheduled_chunks
        and len(st.session_state.chunk_queue) > st.session_state.generated_chunks
        and state.get("first_question_displayed", False)
    ):
        next_chunk = st.session_state.chunk_queue[st.session_state.generated_chunks]
        prompt = generate_prompt(next_chunk)
        response_text, error = call_deepseek_api(prompt)
        if not error:
            new_qs = parse_question_json(response_text)
            st.session_state.all_questions.extend(new_qs)
            st.session_state.questions_by_difficulty = group_by_difficulty(st.session_state.all_questions)
            st.session_state.generated_chunks += 1
            # silently rerun to continue loading next chunk
            st.rerun()

# === STREAMLIT APP ===

st.title("AscendQuiz")

# Initialize quiz state dict if missing
if "quiz_state" not in st.session_state:
    st.session_state.quiz_state = None

# === If user just uploaded PDF and initial questions not generated ===
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

**Built using the DeepSeek-R1-0528 model**, this app is a proof-of-concept showing what modern AI can do for personalized education. It blends mastery learning, real-time feedback, and adaptive testing into one clean experience.

---
""")

    uploaded_pdf = st.file_uploader("Upload class notes (PDF)", type="pdf")
    if uploaded_pdf:
        with st.spinner("Generating initial questions..."):
            chunks = extract_text_from_pdf(uploaded_pdf)
            grouped_chunks = ["\n\n".join(chunks[i:i+4]) for i in range(0, len(chunks), 4)]

            st.session_state.chunk_queue = grouped_chunks
            st.session_state.generated_chunks = 0
            st.session_state.scheduled_chunks = 5  # total chunks to load (initial + 4 background)

            all_questions = []

            # Generate first chunk synchronously (blocking)
            initial_chunk = grouped_chunks[0]
            prompt = generate_prompt(initial_chunk)
            response_text, error = call_deepseek_api(prompt)
            if not error:
                parsed = parse_question_json(response_text)
                all_questions.extend(parsed)
                st.session_state.generated_chunks = 1

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
                    "first_question_displayed": False,
                }
                st.success("✅ Questions generated! Starting the quiz...")
                st.session_state.quiz_ready = True
                st.rerun()
            else:
                st.error("No questions were generated.")

# === If quiz started and ready ===
elif "quiz_ready" in st.session_state and st.session_state.quiz_ready:

    # Silent background chunk loading before UI renders
    load_next_chunk_in_background()

    all_qs = st.session_state.questions_by_difficulty
    state = st.session_state.get("quiz_state", None)

    if state is None:
        st.warning("Quiz state not found. Please restart the app or re-upload a PDF.")
        st.stop()

    score = compute_mastery_score(state.get("answers", []))
    render_mastery_bar(score)

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
            st.markdown(f"**Question:** {q['question']}", unsafe_allow_html=True)
            state["first_question_displayed"] = True

            def strip_leading_label(text):
                return re.sub(r"^[A-Da-d][\).:\-]?\s+", "", text).strip()

            option_labels = ["A", "B", "C", "D"]
            cleaned_options = [strip_leading_label(opt) for opt in q["options"]]
            for i, (label, text) in enumerate(zip(option_labels, cleaned_options)):
                st.markdown(f"**{label}.** {text}", unsafe_allow_html=True)

            selected = st.radio("Select your answer:", option_labels, key=f"radio_{idx}", index=None)

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
                if state["last_correct"]:
                    st.markdown("✅ **Correct!**", unsafe_allow_html=True)
                else:
                    st.markdown(f"❌ **Incorrect.** {state['last_explanation']}", unsafe_allow_html=True)

                if st.button("Next Question"):
                    def find_next_difficulty(current_diff, going_up, asked, all_qs):
                        diffs = range(current_diff + 1, 9) if going_up else range(current_diff - 1, 0, -1)
                        for d in diffs:
                            if pick_question(d, asked, all_qs):
                                return d
                        return current_diff

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

    else:  # quiz_end = True
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

# === Mastery bar render function from your code ===
