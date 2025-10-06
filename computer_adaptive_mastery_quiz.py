
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

**Each question must include the following fields:**

- "question": A clear, concise, and unambiguous question that tests understanding of concepts from the passage. The question should be COMPLETELY SELF-CONTAINED with all necessary context included. Never reference "the passage," "the text," specific examples by position (first, second, etc.), or figures/tables. Ask about the concept directly. Make the question slightly more difficult than typical for the specified difficulty level. Ensure it tests conceptual understanding that would be valuable for learning, not memorization of text structure.

- "options": A list of 4 plausible answer choices labeled "A", "B", "C", and "D" (with one being correct). For medium/hard questions, create wrong answers that reflect common misconceptions. Ensure only one answer is clearly correct.

- "correct_answer": The letter ("A", "B", "C", or "D") corresponding to the correct option.

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

Return a valid JSON list of 20 questions. Focus on testing conceptual understanding rather than text memorization.

If the passage contains code or table output, generate questions about how the code works and what outputs mean - but present these as general programming/analysis questions, not as references to "the code shown" or "the table above."

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


def clean_response_text(text: str) -> str:
    """
    Extracts the JSON part from a model response.
    Strips ```json fences, trailing commentary, and truncates at the last bracket.
    """
    text = text.strip()

    # Remove ```json ... ``` fences (more flexible pattern)
    fence_patterns = [
        r"```json\s*(.*?)```",  # ```json content ```
        r"```\s*(.*?)```",      # ``` content ```
        r"`{3,}\s*json\s*(.*?)`{3,}",  # Multiple backticks with json
        r"`{3,}\s*(.*?)`{3,}"   # Multiple backticks without json
    ]
    
    for pattern in fence_patterns:
        fence_match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if fence_match:
            text = fence_match.group(1).strip()
            break

    # Find the JSON array boundaries
    # Look for the first '[' and last ']'
    start_idx = text.find('[')
    end_idx = text.rfind(']')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        text = text[start_idx:end_idx + 1]
        return text.strip()

    # Fallback: look for object boundaries
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        text = text[start_idx:end_idx + 1]
        return text.strip()

    return text


def repair_json(text: str) -> str:
    """
    Repairs common JSON formatting issues
    """
    # Remove trailing commas before ] or }
    text = re.sub(r',\s*([\]}])', r'\1', text)

    # Fix }{ into }, {
    text = re.sub(r'}\s*{', r'}, {', text)

    # Fix ] [ into ], [
    text = re.sub(r']\s*\[', r'], [', text)

    # Replace percent signs in numbers (e.g. 92% -> 92)
    text = re.sub(r'(\d+)\s*%', r'\1', text)

    # Fix unescaped quotes in strings (more conservative approach)
    # This is a simplified fix - for more complex cases, you might need a proper JSON parser
    text = re.sub(r'(?<!\\)"([^"]*?)(?<!\\)"(?=\s*[,\]}])', lambda m: '"' + m.group(1).replace('"', '\\"') + '"', text)
    
    # Ensure the text starts with [ if it looks like an array
    text = text.strip()
    if not text.startswith('[') and not text.startswith('{'):
        # Try to find the start of JSON
        json_start = re.search(r'[\[{]', text)
        if json_start:
            text = text[json_start.start():]
    
    return text


def parse_question_json(text: str):
    """
    Parse JSON with better error handling and debugging
    """
    # Print raw text for debugging
    print(f"Raw API response length: {len(text)}")
    print(f"Raw API response (first 200 chars): {text[:200]}")
    
    cleaned = clean_response_text(text)
    print(f"Cleaned text length: {len(cleaned)}")
    print(f"Cleaned text (first 200 chars): {cleaned[:200]}")
    
    cleaned = repair_json(cleaned)
    print(f"Repaired text (first 200 chars): {cleaned[:200]}")

    # Try standard JSON parsing first
    try:
        result = json.loads(cleaned)
        print(f"Successfully parsed {len(result) if isinstance(result, list) else 1} questions")
        return result
    except json.JSONDecodeError as e:
        print(f"Standard JSON parsing failed: {e}")
        
        # Try json5 as fallback (more lenient parsing)
        try:
            import json5
            result = json5.loads(cleaned)
            print(f"JSON5 parsing successful: {len(result) if isinstance(result, list) else 1} questions")
            return result
        except Exception as e2:
            print(f"JSON5 parsing also failed: {e2}")
            
            # Final fallback: try to extract individual questions manually
            try:
                # Look for question objects and try to parse them individually
                questions = []
                # Split by question boundaries and try to parse each
                question_pattern = r'\{\s*"question":[^}]*?"reasoning":[^}]*?\}'
                potential_questions = re.findall(question_pattern, cleaned, re.DOTALL)
                
                for q_text in potential_questions:
                    try:
                        q_obj = json.loads(q_text)
                        questions.append(q_obj)
                    except:
                        continue
                
                if questions:
                    print(f"Manual extraction successful: {len(questions)} questions")
                    return questions
            except:
                pass
            
            # Log the error with more context
            st.error("⚠️ JSON parse failed:")
            st.error(f"Standard JSON error: {e}")
            st.error(f"JSON5 error: {e2}")
            st.text("Raw cleaned text (first 1000 chars):")
            st.text(cleaned[:1000])
            
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
        if not isinstance(q, dict):  # skip anything not a dict
            invalid.append(q)
            continue

        cog = str(q.get("cognitive_level", "")).strip().capitalize()
        try:
            pct = int(q.get("estimated_correct_pct", -1))
        except Exception:
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

    # Try one step in intended direction
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
    #limits mastery scores based on difficulty attempted
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
        #does not give mastery points for guessing, especially on hard questions
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

#app frontend
st.title("AscendQuiz")
def render_mastery_bar(score):
    if score < 30:
        color = "#dc3545"  # red
        text_color = "white"
    elif score < 70:
        color = "#ffc107"  # yellow
        text_color = "black"
    else:
        color = "#28a745"  # green
        text_color = "white"

    st.markdown(f"""
    <style>
        /* Hide Streamlit's default header padding */
        .stApp {{
            padding-top: 70px;
        }}
        
        .mastery-bar-wrapper {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            width: 100%;
            z-index: 999999;
            background-color: white;
            padding: 12px 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            border-bottom: 1px solid #ddd;
        }}
        
        .mastery-bar {{
            border: 2px solid #ccc;
            border-radius: 8px;
            overflow: hidden;
            height: 28px;
            width: 100%;
            background-color: #f0f0f0;
            position: relative;
        }}
        
        .mastery-bar-fill {{
            height: 100%;
            width: {score}%;
            background-color: {color};
            transition: width 0.3s ease;
            position: absolute;
            top: 0;
            left: 0;
        }}
        
        .mastery-bar-text {{
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: {text_color if score > 10 else 'black'};
            font-weight: bold;
            font-size: 14px;
            z-index: 2;
            text-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }}
    </style>

    <div class="mastery-bar-wrapper">
        <div style="font-size: 12px; margin-bottom: 4px; color: #666; font-weight: 500;">
            Mastery Progress
        </div>
        <div class="mastery-bar">
            <div class="mastery-bar-fill"></div>
            <div class="mastery-bar-text">{score}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


if "all_questions" not in st.session_state:
    st.markdown("""
Welcome to your personalized learning assistant — an AI-powered tool that transforms any PDF into a mastery-based, computer-adaptive quiz.

**How it works:**
This app uses a large language model (LLM) and an adaptive difficulty engine to create multiple-choice questions from your uploaded notes or textbook excerpts. These questions are labeled with how likely students are to answer them correctly, allowing precise control over quiz difficulty.

The quiz adapts in real-time based on your performance. Starting at a medium level, each correct answer raises the difficulty, while incorrect answers lower it — just like the GRE or ALEKS. Once your **mastery score reaches 70% or higher** (calculated using your accuracy weighted by difficulty level), the system considers you to have achieved **mastery** and ends the quiz.

Each question includes:
- Four answer options
- The correct answer
- An explanation
- A predicted correctness percentage

Unlike static tools like Khanmigo, this app uses generative AI to dynamically create the quiz from **your own content** — no rigid question banks required.

---

**Built using the DeepSeek-R1-0528 model**, this app is a proof-of-concept showing what modern AI can do for personalized education. It blends mastery learning, real-time feedback, and adaptive testing into one clean experience. Please keep in mind that it currently takes about 4-5 minutes to generate questions from a pdf... please be patient as it generates questions. Furthermore, it only accepts text output and cannot read handwriting or drawings at this time.

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
            # Pick first 2 chunks or duplicate the first if only one exists
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
        st.markdown(q["question"], unsafe_allow_html=True)
        def strip_leading_label(text):
            # Removes A), A., A:, A - etc.
            return re.sub(r"^[A-Da-d][\).:\-]?\s+", "", text).strip()

        option_labels = ["A", "B", "C", "D"]
        cleaned_options = [strip_leading_label(opt) for opt in q["options"]]
        rendered_options = []
        for label, text in zip(option_labels, cleaned_options):
            # Wrap LaTeX content in markdown with inline math if any '$' is present
            if "$" in text or "\\" in text:
                rendered_text = f"{label}. $${text}$$"
            else:
                rendered_text = f"{label}. {text}"
            rendered_options.append(rendered_text)

        selected = st.radio("Select your answer:", options=rendered_options, key=f"radio_{idx}", index=None)

        if st.button("Submit Answer", key=f"submit_{idx}") and not state.get("show_explanation", False):
            if selected is None:
                st.warning("Please select an answer before submitting.")
            else:
                selected_letter = selected.split(".")[0].strip().upper()
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

                # Update mastery score according to quiz question
                score = compute_mastery_score(state["answers"])
                #end when students reach mastery at 70/100
                if score >= 70:
                    state["quiz_end"] = True

        if state.get("show_explanation", False):
            if state["last_correct"]:
                st.success("✅ Correct!")
            else:
                st.markdown("❌ **Incorrect.**", unsafe_allow_html=True)
                st.markdown(state["last_explanation"], unsafe_allow_html=True)

            if st.button("Next Question"):
                # Adjust difficulty
                def find_next_difficulty(current_diff, going_up, asked, all_qs):
                    diffs = range(current_diff + 1, 9) if going_up else range(current_diff - 1, 0, -1)
                    for d in diffs:
                        if pick_question(d, asked, all_qs):
                            return d
                    return current_diff  # fallback to current if no higher/lower available

                # Adjust difficulty based on performance, scaffolding learning and challenging students
                if state["last_correct"]:
                    state["current_difficulty"] = find_next_difficulty(
                    state["current_difficulty"], going_up=True, asked=state["asked"], all_qs=all_qs
                    )
                else:
                    state["current_difficulty"] = find_next_difficulty(
                    state["current_difficulty"], going_up=False, asked=state["asked"], all_qs=all_qs
                    )
                # Clear current question to trigger fetching a new one
                state["current_q"] = None
                state["current_q_idx"] = None
                state["show_explanation"] = False
                state["last_correct"] = None
                state["last_explanation"] = None
                st.rerun()

    elif state["quiz_end"]:
        acc = accuracy_on_levels(state["answers"], [5, 6, 7, 8])
        hard_attempts = len([1 for d, _ in state["answers"] if d >= 5])
        st.markdown("## Quiz Completed 🎉")
        if score >= 70:
            st.success(f"🎉 You have mastered the content! Your mastery score is {score}%. Great job!")
        else:
            st.warning(f"Mastery not yet achieved. Your mastery score is {score}%. Review the material and try again.")

        if "all_questions" in st.session_state:
            all_qs_json = json.dumps(st.session_state.all_questions, indent=2)
            st.download_button(
                label="📥 Download All Quiz Questions (JSON)",
                data=all_qs_json,
                file_name="ascendquiz_questions.json",
                mime="application/json"
            )














