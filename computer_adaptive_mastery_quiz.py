import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import requests
import json
import re
import random
import time
from typing import List, Dict, Tuple, Optional

API_KEY = st.secrets["ANTHROPIC_API_KEY"]

headers = {
    "x-api-key": f"{API_KEY}",
    "anthropic-version": "2023-06-01",
    "Content-Type": "application/json"
}
CLAUDE_URL = "https://api.anthropic.com/v1/messages"
MODEL_NAME = "claude-sonnet-4-20250514"

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF with error handling"""
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        pages = []
        for page in doc:
            text = page.get_text()
            if text.strip():  # Only include non-empty pages
                pages.append(text)
        doc.close()
        return pages
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return []

def generate_prompt(text_chunk):
    return f"""
You are a teacher who is designing a test with multiple choice questions (each with 4 answer choices) to test content from a passage.

Given the following passage or notes, generate exactly 8 multiple choice questions that test comprehension and critical thinking. The questions must vary in difficulty. If there is not enough content to write 8 good questions, repeat or expand the material, or create additional plausible questions that still test content that is similar to what is in the passage.

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
- 2 easy (>=85%), 2 medium (60–84%), 2 medium-hard (40-60%), 2 hard(<40%).(These are the percentages of students are expected to get correctly on a multiple choiced assessment testing this content)
You tend to make the questions easier than the respective labels(for instance, you make hard questions that 60% of students answer correctly or medium-hard questions that 70% of students answer correctly), so please try to make the questions more significantly more challenging than you would think they should be for medium-hard and hard questions

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

Return a valid JSON list of 8 questions. Focus on testing conceptual understanding rather than text memorization.

If the passage contains code or table output, generate questions about how the code works and what outputs mean - but present these as general programming/analysis questions, not as references to "the code shown" or "the table above."

Passage:
{text_chunk}
"""

def call_claude_api(prompt: str, max_retries: int = 3, base_delay: float = 2.0) -> Tuple[Optional[str], Optional[str]]:
    """
    Call Claude API with exponential backoff retry logic
    """
    data = {
        "model": MODEL_NAME,
        "max_tokens": 4500,
        "temperature": 0.7,
        "system": "You are a helpful educational assistant.",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(CLAUDE_URL, headers=headers, json=data, timeout=60)
            
            if response.status_code == 200:
                return response.json()["content"][0]["text"], None
            elif response.status_code == 529:  # Overloaded
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                st.warning(f"API overloaded. Retrying in {delay:.1f} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue
            elif response.status_code == 500:  # Internal server error
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                st.warning(f"Internal server error. Retrying in {delay:.1f} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                if attempt == max_retries - 1:
                    return None, error_msg
                st.warning(f"API error (attempt {attempt + 1}/{max_retries}): {error_msg}")
                time.sleep(base_delay * (2 ** attempt))
                continue
                
        except requests.exceptions.Timeout:
            if attempt == max_retries - 1:
                return None, "Request timeout"
            st.warning(f"Request timeout. Retrying... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(base_delay * (2 ** attempt))
            continue
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                return None, f"Request error: {str(e)}"
            st.warning(f"Request error. Retrying... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(base_delay * (2 ** attempt))
            continue
    
    return None, "Max retries exceeded"

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
    text = re.sub(r'(?<!\\)"([^"]*?)(?<!\\)"(?=\s*[,\]}])', lambda m: '"' + m.group(1).replace('"', '\\"') + '"', text)
    
    # Ensure the text starts with [ if it looks like an array
    text = text.strip()
    if not text.startswith('[') and not text.startswith('{'):
        # Try to find the start of JSON
        json_start = re.search(r'[\[{]', text)
        if json_start:
            text = text[json_start.start():]
    
    return text

def parse_question_json(text: str) -> List[Dict]:
    """
    Parse JSON with better error handling and debugging
    """
    if not text:
        return []
    
    cleaned = clean_response_text(text)
    cleaned = repair_json(cleaned)

    # Try standard JSON parsing first
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
        elif isinstance(result, dict):
            return [result]
        else:
            return []
    except json.JSONDecodeError as e:
        # Try json5 as fallback (more lenient parsing)
        try:
            import json5
            result = json5.loads(cleaned)
            if isinstance(result, list):
                return result
            elif isinstance(result, dict):
                return [result]
            else:
                return []
        except Exception:
            # Final fallback: try to extract individual questions manually
            try:
                questions = []
                # Look for question objects and try to parse them individually
                question_pattern = r'\{\s*"question":[^}]*?"reasoning":[^}]*?\}'
                potential_questions = re.findall(question_pattern, cleaned, re.DOTALL)
                
                for q_text in potential_questions:
                    try:
                        q_obj = json.loads(q_text)
                        questions.append(q_obj)
                    except:
                        continue
                
                return questions
            except:
                st.error(f"JSON parsing failed completely. Error: {str(e)}")
                st.text(f"Cleaned text (first 500 chars): {cleaned[:500]}")
                return []

def filter_invalid_difficulty_alignment(questions: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Filter questions based on Bloom's taxonomy difficulty alignment"""
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

        # Validate required fields
        required_fields = ["question", "options", "correct_answer", "explanation", 
                          "cognitive_level", "estimated_correct_pct", "reasoning"]
        if not all(field in q for field in required_fields):
            invalid.append(q)
            continue

        cog = str(q.get("cognitive_level", "")).strip().capitalize()
        try:
            pct = int(q.get("estimated_correct_pct", -1))
        except (ValueError, TypeError):
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

def assign_difficulty_label(estimated_pct: int) -> Optional[int]:
    """Assign difficulty label based on estimated correctness percentage"""
    try:
        pct = int(estimated_pct)
    except (ValueError, TypeError):
        return None
    
    if pct < 30: return 8
    elif pct < 40: return 7
    elif pct < 50: return 6
    elif pct < 65: return 5
    elif pct < 75: return 4
    elif pct < 85: return 3
    elif pct < 90: return 2
    else: return 1

def group_by_difficulty(questions: List[Dict]) -> Dict[int, List[Dict]]:
    """Group questions by difficulty level"""
    groups = {i: [] for i in range(1, 9)}
    for q in questions:
        pct = q.get("estimated_correct_pct", 0)
        label = assign_difficulty_label(pct)
        if label:
            q["difficulty_label"] = label
            groups[label].append(q)
    return groups

def pick_question(diff: int, asked: set, all_qs: Dict[int, List[Dict]]) -> List[Tuple[int, Dict]]:
    """Pick available questions from a difficulty level"""
    pool = all_qs.get(diff, [])
    return [(i, q) for i, q in enumerate(pool) if (diff, i) not in asked]

def find_next_difficulty(current_diff: int, going_up: bool, asked: set, all_qs: Dict[int, List[Dict]]) -> int:
    """Find next available difficulty level"""
    next_diff = current_diff + 1 if going_up else current_diff - 1

    # Try one step in intended direction
    if 1 <= next_diff <= 8 and pick_question(next_diff, asked, all_qs):
        return next_diff
    
    # Search further in the intended direction
    search_range = (
        range(next_diff + 1, 9) if going_up else range(next_diff - 1, 0, -1)
    )
    for d in search_range:
        if pick_question(d, asked, all_qs):
            return d
    return current_diff

def get_next_question(current_diff: int, asked: set, all_qs: Dict[int, List[Dict]]) -> Tuple[int, Optional[int], Optional[Dict]]:
    """Get the next question to ask"""
    available = pick_question(current_diff, asked, all_qs)
    if not available:
        return current_diff, None, None
    idx, q = random.choice(available)
    return current_diff, idx, q

def accuracy_on_levels(answers: List[Tuple[int, bool]], levels: List[int]) -> float:
    """Calculate accuracy on specific difficulty levels"""
    filtered = [c for d, c in answers if d in levels]
    return sum(filtered) / len(filtered) if filtered else 0

def compute_mastery_score(answers: List[Tuple[int, bool]]) -> int:
    """Compute mastery score based on performance across difficulty levels"""
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
        
        # Normalize for guessing (25% baseline for multiple choice)
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

def render_mastery_bar(score: int):
    """Render the mastery progress bar"""
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

# Main Streamlit App
st.title("AscendQuiz")

# Render mastery bar if quiz is active
if "quiz_state" in st.session_state:
    score = compute_mastery_score(st.session_state.quiz_state.get("answers", []))
    render_mastery_bar(score)

# Initialize session state
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

**Built using the Claude-4 Sonnet model**, this app is a proof-of-concept showing what modern AI can do for personalized education. It blends mastery learning, real-time feedback, and adaptive testing into one clean experience. 

**⚠️ Note:** Question generation takes 4-5 minutes. Please be patient. The app only processes text content and cannot read handwriting or drawings.

---
""")

    uploaded_pdf = st.file_uploader("Upload class notes (PDF)", type="pdf")
    
    if uploaded_pdf:
        with st.spinner("Extracting text from PDF..."):
            chunks = extract_text_from_pdf(uploaded_pdf)
        
        if not chunks:
            st.error("No text could be extracted from the PDF. Please ensure it contains readable text.")
            st.stop()
        
        st.success(f"✅ Extracted text from {len(chunks)} pages")
        
        with st.spinner("Generating questions... This may take 4-5 minutes."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Adaptive chunking
            if len(chunks) <= 2:
                grouped_chunks = ["\n\n".join(chunks)]  
            else:
                grouped_chunks = ["\n\n".join(chunks[i:i+3]) for i in range(0, len(chunks), 3)]

            all_questions = []
            total_chunks = min(4, len(grouped_chunks))  # Limit to 4 chunks max
            
            for ci, chunk in enumerate(grouped_chunks[:total_chunks]):
                status_text.text(f"Processing chunk {ci+1}/{total_chunks}...")
                progress_bar.progress((ci + 0.5) / total_chunks)
                
                prompt = generate_prompt(chunk)
                response_text, error = call_claude_api(prompt)
                
                if error:
                    st.error(f"❌ Failed to generate questions for chunk {ci+1}: {error}")
                    continue
                    
                parsed = parse_question_json(response_text)
                valid, invalid = filter_invalid_difficulty_alignment(parsed)
                
                if valid:
                    all_questions.extend(valid)
                    st.success(f"✅ Generated {len(valid)} valid questions from chunk {ci+1}")
                else:
                    st.warning(f"⚠️ No valid questions generated from chunk {ci+1}")
                
                if invalid:
                    st.warning(f"⚠️ {len(invalid)} questions from chunk {ci+1} were filtered out due to invalid difficulty alignment")
                
                progress_bar.progress((ci + 1) / total_chunks)
            
            progress_bar.empty()
            status_text.empty()
        
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
            
            # Show question distribution
            diff_counts = {d: len(qs) for d, qs in st.session_state.questions_by_difficulty.items() if qs}
            st.success(f"🎉 Generated {len(all_questions)} total questions!")
            st.info(f"Distribution by difficulty: {diff_counts}")
            
            st.session_state.quiz_ready = True
            st.rerun()
        else:
            st.error("❌ No valid questions were generated. Please try with a different PDF or check your content.")

elif "quiz_ready" in st.session_state and st.session_state.quiz_ready:
    all_qs = st.session_state.questions_by_difficulty
    state = st.session_state.get("quiz_state", None)

    if state is None:
        st.warning("Quiz state not found. Please restart the app or re-upload a PDF.")
        st.stop()

    score = compute_mastery_score(state.get("answers", []))
    
    if not state["quiz_end"]:
        # Get next question if we don't have one
        if state["current_q"] is None and not state.get("show_explanation", False):
            diff, idx, q = get_next_question(state["current_difficulty"], state["asked"], all_qs)
            if q is None:
                state["quiz_end"] = True
                st.rerun()
            else:
                state["current_q"] = q
                state["current_q_idx"] = idx
                state["current_difficulty"] = diff

    if not state["quiz_end"] and state["current_q"]:
        q = state["current_q"]
        idx = state["current_q_idx"]

        st.markdown(f"### Question (Difficulty Level {state['current_difficulty']})")
        st.markdown(q["question"], unsafe_allow_html=True)
        
        def strip_leading_label(text):
            return re.sub(r"^[A-Da-d][\).:\-]?\s+", "", text).strip()

        option_labels = ["A", "B", "C", "D"]
        cleaned_options = [strip_leading_label(str(opt)) for opt in q["options"]]
        rendered_options = []
        
        for label, text in zip(option_labels, cleaned_options):
            if "$" in text or "\\" in text:
                rendered_text = f"{label}. {text}"
            else:
                rendered_text = f"{label}. {text}"
            rendered_options.append(rendered_text)

        selected = st.radio("Select your answer:", options=rendered_options, key=f"radio_{idx}", index=None)

        if st.button("Submit Answer", key=f"submit_{idx}") and not state.get("show_explanation", False):
            if selected is None:
                st.warning("Please select an answer before submitting.")
                st.stop()
                
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

            # Update mastery score
            score = compute_mastery_score(state["answers"])
            if score >= 70:
                state["quiz_end"] = True

            st.rerun()

        if state.get("show_explanation", False):
            if state["last_correct"]:
                st.success("✅ Correct!")
            else:
                st.error("❌ Incorrect.")
            
            st.markdown("**Explanation:**")
            st.markdown(state["last_explanation"], unsafe_allow_html=True)

            if st.button("Next Question"):
                # Adjust difficulty based on performance
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
        st.markdown("## Quiz Completed 🎉")
        
        if score >= 70:
            st.success(f"🎉 **Mastery Achieved!** Your mastery score is {score}%. Excellent work!")
        else:
            st.warning(f"Mastery not yet achieved. Your mastery score is {score}%. Review the material and try again.")

        # Show performance summary
        total_questions = len(state["answers"])
        correct_answers = sum(1 for _, correct in state["answers"] if correct)
        overall_accuracy = (correct_answers / total_questions * 100) if total_questions > 0 else 0
        
        st.markdown(f"""
        ### Performance Summary
        - **Questions Answered:** {total_questions}
        - **Overall Accuracy:** {overall_accuracy:.1f}%
        - **Mastery Score:** {score}%
        """)

        # Download questions
        if "all_questions" in st.session_state:
            df = pd.DataFrame(st.session_state.all_questions)
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download All Quiz Questions (CSV)",
                data=csv_data,
                file_name="ascendquiz_questions.csv",
                mime="text/csv"
            )

        # Restart button
        if st.button("🔄 Start New Quiz"):
            # Clear session state to restart
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
