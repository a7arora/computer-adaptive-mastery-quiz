🌟 StarQuiz: AI-Powered Computer-Adaptive Mastery-Based Quizzing from Any PDF

StarQuiz is a fully functional Streamlit web app that automatically generates personalized, mastery-driven quizzes from any uploaded PDF (lecture notes, textbook excerpts, etc). Built with Groq’s meta-llama/llama-4-scout-17b-16e-instruct, the app uses generative AI to create high-quality, curriculum-agnostic assessments with adaptive difficulty and real-time feedback.

🚀 Why It Matters: The Pedagogy Behind StarQuiz

StarQuiz is more than just an AI demo — it’s grounded in proven learning science and addresses three fundamental gaps in current edtech tools:

🧠 1. Mastery-Based Learning (MBL)

As popularized by Sal Khan, mastery-based learning ensures students fully grasp a concept before moving on, unlike traditional models that allow learning gaps to accumulate.

Research shows that in MBL classrooms, the average student performs as well as the top 15% of traditional classrooms (Kulik et al., 1990).

StarQuiz operationalizes this by ending quizzes only when a student demonstrates mastery — defined as correctly answering 5+ questions at high difficulty (Level 6+) with at least 75% accuracy.

📈 2. Computer-Adaptive Testing (CAT)

CAT improves assessment accuracy by dynamically adjusting question difficulty based on a student's responses.

A 2017 study in Journal of Educational Psychology showed that CAT leads to higher achievement, increased engagement, and greater test precision (Martin & Lazendic, 2017).

StarQuiz uses predicted student accuracy (e.g. “72% of students likely get this right”) to adjust quiz difficulty in real time, starting at ~70% and adapting up or down after each response.

🧪 3. The Power of Practice Testing

Decades of cognitive science highlight practice testing (retrieval practice) as one of the most effective learning strategies.

When students test themselves — especially with feedback — they retain more, close knowledge gaps, and better transfer knowledge to new contexts.

StarQuiz builds in immediate feedback with explanations after every question, reinforcing learning via active recall and self-explanation.

🧠 How It Works

Upload ContentEducators upload a PDF (e.g., lecture notes, textbook chapters).

Question Generation with Difficulty Estimation

The LLM creates 15 multiple-choice questions with:

4 answer options

Correct answer + explanation

Predicted correctness percentage (e.g. 68% of students expected to get this right)

Adaptive Quiz Loop

Starts with a medium-difficulty question (~70% correctness)

Correct answer → harder question

Incorrect answer → easier question

Quiz stops when mastery is reached

Mastery Determination

Mastery is achieved when a student answers ≥5 high-difficulty questions correctly (with ≥75% accuracy).

Results are presented alongside personalized feedback.

Difficulty Tiering & Error Handling

Questions are grouped into 8 tiers (Very Easy → Very Hard)

Fallback logic ensures smooth user experience even when certain difficulty levels are unavailable.

🛠️ Technical Overview

Frontend: Streamlit (clean UI, session-based state management)

Backend:

call_groq_api: Structured prompt querying for JSON-formatted question objects

Adaptive quiz engine dynamically selects next question

Smart fallback engine searches nearby difficulty bands if needed

🧹 Why StarQuiz Is Unique

No Rigid Curricula: Unlike ALEKS or IXL, StarQuiz works with any input content.

Truly Adaptive: Combines real-time content generation + adaptive delivery — not just static question banks.

Immediate Feedback: Every answer includes rationale and remediation support.

Pedagogical Alignment: Synthesizes best practices in retrieval-based learning, mastery models, and adaptive testing — supported by decades of research.

🧠 Future Directions

Support for non-MCQ formats (short answer, open-ended)

Integration with video uploads and transcript parsing

Data collection pipelines to improve difficulty estimation models

Teacher dashboards for group-level insights and intervention planning

📚 References

Kulik, C., Kulik, J., & Bangert-Drowns, R. (1990). Effectiveness of Mastery Learning Programs: A Meta-Analysis. Review of Educational Research.

Khan, S. (2016). Let’s Teach for Mastery, Not Test Scores. TED Talk

Martin, A.J., & Lazendic, G. (2017). Computer-adaptive testing: Implications for students’ achievement, motivation, and engagement. Journal of Educational Psychology.

Roediger, H. L., & Karpicke, J. D. (2006). Test-enhanced learning: Taking memory tests improves long-term retention. Psychological Science.

