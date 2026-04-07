# AscendQuiz: An AI-Powered Adaptive Mastery Quiz 🚀

Ascend Quiz is a fully functional Streamlit web app that automatically generates personalized, mastery-driven quizzes from any uploaded PDF (lecture notes, textbook excerpts, etc). Built with Deepseek's R1-0528 API, the app uses generative AI to create high-quality, curriculum-agnostic assessments with adaptive difficulty and real-time feedback.

Main file: [ascendquiz_db.py](https://github.com/tomragus/computer-adaptive-mastery-quiz/blob/main/ascendquiz_db.py)

### To run the quiz from your device:
Download the streamlit library and then set up your "secrets.toml" file (in your directory create a hidden ".stramlit" folder and put into it a "secrets.toml" file containing your Gemini API key: type GEMINI_API_KEY = "key". To get your Gemini API key along with $300 worth of free API credits, create an account on Google AI Studio and go to [https://aistudio.google.com/api-keys](https://aistudio.google.com/api-keys) to get your key).

Once your environment is set up, run using 'streamlit run ascendquiz_db.py'

#### What's new:
- User authentication — Create account, log back in to see progress
- Topic tagging — Each question is tagged with a topic (e.g., "Cell Biology", "Algorithms"), enabling per-topic performance tracking
- Dashboard — View quizzes taken, average score, and potential weaker topics
- Quiz history — Review all past quiz attempts with scores and mastery status

### This project is a collaboration between Ashley Zhou, Tom Ragus, Ethan Baquarian, and Timothy Chen. It is a work in progress and we are actively working to improve it.

Some key challenges (as of April 2026):
- Having a determined set of quiz questions generated in the question pool at the start of the quiz (current problem: not enough questions are generated, making the quiz sometimes end early due to lack of questions, even if the user answers them correctly)
- Make the infrastructure more robust and stable, and more visually appealing - in its current state, it looks like a bit like "AI slop"
- Simplify by removing some of the extra features (dashboard, history), focus on getting the key quiz feature solid before adding new stuff
