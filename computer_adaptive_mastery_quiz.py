if st.button("Submit Answer", key=f"submit_{idx}") and not state.get("show_explanation", False):
    selected_letter = selected.split(".")[0].strip().upper()
    try:
        correct_letter = q["correct_answer"].strip().upper()
    except Exception:
        st.error("⚠️ Question error: Correct answer not found.")
        state["quiz_end"] = True
        st.stop()

    correct = (selected_letter == correct_letter)

    # Record answer
    state["asked"].add((state["current_difficulty"], idx))
    state["answers"].append((state["current_difficulty"], correct))

    # Show explanation and feedback
    state["show_explanation"] = True
    state["last_correct"] = correct
    state["last_explanation"] = q.get("explanation", "")
    state["last_option_feedback"] = q.get("option_feedback", {})
    state["current_q"] = None
    state["current_q_idx"] = None

    st.experimental_rerun()
