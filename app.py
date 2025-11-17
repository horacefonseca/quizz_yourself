"""
Interactive Quiz Application
Built with Streamlit - Works with any YAML question bank
"""

import streamlit as st
import os
from pathlib import Path
from utils import (
    load_questions,
    sample_questions,
    grade_quiz,
    generate_pdf,
    calculate_quiz_length
)

# Page configuration
st.set_page_config(
    page_title="Interactive Quiz App",
    page_icon="📝",
    layout="wide"
)

# Initialize session state
if 'quiz_started' not in st.session_state:
    st.session_state.quiz_started = False
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'quiz_questions' not in st.session_state:
    st.session_state.quiz_questions = []
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = {}
if 'quiz_submitted' not in st.session_state:
    st.session_state.quiz_submitted = False
if 'grading_results' not in st.session_state:
    st.session_state.grading_results = None
if 'random_seed' not in st.session_state:
    st.session_state.random_seed = None
if 'timer_enabled' not in st.session_state:
    st.session_state.timer_enabled = False
if 'timer_per_question' not in st.session_state:
    st.session_state.timer_per_question = 30
if 'include_open_questions' not in st.session_state:
    st.session_state.include_open_questions = True
if 'quiz_start_time' not in st.session_state:
    st.session_state.quiz_start_time = None
if 'open_question_grades' not in st.session_state:
    st.session_state.open_question_grades = {}


def reset_quiz():
    """Reset quiz to initial state"""
    st.session_state.quiz_started = False
    st.session_state.quiz_questions = []
    st.session_state.user_answers = {}
    st.session_state.quiz_submitted = False
    st.session_state.grading_results = None
    st.session_state.quiz_start_time = None
    st.session_state.open_question_grades = {}


def main():
    """Main application"""

    # Title
    st.title("📝 Interactive Quiz Application")
    st.markdown("---")

    # Sidebar for quiz setup
    with st.sidebar:
        st.header("Quiz Setup")

        # Reset button - always visible
        if st.button("🔄 Reset & Start Over", use_container_width=True, type="secondary"):
            reset_quiz()
            st.rerun()

        st.markdown("---")

        # Step 1: Question Bank Selection
        st.subheader("1. Select Question Bank")

        source_option = st.radio(
            "Choose question source:",
            ["Upload YAML/Markdown file", "Paste ChatGPT-formatted text", "Use bundled quiz"]
        )

    # Main area content based on selection
    questions_loaded = False

    if source_option == "Paste ChatGPT-formatted text":
        # CHATGPT WORKFLOW IN CENTER - New 2-step process
        st.header("📝 Create Quiz from Raw Text with ChatGPT AI")

        # Visual workflow diagram
        st.info("""
**📋 Workflow Steps:**

**Step 1** → Paste your raw text below
**Step 2** → Click "Generate Prompt" to combine instructions + your text
**Step 3** → Copy the combined prompt
**Step 4** → Open ChatGPT and paste it there
**Step 5** → Copy ChatGPT's formatted output
**Step 6** → Paste ChatGPT's output in the final box below
        """)

        st.markdown("---")

        # STEP 1: User pastes raw text
        st.subheader("Step 1: Paste Your Raw Text")
        user_raw_text = st.text_area(
            "Enter your raw quiz content here:",
            height=200,
            placeholder="Example:\n\nWhat is the capital of France?\nParis is the capital\n\nWho wrote Romeo and Juliet?\nShakespeare wrote it\n...",
            key="user_raw_text_input"
        )

        # Show preview of pasted text (last 4 lines)
        if user_raw_text and user_raw_text.strip():
            lines = user_raw_text.strip().split('\n')
            if len(lines) > 4:
                preview_lines = lines[-4:]
                st.caption(f"📝 {len(lines)} lines pasted. Showing last 4 lines:")
                st.code('\n'.join(preview_lines), language=None)
            else:
                st.caption(f"📝 {len(lines)} lines pasted")

        # STEP 2: Generate combined prompt
        if user_raw_text and user_raw_text.strip():
            if st.button("📝 Generate Combined Prompt", type="primary", use_container_width=True):
                # Store in session state
                st.session_state.combined_prompt = f"""Please convert the following raw text into a structured quiz format and coding text format ready to cut and paste with indentations. Follow these rules EXACTLY:

FORMAT RULES:
Each question must start with "QUESTION" followed by a number (QUESTION 1, QUESTION 2, etc.)
Each question must have these fields (one per line):
Type: (either "mc" for multiple choice or "open" for open-ended)
Question: (the question text)

FOR MULTIPLE CHOICE (mc):
A) (first option)
B) (second option)
C) (third option)
D) (fourth option)
Correct: (the correct letter: A, B, C, or D)

FOR OPEN-ENDED (open):
Answer: (the correct answer text)

OPTIONAL FIELDS (for both types):
Explanation: (5-15 words explaining why the answer is correct)
Chapter: (chapter or topic reference)

Leave a blank line between questions
Use EXACTLY this format - no extra formatting, no bold, no italics

EXAMPLE OUTPUT:

QUESTION 1
Type: mc
Question: What is the range of valid probability values?
A) Between -1 and 1
B) Between 0 and 1
C) Between 0 and 100
D) Any positive number
Correct: B
Explanation: Probabilities must be between 0 (impossible) and 1 (certain).
Chapter: 1.1

QUESTION 2
Type: open
Question: Define binary classification.
Answer: Classification with exactly two possible class labels
Explanation: Binary classification has exactly two possible outcomes.
Chapter: 3

====================================================
RAW TEXT TO CONVERT:
====================================================

{user_raw_text}"""
                st.rerun()

        # STEP 3-4: Show combined prompt with copy and ChatGPT buttons
        if 'combined_prompt' in st.session_state and st.session_state.combined_prompt:
            st.markdown("---")
            st.subheader("Step 2: Copy & Paste to ChatGPT")

            st.success("✅ Combined prompt generated! Now:")

            # Show the combined prompt
            with st.expander("📖 View Combined Prompt", expanded=True):
                st.code(st.session_state.combined_prompt, language=None)

            # Action buttons
            col1, col2 = st.columns([1, 1])

            with col1:
                # Copy button - using properly escaped content
                import html
                escaped_prompt = html.escape(st.session_state.combined_prompt)

                copy_button_html = f"""
                <button onclick="copyPrompt()" style="
                    background-color: #4CAF50;
                    color: white;
                    padding: 12px 24px;
                    font-size: 16px;
                    border: none;
                    border-radius: 8px;
                    cursor: pointer;
                    width: 100%;
                    font-weight: bold;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                    transition: all 0.3s;
                " onmouseover="this.style.backgroundColor='#45a049'"
                   onmouseout="this.style.backgroundColor='#4CAF50'">
                    📋 Copy Combined Prompt
                </button>
                <textarea id="combined-prompt" style="position: absolute; left: -9999px;">{escaped_prompt}</textarea>
                <p id="copy-prompt-status" style="color: green; font-weight: bold; margin-top: 8px; min-height: 24px;"></p>
                <script>
                function copyPrompt() {{
                    var copyText = document.getElementById("combined-prompt");
                    copyText.style.position = "static";
                    copyText.select();
                    copyText.setSelectionRange(0, 999999);

                    // Try modern clipboard API first
                    if (navigator.clipboard && navigator.clipboard.writeText) {{
                        navigator.clipboard.writeText(copyText.value).then(function() {{
                            document.getElementById("copy-prompt-status").innerHTML = "✅ Copied to clipboard!";
                            setTimeout(function() {{
                                document.getElementById("copy-prompt-status").innerHTML = "";
                            }}, 3000);
                        }});
                    }} else {{
                        // Fallback to execCommand
                        document.execCommand("copy");
                        document.getElementById("copy-prompt-status").innerHTML = "✅ Copied to clipboard!";
                        setTimeout(function() {{
                            document.getElementById("copy-prompt-status").innerHTML = "";
                        }}, 3000);
                    }}

                    copyText.style.position = "absolute";
                }}
                </script>
                """
                st.markdown(copy_button_html, unsafe_allow_html=True)

            with col2:
                # ChatGPT link button
                chatgpt_button_html = """
                <a href="https://chatgpt.com" target="_blank" style="text-decoration: none;">
                    <button style="
                        background: linear-gradient(135deg, #10a37f 0%, #1a7f64 100%);
                        color: white;
                        padding: 12px 24px;
                        font-size: 16px;
                        border: none;
                        border-radius: 8px;
                        cursor: pointer;
                        width: 100%;
                        font-weight: bold;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                        transition: all 0.3s;
                    " onmouseover="this.style.transform='scale(1.05)'"
                       onmouseout="this.style.transform='scale(1)'">
                        🤖 Open ChatGPT
                    </button>
                </a>
                """
                st.markdown(chatgpt_button_html, unsafe_allow_html=True)

            st.markdown("---")

            # STEP 5-6: Paste ChatGPT output
            st.subheader("Step 3: Paste ChatGPT's Output")
            chatgpt_output = st.text_area(
                "After ChatGPT formats your questions, paste the output here:",
                height=300,
                max_chars=120000,
                placeholder="Paste ChatGPT's formatted output here...\n\nExample:\nQUESTION 1\nType: mc\nQuestion: What is...?\nA) Option A\nB) Option B\n...",
                key="chatgpt_output_input"
            )

            if chatgpt_output and chatgpt_output.strip():
                # Show preview (last 4 lines)
                lines = chatgpt_output.strip().split('\n')
                word_count = len(chatgpt_output.split())

                st.caption(f"📝 {len(lines)} lines pasted ({word_count:,} words)")

                if len(lines) > 4:
                    preview_lines = lines[-4:]
                    with st.expander("👁️ Preview (last 4 lines)", expanded=False):
                        st.code('\n'.join(preview_lines), language=None)

                # Parse button
                if st.button("🔍 Parse Questions", type="primary", use_container_width=True):
                    questions, error = load_questions(chatgpt_output, 'gemini')

                    if error:
                        st.error(f"❌ Error parsing format: {error}")
                        st.info("💡 Make sure you copied the EXACT format from ChatGPT.")
                    else:
                        st.session_state.questions = questions
                        questions_loaded = True
                        st.success(f"✅ Successfully parsed {len(questions)} questions!")
                        st.balloons()

    # Other options stay in sidebar
    with st.sidebar:
        if source_option == "Upload YAML/Markdown file":
            uploaded_file = st.file_uploader(
                "Upload your question bank",
                type=['yaml', 'yml', 'md'],
                help="Upload a YAML or Markdown file containing quiz questions"
            )

            if uploaded_file is not None:
                file_content = uploaded_file.read().decode('utf-8')
                file_extension = uploaded_file.name.split('.')[-1].lower()
                questions, error = load_questions(file_content, file_extension)

                if error:
                    st.error(f"Error loading questions: {error}")
                else:
                    st.session_state.questions = questions
                    questions_loaded = True
                    st.success(f"✅ Loaded {len(questions)} questions!")

        elif source_option == "Use bundled quiz":
            # Look for sample quizzes in sample_quizzes folder
            sample_dir = Path(__file__).parent / "sample_quizzes"

            if sample_dir.exists():
                sample_files = list(sample_dir.glob("*.yaml")) + list(sample_dir.glob("*.yml"))

                if sample_files:
                    selected_file = st.selectbox(
                        "Select a quiz:",
                        sample_files,
                        format_func=lambda x: x.stem
                    )

                    if selected_file:
                        with open(selected_file, 'r', encoding='utf-8') as f:
                            file_content = f.read()

                        file_extension = selected_file.suffix.lstrip('.').lower()
                        questions, error = load_questions(file_content, file_extension)

                        if error:
                            st.error(f"Error loading questions: {error}")
                        else:
                            st.session_state.questions = questions
                            questions_loaded = True
                            st.success(f"✅ Loaded {len(questions)} questions!")
                else:
                    st.warning("No sample quizzes found in sample_quizzes folder")
            else:
                st.warning("sample_quizzes folder not found")

        # Step 2: Quiz Length Selection
        if st.session_state.questions:
            st.markdown("---")
            st.subheader("2. Quiz Length")

            total_questions = len(st.session_state.questions)
            st.info(f"Total available questions: {total_questions}")

            # Percentage slider
            percentage = st.slider(
                "Select percentage of questions:",
                min_value=10,
                max_value=100,
                value=50,
                step=10,
                key="percentage_slider",
                help="Choose what percentage of the question bank to include"
            )

            suggested_num = int(total_questions * percentage / 100)
            st.write(f"Suggested: {suggested_num} questions")

            # Custom override
            use_custom = st.checkbox("Use custom number of questions", key="use_custom_checkbox")

            if use_custom:
                custom_num = st.number_input(
                    "Number of questions:",
                    min_value=1,
                    max_value=total_questions,
                    value=min(suggested_num, total_questions),
                    key="custom_num_input"
                )
                final_num = custom_num
            else:
                final_num = suggested_num

            st.markdown("---")

            # Advanced Settings (for larger quizzes)
            with st.expander("⚙️ Advanced Quiz Settings"):
                st.markdown("**Random Seed (for reproducibility)**")
                use_seed = st.checkbox(
                    "Use random seed",
                    value=st.session_state.random_seed is not None,
                    key="use_seed_checkbox"
                )
                if use_seed:
                    seed_value = st.number_input(
                        "Seed value:",
                        min_value=0,
                        max_value=9999,
                        value=42 if st.session_state.random_seed is None else st.session_state.random_seed,
                        key="seed_value_input",
                        help="Same seed = same quiz version every time"
                    )
                    st.session_state.random_seed = seed_value
                else:
                    st.session_state.random_seed = None

                st.markdown("---")

                st.markdown("**Timer Settings**")
                timer_enabled = st.checkbox(
                    "Enable quiz timer",
                    value=st.session_state.timer_enabled,
                    key="timer_enabled_checkbox"
                )
                st.session_state.timer_enabled = timer_enabled

                if timer_enabled:
                    timer_per_question = st.select_slider(
                        "Time per question:",
                        options=[30, 60, 90],
                        value=st.session_state.timer_per_question,
                        key="timer_per_question_slider",
                        format_func=lambda x: f"{x} seconds"
                    )
                    st.session_state.timer_per_question = timer_per_question

                    # Calculate total time
                    total_time_seconds = final_num * timer_per_question
                    total_minutes = total_time_seconds // 60
                    st.info(f"⏱️ Total quiz time: {total_minutes} minutes ({total_time_seconds} seconds)")

                st.markdown("---")

                st.markdown("**Open Questions**")
                include_open = st.checkbox(
                    "Include open questions",
                    value=st.session_state.include_open_questions,
                    key="include_open_checkbox",
                    help="Open questions must be evaluated by instructor"
                )
                st.session_state.include_open_questions = include_open

                if not include_open:
                    st.warning("⚠️ Open questions will be excluded from quiz")
                else:
                    # Count available open questions
                    open_count = sum(1 for q in st.session_state.questions if q.get('type') == 'open')
                    st.caption(f"📝 Available open questions: {open_count}")

            st.markdown("---")

            # Start Quiz Button
            if not st.session_state.quiz_started:
                if st.button("🚀 Start Quiz", use_container_width=True):
                    # Filter questions based on settings
                    available_questions = st.session_state.questions

                    # Filter out open questions if disabled
                    if not st.session_state.include_open_questions:
                        available_questions = [q for q in available_questions if q.get('type') != 'open']

                    # Sample questions with optional seed
                    import random
                    if st.session_state.random_seed is not None:
                        random.seed(st.session_state.random_seed)

                    st.session_state.quiz_questions = sample_questions(
                        available_questions,
                        min(final_num, len(available_questions))
                    )

                    # Start timer if enabled
                    if st.session_state.timer_enabled:
                        import time
                        st.session_state.quiz_start_time = time.time()

                    st.session_state.quiz_started = True
                    st.session_state.user_answers = {}
                    st.session_state.quiz_submitted = False
                    st.rerun()
            else:
                if st.button("🔄 Reset Quiz", use_container_width=True):
                    reset_quiz()
                    st.rerun()

    # Main content area
    # DEBUG INFO
    with st.expander("🔍 Debug Info", expanded=False):
        st.write(f"questions loaded: {len(st.session_state.questions)}")
        st.write(f"quiz_started: {st.session_state.quiz_started}")
        st.write(f"quiz_questions: {len(st.session_state.quiz_questions)}")
        st.write(f"quiz_submitted: {st.session_state.quiz_submitted}")

    if not st.session_state.quiz_started:
        # Welcome screen
        st.markdown("""
        ## Welcome to the Interactive Quiz Application! 👋

        ### How to use:
        1. **Select a question bank** from the sidebar (upload or use bundled)
        2. **Choose quiz length** using the percentage slider or custom number
        3. **Click "Start Quiz"** to begin
        4. **Answer all questions** and submit
        5. **Download your results** as a PDF

        ### Supported File Formats:
        📄 **YAML** (.yaml, .yml) - Structured format
        📝 **Markdown** (.md) - Human-readable format

        ### Supported Question Types:
        - **Multiple Choice (MC):** Select from provided options
        - **Open Questions:** Type your answer

        ### Features:
        ✅ Random question sampling
        ✅ Instant grading
        ✅ Detailed explanations
        ✅ PDF results export
        """)

    elif not st.session_state.quiz_submitted:
        # Quiz Interface
        st.header(f"Quiz: {len(st.session_state.quiz_questions)} Questions")

        # DEBUG
        st.info(f"DEBUG: About to display {len(st.session_state.quiz_questions)} questions")

        # Timer display (if enabled)
        if st.session_state.timer_enabled and st.session_state.quiz_start_time:
            import time
            elapsed_time = time.time() - st.session_state.quiz_start_time
            total_allowed = len(st.session_state.quiz_questions) * st.session_state.timer_per_question
            remaining_time = max(0, total_allowed - int(elapsed_time))

            minutes_left = remaining_time // 60
            seconds_left = remaining_time % 60

            if remaining_time > 60:
                timer_color = "🟢"
            elif remaining_time > 30:
                timer_color = "🟡"
            else:
                timer_color = "🔴"

            st.warning(f"{timer_color} **Time Remaining:** {minutes_left}:{seconds_left:02d} (Total: {total_allowed//60} min)")

        st.markdown("---")

        # Display all questions
        st.warning(f"DEBUG: Starting question loop with {len(st.session_state.quiz_questions)} questions")
        for i, q in enumerate(st.session_state.quiz_questions, 1):
            with st.container():
                st.subheader(f"Question {i} of {len(st.session_state.quiz_questions)}")

                # Question text
                st.markdown(f"**{q['question']}**")

                # Chapter reference if available
                if 'chapter' in q:
                    st.caption(f"📚 Chapter: {q['chapter']}")

                # Answer input based on question type
                if q['type'] == 'mc':
                    # Multiple choice
                    options = q['options']
                    user_answer = st.radio(
                        "Select your answer:",
                        options,
                        key=f"q_{q['id']}",
                        index=None
                    )

                    if user_answer:
                        # Extract letter (A, B, C, etc.)
                        answer_letter = user_answer.split(')')[0].strip()
                        st.session_state.user_answers[q['id']] = answer_letter

                elif q['type'] == 'open':
                    # Open question
                    user_answer = st.text_input(
                        "Your answer:",
                        key=f"q_{q['id']}",
                        placeholder="Type your answer here..."
                    )

                    if user_answer:
                        st.session_state.user_answers[q['id']] = user_answer

                st.markdown("---")

        # Submit button
        st.markdown("### Ready to submit?")

        answered = len(st.session_state.user_answers)
        total = len(st.session_state.quiz_questions)

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.info(f"Questions answered: {answered}/{total}")

        with col2:
            if answered < total:
                st.warning("⚠️ Some questions unanswered")

        with col3:
            if st.button("📤 Submit Quiz", type="primary", use_container_width=True):
                # Grade the quiz
                st.session_state.grading_results = grade_quiz(
                    st.session_state.quiz_questions,
                    st.session_state.user_answers
                )
                st.session_state.quiz_submitted = True
                st.rerun()

        # Auto-refresh timer after displaying all questions
        if st.session_state.timer_enabled and st.session_state.quiz_start_time:
            import time
            elapsed_time = time.time() - st.session_state.quiz_start_time
            total_allowed = len(st.session_state.quiz_questions) * st.session_state.timer_per_question
            remaining_time = max(0, total_allowed - int(elapsed_time))

            if remaining_time > 0:
                time.sleep(1)
                st.rerun()

    else:
        # Results Screen
        results = st.session_state.grading_results

        st.header("📊 Quiz Results")
        st.markdown("---")

        # Score summary
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Score",
                f"{results['score_percentage']:.1f}%",
                delta=None
            )

        with col2:
            st.metric(
                "Correct",
                results['correct'],
                delta=None
            )

        with col3:
            st.metric(
                "Incorrect",
                results['incorrect'],
                delta=None
            )

        st.markdown("---")

        # Performance indicator
        score = results['score_percentage']
        if score >= 90:
            st.success("🎉 Excellent! Outstanding performance!")
        elif score >= 75:
            st.success("✅ Great job! Very good performance!")
        elif score >= 60:
            st.info("👍 Good work! Room for improvement.")
        else:
            st.warning("📚 Keep studying! Review the explanations below.")

        st.markdown("---")

        # Detailed results
        st.subheader("Detailed Answer Sheet")

        for i, result in enumerate(results['results'], 1):
            with st.expander(
                f"Question {i}: {'✅ Correct' if result['is_correct'] else '❌ Incorrect'}",
                expanded=not result['is_correct']  # Auto-expand incorrect answers
            ):
                st.markdown(f"**Question:** {result['question']}")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**Your answer:**")
                    if result['is_correct']:
                        st.success(result['user_answer'])
                    else:
                        st.error(result['user_answer'])

                with col2:
                    st.markdown(f"**Correct answer:**")
                    st.info(result['correct_answer'])

                st.markdown(f"**💡 Explanation:** {result['explanation']}")

                if result['chapter'] != 'N/A':
                    st.caption(f"📚 Chapter: {result['chapter']}")

                # For open questions, add manual grading toggle
                if result.get('type') == 'open':
                    st.markdown("---")
                    st.warning("⚠️ **Instructor Grading Required:** Open question must be evaluated manually")

                    # Toggle for manual grading (default to False/wrong)
                    q_id = result['id']
                    current_grade = st.session_state.open_question_grades.get(q_id, False)

                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.markdown("**Mark this answer as:**")
                    with col_b:
                        new_grade = st.checkbox(
                            "Correct",
                            value=current_grade,
                            key=f"grade_open_{q_id}_{i}"
                        )

                    # Update grade if changed
                    if new_grade != current_grade:
                        st.session_state.open_question_grades[q_id] = new_grade
                        # Recalculate score
                        total_open = sum(1 for r in results['results'] if r.get('type') == 'open')
                        correct_open = sum(1 for q_id, is_correct in st.session_state.open_question_grades.items() if is_correct)
                        mc_correct = sum(1 for r in results['results'] if r.get('type') == 'mc' and r['is_correct'])

                        total_questions = len(results['results'])
                        total_correct = mc_correct + correct_open
                        results['correct'] = total_correct
                        results['incorrect'] = total_questions - total_correct
                        results['score_percentage'] = (total_correct / total_questions * 100) if total_questions > 0 else 0
                        st.session_state.grading_results = results
                        st.rerun()

                    if new_grade:
                        st.success("✅ Marked as CORRECT")
                    else:
                        st.error("❌ Marked as WRONG (default)")

        st.markdown("---")

        # PDF Download
        st.subheader("📄 Download Results")

        col1, col2 = st.columns([3, 1])

        with col1:
            st.write("Download a PDF report of your quiz results including all questions, answers, and explanations.")

        with col2:
            # Generate PDF
            pdf_bytes = generate_pdf(results, "Quiz Results")

            st.download_button(
                label="📥 Download PDF",
                data=pdf_bytes,
                file_name=f"quiz_results_{st.session_state.grading_results['score_percentage']:.0f}pct.pdf",
                mime="application/pdf",
                use_container_width=True
            )

        # Retake button
        st.markdown("---")
        if st.button("🔄 Take Another Quiz", use_container_width=False):
            reset_quiz()
            st.rerun()


if __name__ == "__main__":
    main()
