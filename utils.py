"""
Utility functions for the Quiz Application
Handles YAML/Markdown loading, question sampling, grading, and PDF generation
"""

import yaml
import random
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple
from fpdf import FPDF


def load_yaml_questions(file_content: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Load questions from YAML content

    Args:
        file_content: String content of YAML file

    Returns:
        Tuple of (list of questions, error message if any)
    """
    try:
        questions = yaml.safe_load(file_content)

        if not questions:
            return [], "YAML file is empty"

        if not isinstance(questions, list):
            return [], "YAML must contain a list of questions"

        # Validate question structure
        required_fields = ['id', 'question', 'type']
        for i, q in enumerate(questions):
            if not isinstance(q, dict):
                return [], f"Question {i+1} is not properly formatted"

            for field in required_fields:
                if field not in q:
                    return [], f"Question {i+1} missing required field: {field}"

            # Validate question type
            if q['type'] not in ['mc', 'open']:
                return [], f"Question {i+1} has invalid type: {q['type']} (must be 'mc' or 'open')"

            # Validate MC questions
            if q['type'] == 'mc':
                if 'options' not in q:
                    return [], f"MC Question {i+1} missing 'options'"
                if 'correct' not in q:
                    return [], f"MC Question {i+1} missing 'correct' answer"

            # Validate open questions
            if q['type'] == 'open':
                if 'answer' not in q:
                    return [], f"Open Question {i+1} missing 'answer'"

        return questions, ""

    except yaml.YAMLError as e:
        return [], f"YAML parsing error: {str(e)}"
    except Exception as e:
        return [], f"Error loading questions: {str(e)}"


def load_markdown_questions(file_content: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Load questions from Markdown content

    Markdown format:
    # Question 1

    **Type:** mc

    What is the sigmoid output range?

    **Options:**
    - A) 0–1
    - B) -1–1
    - C) 0–10

    **Correct:** A

    **Explanation:** Sigmoid always outputs 0–1.

    **Chapter:** 1.4

    ---

    Args:
        file_content: String content of Markdown file

    Returns:
        Tuple of (list of questions, error message if any)
    """
    try:
        questions = []

        # Split by question separators (---) or markdown headers
        # First, split by --- or by # Question pattern
        question_blocks = re.split(r'\n---+\n|\n(?=# Question \d+)', file_content.strip())

        question_id = 1

        for block in question_blocks:
            block = block.strip()
            if not block or block.startswith('#') and 'Question' not in block:
                continue

            # Extract question text (first line after header or first paragraph)
            lines = block.split('\n')

            # Initialize question dict
            question = {
                'id': question_id,
                'question': '',
                'type': 'mc',  # default
                'options': [],
                'correct': '',
                'answer': '',
                'explanation': '',
                'chapter': ''
            }

            # Parse the block
            i = 0
            while i < len(lines):
                line = lines[i].strip()

                # Skip empty lines and headers
                if not line or line.startswith('#'):
                    i += 1
                    continue

                # Type field
                if line.startswith('**Type:**'):
                    question['type'] = line.replace('**Type:**', '').strip().lower()
                    i += 1
                    continue

                # Options section
                if line.startswith('**Options:**'):
                    i += 1
                    while i < len(lines) and lines[i].strip().startswith('-'):
                        option = lines[i].strip()[2:].strip()  # Remove "- " prefix
                        question['options'].append(option)
                        i += 1
                    continue

                # Correct answer
                if line.startswith('**Correct:**'):
                    question['correct'] = line.replace('**Correct:**', '').strip()
                    i += 1
                    continue

                # Answer (for open questions)
                if line.startswith('**Answer:**'):
                    question['answer'] = line.replace('**Answer:**', '').strip()
                    i += 1
                    continue

                # Explanation
                if line.startswith('**Explanation:**'):
                    question['explanation'] = line.replace('**Explanation:**', '').strip()
                    i += 1
                    continue

                # Chapter
                if line.startswith('**Chapter:**'):
                    question['chapter'] = line.replace('**Chapter:**', '').strip()
                    i += 1
                    continue

                # Question text (if not yet set)
                if not question['question'] and line and not line.startswith('**'):
                    question['question'] = line
                    i += 1
                    continue

                i += 1

            # Validate and add question
            if question['question']:
                # Validate required fields
                if question['type'] == 'mc':
                    if not question['options'] or not question['correct']:
                        continue  # Skip incomplete MC questions
                elif question['type'] == 'open':
                    if not question['answer']:
                        continue  # Skip incomplete open questions

                questions.append(question)
                question_id += 1

        if not questions:
            return [], "No valid questions found in markdown file"

        return questions, ""

    except Exception as e:
        return [], f"Error parsing markdown: {str(e)}"


def load_gemini_questions_singleline(file_content: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Fallback parser for single-line Gemini format
    Handles: QUESTION 1 Type: mc Question: ... A) ... B) ... Correct: A Explanation: ... Chapter: ...
    """
    try:
        questions = []

        # Split by QUESTION pattern (with or without newline)
        question_blocks = re.split(r'QUESTION\s+\d+\s*', file_content)

        # Remove empty first element
        if question_blocks and not question_blocks[0].strip():
            question_blocks.pop(0)

        for idx, block in enumerate(question_blocks, 1):
            if not block.strip():
                continue

            # Single-line parsing with regex
            question = {
                'id': idx,
                'type': '',
                'question': '',
                'options': [],
                'correct': '',
                'answer': '',
                'explanation': '',
                'chapter': ''
            }

            # Extract Type
            type_match = re.search(r'Type:\s*(mc|open)', block, re.IGNORECASE)
            if type_match:
                question['type'] = type_match.group(1).lower()

            # Extract Question
            question_match = re.search(r'Question:\s*(.+?)(?=\s+[A-D]\)|Answer:|Correct:|Explanation:|Chapter:|$)', block)
            if question_match:
                question['question'] = question_match.group(1).strip()

            if question['type'] == 'mc':
                # Extract options A, B, C, D
                for letter in ['A', 'B', 'C', 'D']:
                    option_match = re.search(rf'{letter}\)\s*(.+?)(?=\s+[A-D]\)|Correct:|Explanation:|Chapter:|$)', block)
                    if option_match:
                        question['options'].append(f"{letter}) {option_match.group(1).strip()}")

                # Extract Correct
                correct_match = re.search(r'Correct:\s*([A-D])', block, re.IGNORECASE)
                if correct_match:
                    question['correct'] = correct_match.group(1).upper()

            elif question['type'] == 'open':
                # Extract Answer
                answer_match = re.search(r'Answer:\s*(.+?)(?=Explanation:|Chapter:|$)', block)
                if answer_match:
                    question['answer'] = answer_match.group(1).strip()

            # Extract Explanation (optional)
            expl_match = re.search(r'Explanation:\s*(.+?)(?=Chapter:|$)', block)
            if expl_match:
                question['explanation'] = expl_match.group(1).strip()

            # Extract Chapter (optional) - match to end of block, stripping whitespace
            chapter_match = re.search(r'Chapter:\s*(.+)', block, re.DOTALL)
            if chapter_match:
                # Get chapter and strip any trailing newlines/whitespace
                question['chapter'] = chapter_match.group(1).strip()

            # Validate and add
            if question['question'] and question['type']:
                if question['type'] == 'mc':
                    if len(question['options']) >= 2 and question['correct']:
                        questions.append(question)
                elif question['type'] == 'open':
                    if question['answer']:
                        questions.append(question)

        return questions, "" if questions else "No valid questions found"

    except Exception as e:
        return [], f"Error parsing single-line format: {str(e)}"


def load_gemini_questions(file_content: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Load questions from Gemini-formatted text
    Tries multi-line format first, then falls back to single-line format

    Expected format (multi-line):
    QUESTION 1
    Type: mc
    Question: What is...?
    A) Option A
    B) Option B
    C) Option C
    D) Option D
    Correct: A
    Explanation: Because...
    Chapter: 1.1

    Args:
        file_content: String content formatted by Gemini

    Returns:
        Tuple of (list of questions, error message if any)
    """
    try:
        questions = []

        # Split by QUESTION pattern
        question_blocks = re.split(r'\n+QUESTION\s+\d+\s*\n', file_content)

        # Remove empty first element if present
        if question_blocks and not question_blocks[0].strip():
            question_blocks.pop(0)

        for idx, block in enumerate(question_blocks, 1):
            if not block.strip():
                continue

            question = {
                'id': idx,
                'type': '',
                'question': '',
                'options': [],
                'correct': '',
                'answer': '',
                'explanation': '',
                'chapter': ''
            }

            lines = block.strip().split('\n')
            i = 0

            while i < len(lines):
                line = lines[i].strip()

                if line.startswith('Type:'):
                    question['type'] = line.split(':', 1)[1].strip().lower()

                elif line.startswith('Question:'):
                    question['question'] = line.split(':', 1)[1].strip()

                elif re.match(r'^[A-D]\)', line):
                    # Extract option (e.g., "A) Option text")
                    question['options'].append(line)

                elif line.startswith('Correct:'):
                    question['correct'] = line.split(':', 1)[1].strip().upper()

                elif line.startswith('Answer:'):
                    question['answer'] = line.split(':', 1)[1].strip()

                elif line.startswith('Explanation:'):
                    question['explanation'] = line.split(':', 1)[1].strip()

                elif line.startswith('Chapter:'):
                    question['chapter'] = line.split(':', 1)[1].strip()

                i += 1

            # Validate and add question
            if question['question'] and question['type']:
                if question['type'] == 'mc':
                    if len(question['options']) >= 2 and question['correct']:
                        questions.append(question)
                elif question['type'] == 'open':
                    if question['answer']:
                        questions.append(question)

        # If no questions found with multi-line format, try single-line format
        if not questions:
            questions, error = load_gemini_questions_singleline(file_content)
            if questions:
                return questions, ""
            return [], "No valid questions found. Please check format matches examples in instructions."

        return questions, ""

    except Exception as e:
        return [], f"Error parsing Gemini format: {str(e)}"


def load_questions(file_content: str, file_type: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Load questions from YAML, Markdown, or Gemini format

    Args:
        file_content: String content of file
        file_type: Either 'yaml', 'markdown', or 'gemini'

    Returns:
        Tuple of (list of questions, error message if any)
    """
    if file_type.lower() in ['yaml', 'yml']:
        return load_yaml_questions(file_content)
    elif file_type.lower() in ['markdown', 'md']:
        return load_markdown_questions(file_content)
    elif file_type.lower() == 'gemini':
        return load_gemini_questions(file_content)
    else:
        return [], f"Unsupported file type: {file_type}"


def sample_questions(questions: List[Dict[str, Any]], num_questions: int) -> List[Dict[str, Any]]:
    """
    Randomly sample questions without replacement

    Args:
        questions: List of all questions
        num_questions: Number of questions to sample

    Returns:
        List of sampled questions
    """
    if num_questions >= len(questions):
        return questions.copy()

    return random.sample(questions, num_questions)


def grade_quiz(questions: List[Dict[str, Any]], user_answers: Dict[int, str]) -> Dict[str, Any]:
    """
    Grade the quiz by comparing user answers to correct answers

    Args:
        questions: List of quiz questions
        user_answers: Dictionary mapping question IDs to user answers

    Returns:
        Dictionary containing grading results
    """
    total = len(questions)
    correct = 0
    results = []

    for q in questions:
        q_id = q['id']
        user_answer = user_answers.get(q_id, "").strip()

        if q['type'] == 'mc':
            correct_answer = q['correct']
            is_correct = user_answer.upper() == correct_answer.upper()

            # Get full text of options for MC questions
            user_answer_text = "No answer"
            correct_answer_text = correct_answer

            if user_answer:
                # Find the full option text for user's answer
                for option in q.get('options', []):
                    if option.strip().upper().startswith(user_answer.upper() + ')'):
                        user_answer_text = option.strip()
                        break

            # Find the full option text for correct answer
            for option in q.get('options', []):
                if option.strip().upper().startswith(correct_answer.upper() + ')'):
                    correct_answer_text = option.strip()
                    break

        else:  # open question
            # For open questions, exact match (case-insensitive)
            correct_answer = q['answer']
            is_correct = user_answer.lower() == correct_answer.lower()
            user_answer_text = user_answer if user_answer else "No answer"
            correct_answer_text = correct_answer

        if is_correct:
            correct += 1

        results.append({
            'id': q_id,
            'question': q['question'],
            'type': q['type'],
            'user_answer': user_answer_text,
            'correct_answer': correct_answer_text,
            'is_correct': is_correct,
            'explanation': q.get('explanation', 'No explanation provided'),
            'chapter': q.get('chapter', 'N/A')
        })

    score_percentage = (correct / total * 100) if total > 0 else 0

    return {
        'total': total,
        'correct': correct,
        'incorrect': total - correct,
        'score_percentage': score_percentage,
        'results': results
    }


class QuizPDF(FPDF):
    """Custom PDF class for quiz results"""

    def header(self):
        """Add header to each page"""
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Quiz Results Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        """Add footer to each page"""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def add_title_section(self, title: str, date_time: str, score: float, correct: int, total: int):
        """Add title section with score summary"""
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, title, 0, 1)

        self.set_font('Arial', '', 11)
        self.cell(0, 8, f'Date: {date_time}', 0, 1)
        self.ln(2)

        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, f'Score: {score:.1f}% ({correct}/{total} correct)', 0, 1, 'C', True)
        self.ln(5)

    def add_question_result(self, num: int, result: Dict[str, Any]):
        """Add individual question result"""
        # Question number and correctness indicator
        self.set_font('Arial', 'B', 11)
        status = '[CORRECT]' if result['is_correct'] else '[INCORRECT]'
        color = (0, 150, 0) if result['is_correct'] else (200, 0, 0)

        self.set_text_color(*color)
        self.cell(0, 8, f"Question {num}: {status}", 0, 1)
        self.set_text_color(0, 0, 0)

        # Question text
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 6, f"Q: {result['question']}")

        # User answer
        self.set_font('Arial', 'I', 10)
        self.multi_cell(0, 6, f"Your answer: {result['user_answer']}")

        # Correct answer
        self.set_font('Arial', 'B', 10)
        self.multi_cell(0, 6, f"Correct answer: {result['correct_answer']}")

        # Explanation
        self.set_font('Arial', '', 9)
        self.set_text_color(50, 50, 150)
        self.multi_cell(0, 5, f"Explanation: {result['explanation']}")
        self.set_text_color(0, 0, 0)

        # Chapter reference
        if result['chapter'] != 'N/A':
            self.set_font('Arial', 'I', 8)
            self.cell(0, 5, f"Chapter: {result['chapter']}", 0, 1)

        self.ln(4)


def generate_pdf(grading_results: Dict[str, Any], quiz_title: str = "Quiz") -> bytes:
    """
    Generate PDF report of quiz results

    Args:
        grading_results: Results from grade_quiz function
        quiz_title: Title of the quiz

    Returns:
        PDF content as bytes
    """
    pdf = QuizPDF()
    pdf.add_page()

    # Add title section
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.add_title_section(
        quiz_title,
        date_time,
        grading_results['score_percentage'],
        grading_results['correct'],
        grading_results['total']
    )

    # Add each question result
    for i, result in enumerate(grading_results['results'], 1):
        # Add new page if needed
        if pdf.get_y() > 250:
            pdf.add_page()

        pdf.add_question_result(i, result)

    # Return PDF as bytes
    return pdf.output(dest='S').encode('latin1')


def calculate_quiz_length(total_questions: int, percentage: int = None, custom_num: int = None) -> int:
    """
    Calculate number of questions based on percentage or custom number

    Args:
        total_questions: Total number of available questions
        percentage: Percentage of questions to use (10-100)
        custom_num: Custom number of questions

    Returns:
        Number of questions for the quiz
    """
    if custom_num is not None and custom_num > 0:
        return min(custom_num, total_questions)

    if percentage is not None:
        num = int(total_questions * percentage / 100)
        return max(1, min(num, total_questions))

    return total_questions
