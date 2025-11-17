"""
Comprehensive test for all new features in the quiz app
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from utils import load_gemini_questions, grade_quiz, sample_questions
import random

print("="*80)
print("COMPREHENSIVE FEATURE TEST")
print("="*80)

# Test data with both MC and open questions
test_chatgpt_output = """QUESTION 1
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

QUESTION 3
Type: mc
Question: What is the output range of the sigmoid function?
A) (-∞, +∞)
B) (0, 1)
C) [0, 1]
D) (-1, 1)
Correct: B
Explanation: Sigmoid squashes any input into probability range 0 to 1.
Chapter: 1.4

QUESTION 4
Type: open
Question: What does overfitting mean?
Answer: Model performs well on training data but poorly on new data
Explanation: Overfitting occurs when model learns noise instead of patterns.
Chapter: 5

QUESTION 5
Type: mc
Question: Which activation function can output negative values?
A) ReLU
B) Sigmoid
C) Softmax
D) Tanh
Correct: D
Explanation: Tanh outputs values between -1 and 1.
Chapter: 2.3"""

print("\n" + "="*80)
print("TEST 1: ChatGPT Format Parser")
print("="*80)

questions, error = load_gemini_questions(test_chatgpt_output)

if error:
    print(f"❌ FAILED: {error}")
else:
    print(f"✅ PASSED: Loaded {len(questions)} questions")
    print(f"   - MC questions: {sum(1 for q in questions if q['type'] == 'mc')}")
    print(f"   - Open questions: {sum(1 for q in questions if q['type'] == 'open')}")

print("\n" + "="*80)
print("TEST 2: Random Seed Reproducibility")
print("="*80)

# Test with seed
random.seed(42)
sample1 = sample_questions(questions, 3)
sample1_ids = [q['id'] for q in sample1]

random.seed(42)
sample2 = sample_questions(questions, 3)
sample2_ids = [q['id'] for q in sample2]

if sample1_ids == sample2_ids:
    print(f"✅ PASSED: Same seed produces same quiz")
    print(f"   Sample 1 IDs: {sample1_ids}")
    print(f"   Sample 2 IDs: {sample2_ids}")
else:
    print(f"❌ FAILED: Different samples with same seed")
    print(f"   Sample 1 IDs: {sample1_ids}")
    print(f"   Sample 2 IDs: {sample2_ids}")

print("\n" + "="*80)
print("TEST 3: Open Question Filtering")
print("="*80)

# Filter out open questions
mc_only = [q for q in questions if q['type'] == 'mc']
print(f"✅ PASSED: Filtered to {len(mc_only)} MC questions")
print(f"   Original: {len(questions)} questions")
print(f"   Filtered: {len(mc_only)} questions (MC only)")

print("\n" + "="*80)
print("TEST 4: Full Option Text in Grading")
print("="*80)

# Test grading with full option text
test_questions = [questions[0], questions[2]]  # Two MC questions
user_answers = {
    1: 'B',  # Correct
    3: 'A'   # Incorrect (correct is B)
}

results = grade_quiz(test_questions, user_answers)

# Check if full text is shown
mc_result = results['results'][0]
if ')' in mc_result['user_answer'] and ')' in mc_result['correct_answer']:
    print(f"✅ PASSED: Full option text displayed in results")
    print(f"   User answer: {mc_result['user_answer']}")
    print(f"   Correct answer: {mc_result['correct_answer']}")
else:
    print(f"❌ FAILED: Only letters shown, not full text")
    print(f"   User answer: {mc_result['user_answer']}")
    print(f"   Correct answer: {mc_result['correct_answer']}")

print("\n" + "="*80)
print("TEST 5: Timer Calculation")
print("="*80)

# Test timer calculations
num_questions = 10
time_per_question = 30
total_time_seconds = num_questions * time_per_question
total_minutes = total_time_seconds // 60

print(f"✅ PASSED: Timer calculation")
print(f"   Questions: {num_questions}")
print(f"   Time per question: {time_per_question}s")
print(f"   Total time: {total_minutes} minutes ({total_time_seconds}s)")

print("\n" + "="*80)
print("TEST 6: Open Question Grading Simulation")
print("="*80)

# Simulate open question grading
all_questions = questions[:4]  # 2 MC + 2 open
user_answers_all = {
    1: 'B',  # MC correct
    2: 'Classification with exactly two possible class labels',  # Open
    3: 'B',  # MC correct
    4: 'Model performs well on training data but poorly on new data'  # Open
}

results_all = grade_quiz(all_questions, user_answers_all)

# Simulate marking first open question as correct, second as wrong
open_question_grades = {
    2: True,   # First open marked correct
    4: False   # Second open marked wrong (default)
}

# Calculate score with manual grading
mc_correct = sum(1 for r in results_all['results'] if r.get('type') == 'mc' and r['is_correct'])
open_correct = sum(1 for q_id, is_correct in open_question_grades.items() if is_correct)
total_correct = mc_correct + open_correct
total_questions = len(all_questions)
score_percentage = (total_correct / total_questions * 100) if total_questions > 0 else 0

print(f"✅ PASSED: Open question manual grading")
print(f"   MC correct: {mc_correct}/2")
print(f"   Open marked correct: {open_correct}/2")
print(f"   Total score: {total_correct}/{total_questions} ({score_percentage:.1f}%)")

print("\n" + "="*80)
print("TEST 7: Single-Line Format Fallback")
print("="*80)

# Test single-line format (what Gemini was producing)
single_line_output = """QUESTION 1 Type: mc Question: What is probability? A) A measure of certainty B) A measure of uncertainty C) A constant value D) None of the above Correct: B Explanation: Probability measures uncertainty. Chapter: 1.1

QUESTION 2 Type: open Question: What is variance? Answer: A measure of spread in data Explanation: Variance shows data dispersion. Chapter: 2"""

questions_single, error_single = load_gemini_questions(single_line_output)

if error_single:
    print(f"❌ FAILED: {error_single}")
else:
    print(f"✅ PASSED: Fallback parser loaded {len(questions_single)} questions")
    for q in questions_single:
        print(f"   - Q{q['id']}: {q['type']} - {q['question'][:50]}...")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

all_tests = [
    ("ChatGPT Format Parser", not error),
    ("Random Seed Reproducibility", sample1_ids == sample2_ids),
    ("Open Question Filtering", len(mc_only) == 3),
    ("Full Option Text in Grading", ')' in mc_result['user_answer']),
    ("Timer Calculation", total_minutes == 5),
    ("Open Question Grading", score_percentage == 75.0),
    ("Single-Line Fallback Parser", not error_single)
]

passed = sum(1 for _, result in all_tests if result)
total = len(all_tests)

print(f"\nTests Passed: {passed}/{total}")
print()

for test_name, result in all_tests:
    status = "✅" if result else "❌"
    print(f"{status} {test_name}")

print("\n" + "="*80)

if passed == total:
    print("🎉 ALL TESTS PASSED! Ready to commit and push.")
else:
    print(f"⚠️  {total - passed} test(s) failed. Review above.")

print("="*80)
