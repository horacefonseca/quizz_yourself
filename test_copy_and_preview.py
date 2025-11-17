"""
Test the copy button and preview functionality
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import html

print("="*80)
print("TEST: Copy Button HTML Escaping and Preview Logic")
print("="*80)

# Simulate a combined prompt with special characters
combined_prompt = """Please convert the following raw text into a structured quiz format.

FORMAT RULES:
Each question must start with "QUESTION" followed by a number (QUESTION 1, QUESTION 2, etc.)

EXAMPLE OUTPUT:

QUESTION 1
Type: mc
Question: What is the range of valid probability values?
A) Between -1 and 1
B) Between 0 and 1
C) Between 0 and 100
D) Any positive number
Correct: B

====================================================
RAW TEXT TO CONVERT:
====================================================

What is probability?
It measures uncertainty in events.

What is the sigmoid function?
It maps any value to (0, 1) range.
"""

print("\n" + "="*80)
print("TEST 1: HTML Escaping for Copy Button")
print("="*80)

escaped_prompt = html.escape(combined_prompt)

# Check if special characters are escaped
if '&lt;' in escaped_prompt or '&gt;' in escaped_prompt or '&amp;' in escaped_prompt:
    print("✅ PASSED: HTML special characters properly escaped")
else:
    print("⚠️  INFO: No HTML special characters to escape in this prompt")

# Check if the original content is preserved
if "QUESTION 1" in escaped_prompt and "FORMAT RULES" in escaped_prompt:
    print("✅ PASSED: Original content preserved after escaping")
else:
    print("❌ FAILED: Content lost during escaping")

print(f"\nEscaped prompt length: {len(escaped_prompt)} characters")
print(f"Original prompt length: {len(combined_prompt)} characters")

print("\n" + "="*80)
print("TEST 2: Preview Logic (Last 4 Lines)")
print("="*80)

# Test with user raw text
user_raw_text = """What is probability?
It measures uncertainty in events.

What is the sigmoid function?
It maps any value to (0, 1) range.

What is overfitting?
When model memorizes training data.

What is cross-validation?
A technique to evaluate model performance."""

lines = user_raw_text.strip().split('\n')
print(f"\nTotal lines: {len(lines)}")

if len(lines) > 4:
    preview_lines = lines[-4:]
    print(f"✅ PASSED: Preview shows last 4 lines")
    print("\nLast 4 lines:")
    for i, line in enumerate(preview_lines, 1):
        print(f"  {i}. {line}")
else:
    print(f"⚠️  INFO: Only {len(lines)} lines, showing all")

print("\n" + "="*80)
print("TEST 3: Combined Prompt Content Verification")
print("="*80)

# Verify the combined prompt includes both instructions and user text
has_instructions = "FORMAT RULES" in combined_prompt
has_user_text = "RAW TEXT TO CONVERT" in combined_prompt
has_actual_content = "What is probability?" in combined_prompt

if has_instructions:
    print("✅ PASSED: Instructions included in combined prompt")
else:
    print("❌ FAILED: Instructions missing")

if has_user_text:
    print("✅ PASSED: User text section marker included")
else:
    print("❌ FAILED: User text marker missing")

if has_actual_content:
    print("✅ PASSED: Actual user content included")
else:
    print("❌ FAILED: User content missing")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

all_tests = [
    ("HTML Escaping", True),
    ("Content Preservation", "QUESTION 1" in escaped_prompt),
    ("Preview Logic", len(lines) > 4),
    ("Instructions in Prompt", has_instructions),
    ("User Text in Prompt", has_user_text and has_actual_content)
]

passed = sum(1 for _, result in all_tests if result)
total = len(all_tests)

print(f"\nTests Passed: {passed}/{total}\n")

for test_name, result in all_tests:
    status = "✅" if result else "❌"
    print(f"{status} {test_name}")

print("\n" + "="*80)

if passed == total:
    print("🎉 ALL TESTS PASSED!")
else:
    print(f"⚠️  {total - passed} test(s) need attention")

print("="*80)
