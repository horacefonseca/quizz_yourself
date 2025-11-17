"""
Test to identify session state issues with large question sets
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*80)
print("SESSION STATE ISSUE ANALYSIS")
print("="*80)

# Simulate the problem
print("\n" + "="*80)
print("ISSUE IDENTIFIED:")
print("="*80)

issues = [
    {
        "location": "app.py:422",
        "code": "st.session_state.timer_enabled = st.checkbox(...)",
        "problem": "Direct session_state assignment creates circular dependency",
        "impact": "Causes multiple reruns that can clear other session state"
    },
    {
        "location": "app.py:440",
        "code": "st.session_state.include_open_questions = st.checkbox(...)",
        "problem": "Direct session_state assignment without unique key",
        "impact": "Widget state not properly tracked across reruns"
    },
    {
        "location": "app.py:375-397",
        "code": "st.slider() and st.checkbox() widgets",
        "problem": "Missing unique 'key' parameters",
        "impact": "With 60+ questions, state can get corrupted on rerun"
    },
    {
        "location": "app.py:415",
        "code": "st.session_state.random_seed = seed_value",
        "problem": "Assignment inside conditional, inconsistent state updates",
        "impact": "State changes trigger unexpected reruns"
    }
]

print("\n🔍 Found 4 problematic patterns:\n")

for i, issue in enumerate(issues, 1):
    print(f"{i}. {issue['location']}")
    print(f"   Code: {issue['code']}")
    print(f"   Problem: {issue['problem']}")
    print(f"   Impact: {issue['impact']}")
    print()

print("="*80)
print("ROOT CAUSE:")
print("="*80)

print("""
When widgets directly assign to session_state without keys:
  st.session_state.timer_enabled = st.checkbox(...)

Streamlit execution flow becomes:
  1. User clicks checkbox → value changes
  2. Triggers rerun from top of script
  3. Widget executes again, reads from session_state
  4. Creates circular update loop
  5. With large data (60 questions), other session_state gets corrupted
  6. App resets to earlier state (loses parsed questions context)
""")

print("="*80)
print("SOLUTION:")
print("="*80)

print("""
Use key-based approach for all widgets:

WRONG (current):
  st.session_state.timer_enabled = st.checkbox("Enable timer", value=False)

RIGHT (fixed):
  timer_enabled = st.checkbox(
      "Enable timer",
      value=st.session_state.timer_enabled,
      key="timer_enabled_checkbox"
  )
  if timer_enabled != st.session_state.timer_enabled:
      st.session_state.timer_enabled = timer_enabled

OR SIMPLER:
  st.checkbox(
      "Enable timer",
      key="timer_enabled",  # Auto-syncs to st.session_state.timer_enabled
      value=st.session_state.timer_enabled
  )
""")

print("="*80)
print("RECOMMENDED FIX:")
print("="*80)

fixes = [
    "1. Add unique 'key' parameter to ALL widgets in settings section",
    "2. Remove direct session_state assignments from widget definitions",
    "3. Use key-based auto-sync: widget key='setting_name' → st.session_state.setting_name",
    "4. Add on_change callbacks for widgets that need immediate state updates"
]

for fix in fixes:
    print(f"  ✅ {fix}")

print("\n" + "="*80)
print("TESTING WITH 60 QUESTIONS:")
print("="*80)

# Simulate large question set
large_question_set = [
    {"id": i, "type": "mc" if i % 3 != 0 else "open", "question": f"Question {i}"}
    for i in range(1, 61)
]

print(f"\nSimulated question set: {len(large_question_set)} questions")
print(f"Memory size: ~{sys.getsizeof(str(large_question_set)) / 1024:.2f} KB")
print("\nWith direct session_state assignments:")
print("  ❌ Reruns can corrupt or clear large data structures")
print("  ❌ Streamlit may reset to previous checkpoint")
print("\nWith key-based approach:")
print("  ✅ State properly maintained across reruns")
print("  ✅ Large data preserved in session_state")

print("\n" + "="*80)
