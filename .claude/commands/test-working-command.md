---
description: Test command to verify working tool patterns
allowed-tools: LS, Glob, Grep
argument-hint: [project-path] - Path to test
---

# Test Working Command

**Target:** $ARGUMENTS

## Testing Direct Tool Invocations

Let me test the working patterns:

**Step 1: Directory Analysis**

Using LS tool to analyze directory at path: $ARGUMENTS

**Step 2: Python File Discovery**  

Using Glob tool to find Python files with path: $ARGUMENTS and pattern: "*.py"

**Step 3: Import Analysis**

Using Grep tool with path: $ARGUMENTS, pattern: "^import|^from.*import" and output_mode: "files_with_matches"

---

**✅ Test Complete:** If you see tool invocations working above, then our approach is correct and the issue is with command caching.