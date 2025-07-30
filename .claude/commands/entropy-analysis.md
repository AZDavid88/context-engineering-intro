---
allowed-tools: Bash(find:*), Bash(wc:*), Bash(git:*), Bash(python:*), Glob, Grep, LS, Read
description: Fresh entropy analysis without caching issues
argument-hint: [path-to-project] | analyze current directory if no path provided
---

# Project Entropy Assessment & Restructuring Plan (FRESH VERSION)

**Context**: You are using the CodeFarm methodology to assess project entropy. This version uses only proper Claude Code tools.

## Execute Project Analysis

### 1. Directory Structure Analysis
Start with these tool calls:

**Use LS tool** to examine ${ARGUMENTS:-"."} directory structure
**Use Glob tool** with pattern "**/*/" to find all subdirectories  
**Use Glob tool** with patterns "**/*.py", "**/*.js", "**/*.ts" to count code files

### 2. Git Status Check
**Use Bash tool** with command: `git status --porcelain` in ${ARGUMENTS:-"."} directory

### 3. Test File Analysis  
**Use Glob tool** with patterns "**/test*", "**/*test*", "**/test_*" to locate test files

### 4. Dependency Analysis
**Use Glob tool** to find: requirements.txt, package.json, Cargo.toml, go.mod
**Use Read tool** to examine these dependency files
**Use Grep tool** to search for commented dependencies or import issues

## Analysis Results

After using the tools above, provide:

1. **Root Directory Clutter Count** - How many files are in the root?
2. **Code File Distribution** - Where are the main code files located?  
3. **Test Organization** - Are tests properly organized or scattered?
4. **Git Status** - How many uncommitted changes?
5. **Dependency Health** - Any issues with requirements or imports?

## Entropy Assessment

### High Entropy Indicators
- 10+ files in root directory
- Debug/test files outside proper directories  
- Multiple backup/cleanup files
- Commented dependencies
- Import path manipulations

### Restructuring Recommendations
Based on findings, suggest:
1. **Immediate fixes** (safe directory creation)
2. **File reorganization** (with git safety)
3. **Documentation improvements** (ai_docs/ structure)

---

**Try this fresh command**: `/entropy-analysis /workspaces/context-engineering-intro/projects/quant_trading`