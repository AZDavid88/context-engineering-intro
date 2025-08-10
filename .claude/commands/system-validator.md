---
description: Validate complex systems through observable behavior - measure what they actually do, not what they claim
allowed-tools: Bash, Read, Write, Glob, Grep, LS, TodoWrite
argument-hint: [system-path] - Path to system for validation (defaults to current directory)
---

# System Validator
## Measure Reality, Not Documentation

**Purpose**: Find out if your complex system actually works by testing what it does, not what it says it does.

## System Context
!`pwd && echo "=== TARGET SYSTEM ===" && ls -la`
!`find . -maxdepth 2 -name "*.py" -executable -o -name "run_*.py" -o -name "main.py" -o -name "docker-compose.yml" | head -10`

## Your Task

For the system at `$ARGUMENTS` (current directory if not specified):

### Find What Can Actually Run
- Discover executable scripts, entry points, and main orchestrators
- Map directory structure and identify component boundaries  
- Find configuration files that control real system behavior
- Trace imports to understand actual dependencies

### Test What Actually Happens
- Execute main components safely and monitor their behavior
- Check if components can actually talk to each other with real data
- Test what happens when things break (missing files, bad data, failures)
- Measure resource usage, timing, and performance patterns

### Compare Claims vs Reality  
- Check if existing tests actually test real functionality
- Verify integration points work with actual data flows
- Identify gaps between documentation and observable behavior
- Spot validation theater (tests that pass but don't validate real function)

### Document What You Find
- Report what the system actually does based on evidence
- Identify specific integration problems with examples
- Recommend concrete fixes for broken connections
- Distinguish working components from system-level issues

## Success Criteria

You succeed when you can answer:
- What does this system actually do when you run it?
- Which parts work independently vs together?
- Where do components fail to connect properly?
- What needs fixing to make it work as intended?

## Evidence Standards

- Only document what you can demonstrate through execution
- Test integration points with real data, not mocked interfaces  
- Measure actual performance under realistic conditions
- Identify specific failure points through actual testing

Start by finding what can run, then test if it actually works together.