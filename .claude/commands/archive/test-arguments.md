---
description: Test command to diagnose $ARGUMENTS parameter passing
allowed-tools: Bash, Read, LS
---

# Argument Testing Command

**Testing Parameter:** $ARGUMENTS

## Method 1: Direct Text Substitution
The provided argument is: $ARGUMENTS

## Method 2: Bash Variable Test
Let me test if bash can see the argument:

!`echo "Argument received: $ARGUMENTS"`

## Method 3: Environment Variable Approach
Testing with environment variable passing:

!`PROJECT_PATH="$ARGUMENTS" && echo "Project path: $PROJECT_PATH"`

## Method 4: Static Path Test
Testing with a known static path:

!`echo "Static test: /workspaces/context-engineering-intro/projects/quant_trading"`

## Method 5: LS Tool Test
Using LS tool with the argument path:

Let me try to list the directory using LS tool with path: $ARGUMENTS