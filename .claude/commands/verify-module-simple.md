---
description: Systematic module verification with comprehensive documentation
allowed-tools: Read, Write, Glob, Grep, LS
argument-hint: [module-path] - Path to module directory for verification
---

# Module Verification & Documentation

You are going to systematically verify and document a code module by analyzing what the code actually does, not what documentation claims it does.

## Your Task

For the module at `$ARGUMENTS`, create comprehensive documentation based on actual code analysis:

### Step 1: Discover Module Structure
- Use LS to understand the directory structure
- Use Glob to find all Python files (`*.py`)
- Map out the overall organization

### Step 2: Analyze Every Function
For each function you find:
- Read the complete file containing the function
- Understand what the function actually does (not what docs say)
- Identify inputs, outputs, and side effects
- Note any discrepancies between documentation and implementation
- Trace dependencies and function calls

### Step 3: Map Data Flow
- Follow how data moves through the module
- Identify external dependencies (APIs, files, databases)
- Document data transformations and processing steps
- Map integration points with other modules

### Step 4: Assess Dependencies
- List all imports and external libraries
- Identify configuration dependencies
- Document error handling patterns
- Note any potential reliability issues

## Analysis Approach

**Be Evidence-Based**: Only document what you can verify by reading the actual code. Don't assume anything.

**Be Thorough**: Read every file completely. Don't skip files or make assumptions based on filenames.

**Be Systematic**: Analyze each function methodically. Don't rush through the analysis.

**Question Everything**: If documentation exists, verify it matches the implementation. Note mismatches.

## Output Requirements

Create these files in `verified_docs/by_module_simplified/[module-name]/`:

### 1. `function_verification_report.md`
Document every function with:
- Function name and location (file:line)
- What it actually does (based on code analysis)
- Parameters and return values (actual behavior, not docs)
- Dependencies and side effects
- Verification status: ✅ Matches docs, ⚠️ Partial match, ❌ Mismatch, 🔍 Undocumented

### 2. `data_flow_analysis.md`
Map how data moves through the module:
- Input sources (parameters, files, APIs, databases)
- Processing stages and transformations
- Output destinations and side effects
- Error handling and edge cases

### 3. `dependency_analysis.md`
Document all dependencies:
- Internal dependencies (other functions/modules)
- External dependencies (libraries, APIs, services)
- Configuration requirements
- Potential failure points and reliability concerns

## Quality Standards

- **95% Confidence**: Only make claims you can back up with specific code evidence
- **No Assumptions**: Don't guess what code does - read it and understand it
- **Evidence-Based**: Include specific code examples to support your analysis
- **Complete Coverage**: Analyze every function, don't skip anything

## Success Criteria

Your verification is successful when:
1. Every function in the module has been analyzed and documented
2. Data flow through the module is completely mapped
3. All dependencies are identified and assessed
4. Documentation reflects actual code behavior, not assumptions
5. Any discrepancies between docs and implementation are clearly noted

Start with the module structure discovery and work systematically through each file.