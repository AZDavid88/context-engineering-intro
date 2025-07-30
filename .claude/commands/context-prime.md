---
allowed-tools: Read, Glob, Grep, Bash(find:*), Bash(ls:*), Bash(git:*), LS
description: Set up AI with essential project knowledge instantly using IndyDevDan methodology
argument-hint: [project-path] | current directory if no path provided
---

# Project Context Priming - AI Knowledge Acceleration

**Context**: You are using the CodeFarm methodology to rapidly establish AI context for a project. This follows IndyDevDan's "Context is KING" principle - give AI tools the right information to understand and work with the project effectively.

## Quick Project Discovery

### 1. Essential Files Analysis
Execute using Claude Code tools to build foundational understanding:

**Project root structure analysis:**
- Use LS tool to examine the directory structure of ${ARGUMENTS:-"."}
- Use Glob tool with pattern "*" to list all files in root directory

**Language detection:**
- Use Glob tool with patterns "**/*.py", "**/*.js", "**/*.ts", "**/*.go", "**/*.rs", "**/*.java" to identify primary languages
- Count files by examining Glob results to determine main language

**Configuration files discovery:**
- Use Glob tool with patterns "*.toml", "*.json", "requirements.txt", "package.json", "Cargo.toml", "go.mod" in ${ARGUMENTS:-"."}
- Use Glob tool with pattern "**/*.toml", "**/*.json" for nested config files

**Git repository information:**
- Use Bash tool with command: `git remote -v` in ${ARGUMENTS:-"."} directory
- Use Bash tool with command: `git status --porcelain` to check if it's a git repo

### 2. Three-Directory System Check
Use Read and LS tools to look for IndyDevDan's essential directories:
- **AI Docs**: Check for ai_docs/, docs/, documentation/ directories using LS tool
- **Specs**: Check for specs/, plans/, specifications/ directories using LS tool
- **Claude Commands**: Check for .claude/commands/ directory using LS tool

### 3. Critical Documentation Reading
Use Read tool to examine these files if they exist (in priority order):
- **Project overview**: README.md, README.rst, or project documentation
- **Planning docs**: PLANNING.md, ARCHITECTURE.md, or ai_docs/project_overview.md
- **Dependencies**: requirements.txt, package.json, Cargo.toml, or go.mod
- **Configuration**: pyproject.toml, setup.py, or config.yaml

## Context Knowledge Building

### Technology Stack Identification
Based on files found using the tools above, identify:
- **Primary language** and version
- **Framework/libraries** used
- **Database/storage** systems
- **Testing framework**
- **Build/deployment** tools

### Project Purpose & Architecture
Extract from documentation using Read tool:
- **What does this project do?**
- **Key business logic or domain**
- **Main modules/components**
- **External dependencies/APIs**
- **Data flow patterns**

### Current Development State
Assess using Bash and Glob tools:
- **Active development** (use Bash tool with `git log --oneline -10` for recent commits)
- **Testing coverage** (use Glob tool with patterns "**/test*", "tests/**/*")
- **Documentation quality** (examine documentation files with Read tool)
- **Code organization** (analyze directory structure with LS and Glob tools)

## AI Context Summary Generation

Based on the analysis above, create a comprehensive context summary:

### Project Identity
```
**Project**: [Name and purpose]
**Language**: [Primary language + version]  
**Framework**: [Main framework/libs]
**Architecture**: [Key architectural patterns]
```

### Key Information for AI Tools
```
**Import Patterns**: [How modules are structured]
**Testing Setup**: [How to run tests]
**Key Dependencies**: [Critical external APIs/services]
**Development Workflow**: [How changes are made]
```

### Critical Context for Development
```
**Code Conventions**: [Naming, structure patterns]
**Key Directories**: [Where important code lives]
**Configuration**: [How settings are managed]
**External Integrations**: [APIs, databases, services]
```

## Quick Reference Generation

Create quick-access information for future sessions:

### Essential Commands
List the key commands someone would need:
```bash
# Setup/Installation
[installation commands]

# Testing  
[test commands]

# Development
[development server/build commands]
```

### Key Files & Directories
```
src/               # [Description of main code]
tests/             # [Testing approach]
docs/              # [Documentation location]
config/            # [Configuration files]
```

### API/Integration Points
```
External APIs:     [List key external services]
Database:          [Database type and connection]
Authentication:    [How auth is handled]
Environment Vars:  [Critical env variables]
```

## Context Validation

### Verify Understanding
After building context, validate by asking:
- Can I identify the main business logic?
- Do I understand how to run/test this code?
- Are the key dependencies and integrations clear?
- Would I know where to make common changes?

### Missing Context Alert
Flag if any of these are unclear:
- [ ] Project purpose and domain
- [ ] How to set up development environment
- [ ] Testing strategy and commands
- [ ] Key business logic location
- [ ] External service integrations
- [ ] Configuration management approach

## Output Format

Present the context in this structured format:

```markdown
# Project Context Summary

## Overview
[2-3 sentences about what this project does]

## Technical Stack
- **Language**: [Primary language + version]
- **Framework**: [Key frameworks]
- **Database**: [Data storage]
- **Testing**: [Test framework]

## Key Information
- **Main Code**: [Where core logic lives]
- **Entry Points**: [How the application starts]
- **Configuration**: [How settings work]
- **Dependencies**: [Critical external services]

## Development Workflow
- **Setup**: [How to get started]
- **Testing**: [How to run tests]
- **Common Tasks**: [Frequent development activities]

## Integration Points
- **External APIs**: [Key external services]
- **Data Flow**: [How data moves through system]
- **Authentication**: [How security works]

## Next Steps
[What an AI tool should know to be immediately productive]
```

---

This context priming becomes your standard first step when working with any project, ensuring AI tools can immediately contribute effectively without extensive back-and-forth exploration.