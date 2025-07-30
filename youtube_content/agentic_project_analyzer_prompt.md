# The Agentic Project Analyzer & Structure Generator

## Meta-Prompt for Understanding and Creating AI-Optimized Project Structure

This prompt implements IndyDevDan's proven methodology from the YouTube videos, combining the 3-folder system with agentic coding principles.

---

## THE SYSTEMATIC PROMPT

```markdown
# Agentic Project Analyzer & Structure Generator

You are an expert AI coding consultant specializing in agentic workflow optimization. Your task is to analyze any codebase and create the essential 3-folder structure that will dramatically improve AI coding tool performance.

## CONTEXT UNDERSTANDING PHASE

**Step 1: Project Analysis**
- Read the README, package.json, requirements.txt, or equivalent project files
- Run `git ls-files` or equivalent to understand the codebase structure
- Identify the tech stack, frameworks, and primary functionality
- Determine the project's complexity level and current AI tool compatibility

**Step 2: Context Assessment**
Answer these critical questions:
- What is this project's core purpose and functionality?
- What are the main technical dependencies and integrations?
- What development patterns and conventions are already in use?
- Where are the pain points for AI tool understanding?
- What documentation or context is missing or unclear?

## THE 3-FOLDER STRUCTURE IMPLEMENTATION

**Step 3: Create AI-Docs Directory**
Purpose: Persistent memory/knowledge repository for AI coding tools

Create `ai-docs/` with these essential files:
- `project-overview.md` - High-level project summary and architecture
- `tech-stack.md` - Complete technology stack and version details
- `api-integrations.md` - Third-party APIs, services, and their documentation
- `custom-patterns.md` - Project-specific patterns, conventions, and best practices
- `troubleshooting.md` - Common issues and their solutions
- `[framework]-best-practices.md` - Framework-specific implementation notes

**Step 4: Create Specs Directory**
Purpose: Plans and specifications - "The plan IS the prompt"

Create `specs/` with:
- `current-architecture.md` - Detailed current system architecture
- `development-roadmap.md` - Planned features and improvements
- `feature-templates/` - Standardized templates for new feature specs
- Example feature spec following this structure:
  ```markdown
  # Feature Name
  ## Problem Statement
  ## Solution Overview
  ## Technical Implementation
  ## Self-Validation Steps
  ## Testing Requirements
  ## Dependencies & Integration Points
  ```

**Step 5: Create .claude Directory (or equivalent)**
Purpose: Reusable prompts and commands for any AI coding tool

Create `.claude/commands/` with essential prompts:
- `context-prime.md` - Primary context setup command
- `feature-spec-generator.md` - Creates feature specifications
- `code-review.md` - Comprehensive code review prompts
- `refactor-analyzer.md` - Code refactoring analysis
- `test-generator.md` - Test creation prompts
- `documentation-updater.md` - Documentation maintenance

**Essential Context Prime Command:**
```markdown
# Context Prime

## Purpose
Quickly set up AI coding tool with essential project context

## Instructions
1. Read README.md and understand project purpose
2. Run appropriate file listing command (git ls-files, ls -la, etc.)
3. Review ai-docs/ directory for project-specific knowledge
4. Summarize project structure and readiness for development
5. Confirm understanding of tech stack and current development patterns

## Response Format
Provide concise summary:
- Project purpose and main functionality
- Tech stack and key dependencies
- Current development status
- Ready for: [specific type of work - features, debugging, refactoring, etc.]
```

## AGENTIC CODING OPTIMIZATION

**Step 6: Workflow Enhancement**
- Create feature-specific context priming commands for complex changes
- Set up self-validating loops in specs (testing, validation steps)
- Design prompt chains that work across AI coding tools (Claude Code, Cursor, Codex)
- Implement "plan-first" development approach where specs become prompts

**Step 7: Scalability Preparation**
- Design modular prompt structure for sub-agent compatibility
- Create clear information flow patterns between different AI tools
- Set up reusable patterns that work across multiple projects
- Establish context window management strategies

## QUALITY ASSURANCE

**Step 8: Validation**
- Verify all essential directories are created with appropriate content
- Test context prime command with the target AI coding tool
- Ensure documentation is specific enough to be actionable
- Confirm reusable prompts work across different development scenarios

**Step 9: Continuous Improvement Setup**
- Create feedback loops for prompt effectiveness
- Set up templates for adding new AI tools or workflows
- Design update mechanisms for keeping ai-docs current
- Establish patterns for scaling to multiple projects

## SUCCESS CRITERIA

A successfully implemented agentic project structure should:
✅ Enable AI tools to understand the project faster than reading the README
✅ Provide comprehensive context without overwhelming the AI tool
✅ Scale across different AI coding tools (tool-agnostic)
✅ Support both small fixes and large feature development
✅ Include self-validating workflows and testing patterns
✅ Maintain relevance as the project evolves

## KEY PRINCIPLES

1. **Context is King** - But balance is critical (too much confuses, too little prevents work)
2. **The Plan IS the Prompt** - Detailed specs become executable prompts
3. **Reusability Scales** - Create once, use across projects and time
4. **Agentic > Iterative** - Plan-first approach beats back-and-forth prompting
5. **Tool Agnostic** - Structure works with any AI coding assistant
6. **Self-Validation** - Build testing and verification into every workflow

Execute this analysis and structure creation systematically. Focus on creating immediately actionable, project-specific content that will measurably improve AI coding tool performance.
```

---

## USAGE INSTRUCTIONS

1. **Copy this entire prompt** into your AI coding tool
2. **Run it on any project** - it will analyze and create the 3-folder structure
3. **Customize the generated content** based on your specific project needs
4. **Test the context prime command** to verify effectiveness
5. **Iterate and improve** based on real usage patterns

This prompt embodies the proven methodologies from IndyDevDan's videos and scales to any project size or complexity level.