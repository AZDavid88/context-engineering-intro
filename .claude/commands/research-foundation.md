---
description: Create comprehensive research foundation for implementation plans using Playwright + Jina r.reader API
allowed-tools: Read, Glob, mcp__playwright__browser_navigate, mcp__playwright__browser_snapshot, mcp__playwright__browser_click, mcp__playwright__browser_links, WebFetch, Bash, Write, mcp__filesystem__create_directory, mcp__filesystem__write_file
argument-hint: [implementation_plan_path] - Path to implementation plan requiring research foundation
---

# Research Foundation Builder - Anti-Hallucination Documentation System

## Dynamic Project Context Discovery
!`pwd`
!`ls -la`
!`echo "Using provided Jina API key for r.reader extraction"`

## Your Task - Create Comprehensive Research Foundation
**Objective:** Extract all external dependencies from implementation plan and build comprehensive research documentation using **Playwright navigation + Jina r.reader API extraction** for anti-hallucination foundation.

**CRITICAL:** This command uses **Jina r.reader API** (https://r.jina.ai/) for superior content extraction optimized for LLMs using the provided API key.

## Phase 1: Implementation Plan Analysis & Dependency Discovery
**Comprehensive analysis of implementation requirements**
- Use Read to analyze the specified implementation plan file
- Extract all external technologies, APIs, frameworks, and services mentioned
- Identify documentation URLs and research requirements
- Create dependency mapping with technology → documentation URL pairs
- Evidence validation: All dependencies extracted and categorized

**Key Extraction Targets:**
- Database systems (Neon, PostgreSQL, etc.)
- Cloud services (AWS, GCP, Azure)
- APIs and integrations
- Frameworks and libraries
- Configuration and deployment tools

## Phase 2: Systematic Documentation Discovery (Playwright MCP)
**Navigate documentation sites to discover all relevant content**
- Use mcp__playwright__browser_navigate to access each technology's documentation site
- Use mcp__playwright__browser_snapshot to understand site structure
- Use mcp__playwright__browser_links to discover all relevant documentation pages
- Use mcp__playwright__browser_click to navigate through documentation sections
- Build comprehensive URL inventory for each technology
- Evidence validation: Complete documentation site mapping

**Navigation Strategy:**
- Start with main documentation URLs from implementation plan
- Navigate through Getting Started, API Reference, Integration Guides
- Follow navigation menus to discover all relevant sections
- Capture URLs for: tutorials, API docs, configuration guides, best practices
- Build systematic URL lists for Jina extraction

## Phase 3: Superior Content Extraction (Jina r.reader API)
**Extract clean, LLM-optimized content using Jina r.reader API**
- Use Bash commands to call Jina r.reader API for each discovered URL
- Apply optimal headers for clean content extraction:
  - `X-With-Links-Summary: true` for related links
  - `X-Target-Selector: main,.content,.documentation` for focused content
  - `X-Remove-Selector: header,footer,.sidebar` for clean extraction
- Extract comprehensive content including code examples and API schemas
- Evidence validation: All content extracted and verified for completeness

**Jina API Implementation:**
```bash
# For each discovered URL, extract using Jina r.reader
curl -X POST 'https://r.jina.ai/' \
  -H "Authorization: Bearer jina_0652bfc906d14590bf46815dc705aab8e7T5kQ5d6RF7vu3QK9Odfn2UjjK6" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -H "X-With-Links-Summary: true" \
  -H "X-Target-Selector: main,.content,.documentation" \
  -H "X-Remove-Selector: header,footer,.sidebar" \
  -d '{"url":"$DOCUMENTATION_URL"}'
```

## Phase 4: Research Organization & Documentation Structure
**Create systematic research database**
- Use mcp__filesystem__create_directory to create `/research/[technology]/` directories
- Use mcp__filesystem__write_file to store extracted content as structured markdown
- Create technology-specific research summaries and quick references
- Organize by: getting_started.md, api_reference.md, integration_guides.md, best_practices.md
- Evidence validation: All research properly organized and accessible

**Directory Structure:**
```
/research/
├── neon/
│   ├── getting_started.md          # From Jina extraction
│   ├── authentication_api.md       # From Jina extraction  
│   ├── connection_pooling.md       # From Jina extraction
│   ├── query_optimization.md       # From Jina extraction
│   └── research_summary.md         # Synthesized overview
├── postgresql/
│   ├── async_queries.md            # From Jina extraction
│   ├── connection_management.md    # From Jina extraction
│   └── research_summary.md         # Synthesized overview
└── validation/
    └── implementation_readiness_report.md
```

## Phase 5: Anti-Hallucination Validation & Implementation Readiness
**Cross-reference implementation plan against comprehensive research**
- Compare implementation plan requirements with extracted documentation
- Identify potential conflicts, missing requirements, or outdated assumptions
- Validate API endpoints, authentication methods, and integration approaches
- Generate validated implementation roadmap with documentation backing
- Evidence validation: All implementation approaches verified against official docs

## Phase 6: Research Summary & Quick Reference Generation
**Create actionable research summaries for implementation**
- Generate technology-specific quick reference guides
- Extract key API endpoints, authentication patterns, and configuration examples
- Create implementation checklists based on official documentation
- Build cross-technology integration notes
- Evidence validation: All summaries accurate to source documentation

## Output Requirements & Quality Standards
**Anti-hallucination safeguards and implementation-ready research foundation**

### Quality Standards:
- **Complete Documentation Coverage**: All discovered pages extracted via Jina
- **LLM-Optimized Content**: Clean, structured content perfect for AI analysis
- **Implementation Ready**: All necessary integration details captured
- **Anti-Hallucination Protection**: Official documentation prevents implementation errors
- **Systematic Organization**: Easy navigation and quick reference capability
- **Cross-Technology Integration**: Documentation supports multi-technology implementations

### Expected Deliverables:
1. **Comprehensive Research Directories**: `/research/[technology]/` for each dependency
2. **Jina-Extracted Documentation**: Clean, structured content from all relevant pages
3. **Research Summaries**: Technology-specific overviews and quick references
4. **Implementation Readiness Report**: Validation of implementation plan against research
5. **Integration Guides**: Cross-technology implementation approaches
6. **API Reference Collections**: Complete API documentation for all dependencies

### Success Criteria:
- ✅ All external dependencies researched comprehensively
- ✅ Jina r.reader API successfully extracts all documentation content
- ✅ Research organized in systematic, navigable structure
- ✅ Implementation plan validated against official documentation
- ✅ Anti-hallucination foundation established for error-free implementation
- ✅ Research database ready to support immediate development work

**CRITICAL REMINDER**: This command creates the anti-hallucination research foundation that prevents implementation errors by ensuring all development work is backed by comprehensive, official documentation extracted via Jina's superior content processing.