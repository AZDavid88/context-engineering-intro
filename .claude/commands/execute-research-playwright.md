# Execute Research - Playwright + Jina Method

## Purpose
Research and extract comprehensive documentation for all technologies listed in a planning_prp.md file using Playwright MCP for dynamic content discovery and Jina AI for clean content extraction.

## Arguments
- `$ARGUMENTS`: Path to planning_prp.md file

## Execution Process

### 1. **Load PRP and Setup Environment**
- Read the planning_prp.md file at the specified path
- Extract ALL research_targets with URLs and purposes from the YAML frontmatter
- Derive project directory from planning_prp.md path (e.g., `/path/to/project/PRPs/planning_prp.md` → `/path/to/project/`)
- Create research directories: `{project_dir}/research/{technology}/`
- Use the configured Jina API key: `jina_0652bfc906d14590bf46815dc705aab8e7T5kQ5d6RF7vu3QK9Odfn2UjjK6`
- Use TodoWrite tool to track progress for each technology

### 2. **Documentation Discovery Process**
For each research target URL:
- Use `mcp__playwright__browser_navigate` to load the main documentation page
- Use `mcp__playwright__browser_snapshot` to capture page structure and assess content quality
- Execute JavaScript to discover all relevant documentation links:
  ```javascript
  () => {
    return Array.from(document.querySelectorAll('a[href]'))
      .map(link => ({
        text: link.textContent.trim(),
        url: new URL(link.href, window.location.href).href,
        isDocLink: /docs|guide|tutorial|api|reference|getting-started|examples/i.test(link.href)
      }))
      .filter(link => link.isDocLink && link.text.length > 5)
      .slice(0, 30); // Limit to prevent overwhelming
  }
  ```
- Filter discovered URLs for relevance to the technology being researched
- Prioritize URLs containing: API references, getting started guides, tutorials, examples

### 3. **Content Extraction and Processing**
For each discovered relevant URL:
- Navigate to the URL using `mcp__playwright__browser_navigate`
- Verify page loaded successfully with `mcp__playwright__browser_snapshot`
- Extract the clean URL and pass to Jina Reader API using the simpler GET method:
  ```bash
  curl "https://r.jina.ai/TARGET_URL" \
    -H "Authorization: Bearer jina_0652bfc906d14590bf46815dc705aab8e7T5kQ5d6RF7vu3QK9Odfn2UjjK6"
  ```
- Validate content quality (minimum 500 characters, contains code examples or technical information)
- Save successful extractions as: `research/{technology}/page_{counter}_{descriptive_name}.md`
- Include metadata header with URL, extraction method, and quality indicators

### 4. **Quality Control and Validation**
- Ensure each technology has at least 3 high-quality documentation files
- Verify extracted content contains implementation-ready information
- Remove any files that are primarily navigation or promotional content
- Cross-reference content to ensure comprehensive coverage of the technology

### 5. **Research Synthesis and Completion**
For each technology:
- Create `research/{technology}/research_summary.md` containing:
  - List of successfully extracted pages with URLs
  - Key implementation patterns discovered
  - Critical API endpoints and methods
  - Integration examples and code snippets
  - Assessment of documentation completeness
- Update planning_prp.md research_targets status from "pending" to "completed"
- Generate final report with extraction metrics and quality assessment

## Quality Standards
- Each extracted file must contain substantial technical content (>500 characters)
- Priority given to pages with code examples, API documentation, or implementation guides
- Navigation-heavy pages (>40% links) should be filtered out
- Each technology should have comprehensive coverage of core concepts

## Error Handling
- **Playwright navigation failures**: Retry once, then skip URL with logged warning
- **Jina API failures**: Retry with exponential backoff, fall back to basic content extraction
- **Content quality failures**: Skip page and try next priority URL

## Success Criteria
- All research_targets from planning_prp.md processed successfully
- Each technology has minimum 3 comprehensive documentation files
- Research summaries created with implementation-ready content
- All files contain clean, LLM-optimized markdown content
- Ready for subsequent `/execute-prp` command execution

## Output Structure
```
{project_dir}/research/
├── {technology1}/
│   ├── page_1_getting_started.md
│   ├── page_2_api_reference.md  
│   ├── page_3_examples.md
│   └── research_summary.md
├── {technology2}/
│   └── [similar structure]
└── research_complete.md
```

## Command Completion Signal
Upon successful completion, output:
```
✅ PLAYWRIGHT + JINA RESEARCH COMPLETE
- Technologies researched: {count}
- Documentation pages extracted: {count}
- Average content quality: {percentage}%
- Total research files created: {count}
- Status: Ready for /execute-prp execution
```