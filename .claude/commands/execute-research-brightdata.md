# Execute Research - Brightdata + Jina Method

## Purpose
Research and extract comprehensive documentation for all technologies listed in a planning_prp.md file using Brightdata MCP for enterprise-grade content discovery and Jina AI for clean content extraction.

## Arguments
- `$ARGUMENTS`: Path to planning_prp.md file

## Execution Process

### 1. **Load PRP and Setup Environment**
- Read the planning_prp.md file at the specified path
- Extract ALL research_targets with URLs and purposes from the YAML frontmatter
- Derive project directory from planning_prp.md path (e.g., `/path/to/project/PRPs/planning_prp.md` → `/path/to/project/`)
- Create research directories: `{project_dir}/research/{technology}/`
- Use the configured Jina API key: `jina_0652bfc906d14590bf46815dc705aab8e7T5kQ5d6RF7vu3QK9Odfn2UjjK6`
- Verify Brightdata MCP connection and API token
- Use TodoWrite tool to track progress for each technology

### 2. **Documentation Discovery Process**
For each research target URL:
- Use `mcp__brightdata__scraping_browser_navigate` to load the main documentation page
- Use `mcp__brightdata__scraping_browser_links` to discover all links on the current page
- Filter discovered links for documentation relevance:
  ```javascript
  // Filter for documentation-specific patterns
  const relevantLinks = allLinks.filter(link => 
    /docs|guide|tutorial|api|reference|getting-started|examples|concepts/i.test(link.url) &&
    !link.url.includes('#') && // Skip anchor links
    link.text.length > 5 && // Skip empty links
    !link.url.includes('login') && // Skip auth pages
    !link.url.includes('pricing') // Skip commercial pages
  ).slice(0, 25); // Limit to prevent overwhelming
  ```
- Prioritize URLs containing: API references, getting started guides, tutorials, examples
- Create comprehensive URL list for each technology

### 3. **Content Extraction and Processing**
For each discovered relevant URL:
- Use `mcp__brightdata__scrape_as_markdown` to extract clean content directly
- If Brightdata content needs enhancement, pass to Jina Reader API using the simpler GET method:
  ```bash
  curl "https://r.jina.ai/TARGET_URL" \
    -H "Authorization: Bearer jina_0652bfc906d14590bf46815dc705aab8e7T5kQ5d6RF7vu3QK9Odfn2UjjK6"
  ```
- Validate content quality (minimum 500 characters, contains code examples or technical information)
- Save successful extractions as: `research/{technology}/page_{counter}_{descriptive_name}.md`
- Include metadata header with URL, extraction method ("Brightdata+Jina"), and quality indicators

### 4. **Advanced Content Enhancement**
For high-priority documentation:
- Use `mcp__brightdata__extract` with custom prompts for structured data extraction:
  ```json
  {
    "url": "TARGET_URL",
    "extraction_prompt": "Extract comprehensive [TECHNOLOGY] documentation including: 1. Installation commands and dependencies, 2. Import statements and initialization code, 3. API methods and usage patterns, 4. Configuration and integration examples, 5. Code blocks and implementation workflows. Focus on implementation-ready patterns. Return as clean markdown with properly formatted code blocks."
  }
  ```
- Process extracted structured data for immediate implementation use
- Combine multiple extraction methods for maximum content quality

### 5. **Quality Control and Validation**
- Ensure each technology has at least 3 high-quality documentation files
- Verify extracted content contains implementation-ready information
- Leverage Brightdata's superior content quality (90-95% useful content ratio)
- Cross-reference content to ensure comprehensive coverage of the technology
- Remove any files that are primarily navigation or promotional content

### 6. **Research Synthesis and Completion**
For each technology:
- Create `research/{technology}/research_summary.md` containing:
  - List of successfully extracted pages with URLs
  - Key implementation patterns discovered
  - Critical API endpoints and methods
  - Integration examples and code snippets
  - Assessment of documentation completeness
  - Quality metrics (content-to-noise ratio)
- Update planning_prp.md research_targets status from "pending" to "completed"
- Generate final report with extraction metrics and quality assessment

## Quality Standards
- Each extracted file must contain substantial technical content (>500 characters)
- Priority given to pages with code examples, API documentation, or implementation guides
- Target 90-95% useful content ratio (Brightdata premium quality threshold)
- Navigation waste should be <5% due to Brightdata's advanced filtering
- Each technology should have comprehensive coverage of core concepts

## Error Handling
- **Brightdata navigation failures**: Retry with different proxy, then skip URL with logged warning
- **Bot detection encountered**: Automatic handling through Brightdata infrastructure
- **Jina API failures**: Fall back to pure Brightdata content extraction
- **Content quality failures**: Retry extraction with different parameters, then skip

## Advanced Features
- **Geo-unblocking**: Access region-restricted documentation automatically
- **Bot detection evasion**: Handle protected documentation sites seamlessly  
- **Dynamic content rendering**: Extract from JavaScript-heavy documentation sites
- **Enterprise reliability**: Higher success rate for complex or protected sites

## Success Criteria
- All research_targets from planning_prp.md processed successfully
- Each technology has minimum 3 comprehensive documentation files
- Research summaries created with implementation-ready content
- Superior content quality (90-95% useful content ratio)
- All files contain clean, LLM-optimized markdown content
- Ready for subsequent `/execute-prp` command execution

## Output Structure
```
{project_dir}/research/
├── {technology1}/
│   ├── page_1_getting_started.md      # Premium quality extraction
│   ├── page_2_api_reference.md        # Structured data extraction
│   ├── page_3_examples.md             # Implementation-ready content
│   └── research_summary.md            # Quality metrics included
├── {technology2}/
│   └── [similar structure]
└── research_complete.md               # Enterprise extraction report
```

## Command Completion Signal
Upon successful completion, output:
```
✅ BRIGHTDATA + JINA RESEARCH COMPLETE
- Technologies researched: {count}
- Documentation pages extracted: {count}
- Premium content quality: {percentage}% (target: 90-95%)
- Total research files created: {count}
- Bot protection bypassed: {count} sites
- Status: Ready for /execute-prp execution
```