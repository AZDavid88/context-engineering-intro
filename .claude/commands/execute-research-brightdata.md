# Execute Research - Brightdata + Jina Method

## Purpose
Research and extract comprehensive documentation for all technologies listed in a planning_prp.md file using Brightdata MCP for enterprise-grade content discovery and Jina AI for clean content extraction.

## Arguments
- `$ARGUMENTS`: Path to planning_prp.md file

## Execution Process

### 1. **Load PRP and Setup Environment** (MANDATORY)
You MUST:
- Read the planning_prp.md file at the specified path
- Extract ALL research_targets with URLs and purposes from the YAML frontmatter
- Derive project directory from planning_prp.md path (e.g., `/path/to/project/PRPs/planning_prp.md` → `/path/to/project/`)
- Create research directories: `{project_dir}/research/{technology}/`
- Use the configured Jina API key: `jina_0652bfc906d14590bf46815dc705aab8e7T5kQ5d6RF7vu3QK9Odfn2UjjK6`
- Use TodoWrite tool to track progress for each technology

**CHECKPOINT 1:** You MUST verify completion:
- [ ] planning_prp.md read successfully
- [ ] ALL research_targets extracted
- [ ] Project directory derived correctly  
- [ ] Research directories created
- [ ] Jina API key configured
- [ ] TodoWrite initialized with all technologies

**FAILURE TO COMPLETE ALL CHECKPOINTS = ABORT COMMAND**

### 2. **Environment Validation** (MANDATORY)
You MUST verify:
- Jina API key is available and functional
- **Brightdata MCP tools are accessible** (`mcp__brightdata__scrape_as_markdown`)
- **WebFetch tool is available** (for hybrid processing)
- Research directories are created successfully
- Exit with clear error message if any validation fails

**CHECKPOINT 2:** You MUST log status:
- [ ] Jina API key validated
- [ ] **Brightdata MCP validated** (test with simple scrape)
- [ ] **WebFetch tool validated** (test with simple URL processing)
- [ ] All research directories exist and are writable

### 3. **Primary Content Extraction** (MANDATORY)
For each research target URL, you MUST follow this EXACT sequence:

#### Step 3A: Direct Brightdata Extraction
- Use `mcp__brightdata__scrape_as_markdown` to extract clean content directly
- **DO NOT use browser navigation tools** (scraping_browser_links causes token overflow)
- Validate content quality (minimum 500 characters, contains technical information)

#### Step 3B: Quality Assessment
You MUST run this validation:
```bash
# Content quality check
content_length=$(echo "$extracted_content" | wc -c)
code_blocks=$(echo "$extracted_content" | grep -c '```' || echo 0)
technical_indicators=$(echo "$extracted_content" | grep -ci 'api\|method\|function\|class\|install' || echo 0)

if [ $content_length -lt 500 ] || [ $technical_indicators -lt 3 ]; then
  echo "❌ QUALITY FAILURE: Content too short or non-technical"
  # Trigger fallback processing
else
  echo "✅ QUALITY PASSED: Proceeding with content"
fi
```

#### Step 3C: Jina Enhancement (Fallback)
If Brightdata content needs enhancement:
- Use WebFetch with Jina processing:
  ```bash
  # Jina API call for content enhancement
  curl "https://r.jina.ai/$TARGET_URL" \
    -H "Authorization: Bearer jina_0652bfc906d14590bf46815dc705aab8e7T5kQ5d6RF7vu3QK9Odfn2UjjK6" \
    -H "Content-Type: application/json"
  ```
- Save enhanced content if quality improves

**CHECKPOINT 3:** You MUST verify each extraction:
- [ ] Content extracted using Brightdata successfully
- [ ] Quality validation passed (>500 chars, technical content)
- [ ] File saved with proper metadata
- [ ] Enhancement applied if needed

### 4. **Multi-URL Discovery** (OPTIONAL)
For GitHub repositories specifically:
- Construct common documentation paths manually:
  ```bash
  # GitHub documentation patterns
  base_url="$target_url"
  additional_urls=(
    "$base_url/tree/master/docs"
    "$base_url/tree/master/api" 
    "$base_url/tree/master/examples"
    "$base_url/blob/master/README.md"
  )
  ```
- Apply same extraction process to each discovered URL
- **Only if initial extraction suggests more documentation exists**

### 5. **Research Synthesis and Completion** (MANDATORY)
For each technology:
- Create `research/{technology}/research_summary.md` containing:
  - List of successfully extracted pages with URLs
  - Key implementation patterns discovered
  - Critical API endpoints and methods
  - Integration examples and code snippets
  - Assessment of documentation completeness
  - Quality metrics (content-to-noise ratio)
- Update planning_prp.md research_targets status from "pending" to "completed"
- Mark TodoWrite items as complete

**CHECKPOINT 4:** You MUST verify completion:
- [ ] Research summary created for each technology
- [ ] Planning PRP status updated to "completed"
- [ ] All TodoWrite items marked complete
- [ ] Minimum quality thresholds met

## Error Handling
- **Brightdata tool unavailable**: Use WebFetch with Jina as fallback
- **Content quality failures**: Retry with WebFetch + specialized Jina prompts
- **API key failures**: Display clear error message and abort
- **Network failures**: Retry once, then skip URL with logged warning

## Success Criteria
- All research_targets from planning_prp.md processed successfully
- Each technology has minimum 1 high-quality documentation file
- Research summaries created with implementation-ready content
- Superior content quality (90%+ useful content ratio with Brightdata)
- All files contain clean, LLM-optimized markdown content

## Output Structure
```
{project_dir}/research/
├── {technology1}/
│   ├── page_1_main_documentation.md    # Brightdata primary extraction
│   ├── page_2_additional_content.md    # Additional URLs if discovered
│   └── research_summary.md             # Quality metrics included
├── {technology2}/
│   └── [similar structure]
└── research_complete.md                # Final extraction report
```

## Command Completion Signal
Upon successful completion, output:
```
✅ BRIGHTDATA + JINA RESEARCH COMPLETE
- Technologies researched: {count}
- Documentation pages extracted: {count}
- Premium content quality: {percentage}% (target: 90%+)
- Total research files created: {count}
- Extraction method: Brightdata primary, Jina enhancement
- Status: Ready for /execute-prp execution
```