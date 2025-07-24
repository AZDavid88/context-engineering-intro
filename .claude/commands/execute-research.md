# Execute Research Process

## Purpose
Research and scrape documentation for all technologies listed in a planning_prp.md file using smart content extraction to avoid navigation bloat and token waste.

## Arguments
- `$ARGUMENTS`: Path to planning_prp.md file

## Execution Process

### 1. **Load Research Targets** (MANDATORY)
You MUST:
- Read the planning_prp.md file at the specified path
- Extract ALL research_targets with URLs and purposes from the YAML frontmatter
- Derive project directory from planning_prp.md path (e.g., `/path/to/project/PRPs/planning_prp.md` → `/path/to/project/`)
- Create research directories: `{project_dir}/research/{technology}/`
- Load JINA_API_KEY from project .env file or environment variables
- Use TodoWrite tool to track progress for each technology

**CHECKPOINT 1:** You MUST verify completion:
- [ ] planning_prp.md read successfully
- [ ] ALL research_targets extracted
- [ ] Project directory derived correctly  
- [ ] Research directories created
- [ ] JINA_API_KEY loaded and verified
- [ ] TodoWrite initialized with all technologies

**FAILURE TO COMPLETE ALL CHECKPOINTS = ABORT COMMAND**

### 2. **Environment Validation** (MANDATORY)
You MUST verify:
- JINA_API_KEY is available (check env, then .env file in project directory)
- All required MCP Playwright tools are accessible
- **Brightdata MCP tools are available** (`mcp__brightdata__scrape_as_markdown`)
- **WebFetch tool is available** (for hybrid Brightdata+Jina processing)
- Research directories are created successfully
- Exit with clear error message if any validation fails

**CHECKPOINT 2:** You MUST log status:
- [ ] JINA_API_KEY validated (show character count)
- [ ] MCP Playwright tools tested with simple navigation
- [ ] **Brightdata MCP validated** (test with simple scrape)
- [ ] **WebFetch tool validated** (test with simple URL processing)
- [ ] All research directories exist and are writable

### 3. **Smart Content Assessment** (MANDATORY)
For each research target URL, you MUST follow this EXACT sequence:

#### Step 3A: Navigate and Assess
- Use `mcp__playwright__browser_navigate` to load the page
- Use `mcp__playwright__browser_snapshot` to capture page structure
- Identify main content selectors: `article`, `main`, `.content`, `#content`, `[role="main"]`

#### Step 3B: Navigation Quality Check (MANDATORY)
You MUST run this exact validation:
```bash
# Count navigation elements vs content  
nav_links=$(echo "$snapshot_content" | grep -c '\[.*\](.*http' || echo 0)
content_indicators=$(echo "$snapshot_content" | grep -c 'code\|example\|api\|method\|tutorial' || echo 0)
nav_ratio=$((nav_links * 100 / (nav_links + content_indicators + 1)))

if [ $nav_ratio -gt 50 ]; then
  echo "❌ REJECTED: $nav_ratio% navigation waste - SKIPPING URL"
  continue
else
  echo "✅ APPROVED: $nav_ratio% navigation, proceeding to extract"
fi
```

#### Step 3C: Priority Ordering
- Create scraping priority list based on content density assessment
- Focus on pages with highest code example / API reference ratio

**CHECKPOINT 3:** You MUST verify before proceeding:
- [ ] All URLs assessed using Playwright navigation
- [ ] Navigation ratios calculated for each page  
- [ ] Pages >50% navigation marked as REJECTED
- [ ] Priority scraping order established
- [ ] At least 2 high-quality pages identified per technology

### 4. **Efficient Content Extraction** (MANDATORY)

#### Step 4A: Content-First Extraction  
For each APPROVED URL, you MUST follow this sequence:
1. Use Playwright to identify main content area using selectors
2. Extract ONLY the main content area (not full page HTML)
3. Remove navigation, sidebars, headers, footers from extracted content
4. THEN send cleaned content to Jina API

#### Step 4B: Mandatory Quality Pre-Check
Before saving ANY file, you MUST run this validation:
```bash
# Auto-quality validation script (COPY EXACTLY)
validate_content() {
  local file="$1"
  local lines=$(wc -l < "$file")
  local nav_links=$(grep -c '\[.*\](.*http' "$file" 2>/dev/null || echo 0)
  local code_blocks=$(grep -c '```' "$file" 2>/dev/null || echo 0)
  local api_refs=$(grep -ci 'api\|method\|endpoint\|client' "$file" 2>/dev/null || echo 0)
  
  local nav_percent=$((nav_links * 100 / lines))
  local content_score=$((code_blocks * 2 + api_refs))
  
  echo "Quality check: $nav_percent% nav, $content_score content score"
  
  # HARD REJECTION CRITERIA
  if [ $nav_percent -gt 30 ]; then
    echo "❌ FAILED: >30% navigation ($nav_percent%)"  
    rm "$file"
    return 1
  elif [ $content_score -lt 3 ]; then
    echo "❌ FAILED: Low content score ($content_score)"
    rm "$file" 
    return 1
  else
    echo "✅ PASSED: Quality approved"
    return 0
  fi
}

# Run validation after each file creation
validate_content "research/{technology}/page{N}_{name}.md" || continue
```

#### Step 4C: Smart Jina Integration
- Use Jina API with cleaned content: `curl "https://r.jina.ai/" -H "Authorization: Bearer $JINA_API_KEY" -H "Content-Type: application/json" -d '{"url": "URL_HERE"}'`
- Save as: `research/{technology}/page{N}_{descriptive_name}.md`
- Include metadata header: URL, purpose, extraction method, content quality score

#### Step 4D: Brightdata + Jina Hybrid Strategy (PREMIUM OPTION)
**TRIGGER CONDITIONS:** Use Brightdata+Jina hybrid when:
- Maximum quality extraction needed (>90% useful content)
- Playwright tools are unstable or slow
- Website has heavy bot protection that blocks Playwright
- Research requires bulletproof reliability

**HYBRID PROCESSING WORKFLOW:**
1. **Raw Content Extraction:** Use `mcp__brightdata__scrape_as_markdown` to capture ALL page content
2. **Intelligent Filtering:** Process raw content through WebFetch with specialized prompt:
   ```
   Extract comprehensive [TECHNOLOGY] documentation including:
   1. Installation commands and dependencies
   2. Import statements and initialization code  
   3. API methods and usage patterns
   4. Configuration and integration examples
   5. Code blocks and implementation workflows
   
   Focus on implementation-ready patterns. Exclude ALL navigation menus, 
   headers, footers, sidebars, and promotional content.
   Return as clean markdown with properly formatted code blocks.
   ```
3. **Quality Validation:** Expect 90-95% useful content (premium threshold)
4. **Metadata Marking:** Mark files as `"extraction_method": "Brightdata+Jina Hybrid (Premium)"`

**HYBRID QUALITY EXPECTATIONS:**
- **Target Quality:** 90-95% useful content (superior to all other methods)
- **Navigation Waste:** 0-5% (Jina eliminates navigation pollution)
- **Token Efficiency:** ~95% useful content ratio
- **Implementation Ready:** Perfect copy/paste code patterns
- **Success Rate:** 90-95% for premium extraction

#### Step 4E: Brightdata MCP Solo Strategy (FALLBACK PROTOCOL)  
**TRIGGER CONDITIONS:** Use Brightdata solo when:
- Playwright tools fail or become unavailable
- JINA_API_KEY authentication fails or WebFetch unavailable
- Network issues prevent hybrid processing
- Emergency backup needed

**BRIGHTDATA SOLO PROCESS:**
1. **Direct Extraction:** Use `mcp__brightdata__scrape_as_markdown` for the same URL
2. **Quality Assessment:** Apply relaxed quality gates (≤40% nav, ≥2 code blocks)
3. **Multiple Attempts:** If first extraction has >60% navigation, retry 2 more times
4. **Manual Filtering:** Remove obvious navigation patterns:
   ```bash
   # Remove navigation pollution patterns
   sed -i '/^\* \[.*\]/d' "$file"  # Remove bullet-point navigation links
   sed -i '/^### Getting Started$/,/^###/d' "$file"  # Remove navigation sections
   sed -i '/^### User Manual$/,/^###/d' "$file"  # Remove sidebar content
   ```
5. **Success Criteria:** Accept if content quality ≥60% (emergency threshold)
6. **Metadata Marking:** Mark files as `"extraction_method": "Brightdata Solo (Emergency)"`

**BRIGHTDATA SOLO EXPECTATIONS:**
- **Target Quality:** 60-80% useful content (acceptable for backup)
- **Acceptable Navigation:** ≤40% (relaxed standards for emergency use)
- **Retry Logic:** Up to 3 attempts per URL if quality insufficient
- **Success Rate:** 70-80% success rate for emergency extraction

**CHECKPOINT 4:** You MUST verify each file:
- [ ] Content extracted from main area only (not full page)
- [ ] Navigation content removed before Jina processing
- [ ] Quality validation passed (≤30% nav, ≥3 content score)
- [ ] File saved with descriptive name and metadata

### 5. **Automated Quality Enforcement** (MANDATORY)

#### Step 5A: Real-Time Quality Gates
After EVERY file creation, you MUST automatically run:
```bash
# Immediate post-creation validation
for file in research/{technology}/page*.md; do
  if [ -f "$file" ]; then
    lines=$(wc -l < "$file")
    nav_links=$(grep -c '\[.*\](.*http' "$file" 2>/dev/null || echo 0)
    code_blocks=$(grep -c '```' "$file" 2>/dev/null || echo 0)
    
    nav_percent=$((nav_links * 100 / lines))
    
    if [ $nav_percent -gt 30 ] || [ $code_blocks -lt 2 ]; then
      echo "🗑️ AUTO-DELETING: $file (quality failure)"
      rm "$file"
    else
      echo "✅ APPROVED: $file (quality passed)"
    fi
  fi
done
```

#### Step 5B: Progressive Failure Protocol
Track failures per technology:
- **1st quality failure:** Warning + retry with different content selector
- **2nd quality failure:** Skip current URL, try next priority URL  
- **3rd quality failure:** Mark technology as "INCOMPLETE" and move to next technology
- **4th quality failure:** ABORT entire research command with error report

**CHECKPOINT 5:** You MUST log quality metrics:
- [ ] All saved files passed automated quality checks
- [ ] Navigation waste <30% for all approved files
- [ ] Minimum content thresholds met
- [ ] Failure count tracked per technology

### 6. **Progress Tracking** (MANDATORY)
You MUST use TodoWrite to:
- Mark each technology research as "in_progress" when starting
- Update progress as pages are successfully scraped and validated
- Mark as "completed" only when quality thresholds met AND minimum file count achieved
- Track failed URLs with specific error reasons and quality scores

### 7. **Research Completion** (MANDATORY)
For each technology, you MUST:
- Create `research/{technology}/research_summary.md` with:
  - List of successfully scraped pages with quality scores
  - Key implementation patterns found
  - Critical API endpoints/methods discovered  
  - Integration examples extracted
  - Token efficiency metrics
- Update planning_prp.md research_targets status from "pending" to "completed"
- Generate efficiency report: useful content tokens vs total tokens consumed

### 8. **Final Validation** (MANDATORY)
Before completing, you MUST verify:
- All research_targets from planning_prp.md have corresponding research directories
- Each technology directory contains ≥2 substantive documentation files
- All files pass quality validation (≤30% navigation, ≥3 content score)
- All research summaries created with quality metrics
- No unresolved authentication/API errors
- Token efficiency report shows >60% useful content extraction

## Enforcement Mechanisms

### Hard Quality Gates (NO EXCEPTIONS)
**IMMEDIATE FILE DELETION if ANY of these criteria are met:**
- >30% navigation links (`[.*](.*http` pattern count)
- <2 code blocks AND <5 API references
- <50 lines of content after header removal
- File primarily contains sidebar/menu/footer content

### Execution Logging (MANDATORY)
You MUST log each step with timestamp and status:
```
[2025-07-24 15:30:15] ✅ CHECKPOINT 1: Environment validation passed
[2025-07-24 15:31:22] ✅ CHECKPOINT 2: Content assessment completed  
[2025-07-24 15:32:45] ❌ QUALITY FAILURE: page1_overview.md (45% navigation) - DELETED
[2025-07-24 15:33:12] ✅ CHECKPOINT 3: page2_tutorial.md quality approved
```

### Command Abort Conditions
**IMMEDIATELY STOP EXECUTION if:**
- 3 consecutive quality failures for single technology
- JINA_API_KEY authentication fails
- MCP Playwright tools become unavailable
- Unable to create research directories
- Any CHECKPOINT fails validation

## Quality Gates (All Must Pass)
- [ ] All research_targets extracted from planning_prp.md
- [ ] JINA_API_KEY validated and working
- [ ] Research directories created for all technologies  
- [ ] Each technology has ≥2 quality documentation files (≤30% nav, ≥2 code blocks)
- [ ] No files with >30% navigation content remain
- [ ] No unresolved authentication/API errors
- [ ] Research summaries created for each technology with quality metrics
- [ ] Planning_prp.md status fields updated to "completed"
- [ ] Token usage efficiency >60%
- [ ] All checkpoints logged with timestamps

## Error Handling (MANDATORY Actions)
- **JINA_API_KEY missing**: Display error with link to https://jina.ai/?sui=apikey and ABORT
- **Quality validation fails**: Auto-delete file, log failure, try alternative URL
- **3 consecutive failures**: Skip technology, mark as "INCOMPLETE", continue
- **Playwright tools fail**: **TRY BRIGHTDATA+JINA HYBRID, then Brightdata solo**
- **Navigation >50%**: Skip URL immediately, mark as "REJECTED - Navigation heavy"
- **Hybrid method fails**: Fall back to Brightdata solo (emergency mode)

### Extraction Method Priority (Cascading Fallback)
1. **Primary:** Playwright + Jina (85% quality baseline)
2. **Premium Option:** Brightdata + Jina Hybrid (90-95% quality when needed)
3. **Backup:** Brightdata Solo (60-80% quality for emergencies)

### Method-Specific Error Handling
**Brightdata+Jina Hybrid Errors:**
- **Brightdata fails**: Fall back to Playwright+Jina or Brightdata solo
- **WebFetch/Jina fails**: Use raw Brightdata content with manual filtering
- **Quality insufficient**: Retry with different extraction prompts

**Brightdata Solo Errors:**
- **Tool unavailable**: Log warning, continue with available methods only
- **Multiple retries fail**: Mark URL as "BRIGHTDATA_FAILED", try next priority URL
- **Quality insufficient**: Accept lower quality (≥60%) with warning in research summary
- **Network/service errors**: Wait 30 seconds, retry once, then skip URL

## Success Criteria
- All planning_prp.md research_targets marked "completed" 
- Research summary created for each technology with implementation-ready content
- Token efficiency report showing >60% useful content extraction
- All files pass automated quality validation
- Ready for `/execute-prp` command execution
- TodoWrite shows all research tasks completed
- Quality enforcement log shows no unresolved failures

## Output Structure
```
{project_dir}/research/
├── {technology1}/
│   ├── page1_getting_started.md      # ≤30% nav, ≥2 code blocks
│   ├── page2_api_reference.md        # ≤30% nav, ≥5 API refs  
│   ├── page3_examples.md             # ≤30% nav, ≥3 examples
│   └── research_summary.md           # includes quality metrics
├── {technology2}/
│   └── [similar structure]
├── research_complete.md               # final efficiency report
└── quality_enforcement.log           # checkpoint and validation log
```

## Command Completion Signal
Upon successful completion, you MUST output:
```
✅ RESEARCH EXECUTION COMPLETE
- Technologies researched: {count}
- Total pages scraped: {count}  
- Quality failures handled: {count}
- Navigation waste eliminated: {percentage}%
- Token efficiency: {percentage}%
- Status: Ready for /execute-prp
```