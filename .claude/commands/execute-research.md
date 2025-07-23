# Execute Research Process

## Purpose
To systematically research and gather comprehensive documentation for all technologies specified in a `planning_prp.md` file using the combined power of Playwright MCP (for navigation and visual content) and Jina AI (for content extraction and processing). This command transforms a research mandate into a complete knowledge base ready for implementation.

## Core Principles
1. **Strategic Navigation:** Use the provided starting URLs as navigation hubs, not just single pages to scrape
2. **Token Efficiency:** Map out all required pages BEFORE scraping to avoid wasted API calls
3. **Comprehensive Coverage:** Gather both textual content (Jina) and visual/interactive context (Playwright)
4. **Systematic Organization:** Store each page as separate .md files in organized directory structure
5. **Quality Validation:** Retry failed scrapes and validate comprehensive coverage
6. **Parallel Optimization:** Automatically assess research scope and offer parallel sub-agent deployment for large projects

---

## Command Flow: `/execute-research`

### Prerequisites
- A valid `planning_prp.md` file exists with populated `research_targets` section
- `JINA_API_KEY` environment variable is set
- Claude Code environment with MCP Playwright tools enabled (mcp__playwright__* functions)
- Bash tool access for Jina API curl commands

### Execution Process

#### Phase 1: Planning & Setup
1. **Load Research Mandate**
   - Read the `planning_prp.md` file
   - Extract all `research_targets` with their URLs and purposes
   - **Derive project context**: Extract project directory path from the planning_prp.md location
     - Example: `/path/to/project/PRPs/planning_prp.md` → Project dir: `/path/to/project/`
   - Create directory structure relative to project: `[project_dir]/research/[technology]/`
   - Initialize progress tracking

2. **Environment Validation**
   - **Smart API Key Discovery**:
     1. Check if `JINA_API_KEY` is already in environment
     2. If not, look for `.env` file in the same directory as the planning_prp.md
     3. If not found, check parent directories up to project root
     4. Load the `.env` file using python-dotenv or source command
   - If no API key found anywhere, display error: "JINA_API_KEY not found. Please either:
     - Set it as environment variable: `export JINA_API_KEY='your_key'`
     - Create a `.env` file in your project directory with: `JINA_API_KEY='your_key'`
     - Get your free API key at: https://jina.ai/?sui=apikey"
   - Test Playwright MCP connectivity
   - Create all required output directories

#### Phase 2: Strategic Reconnaissance (Playwright-Led)
For each technology in `research_targets`:

3. **Navigation Hub Discovery**
   - Use `mcp__playwright__browser_navigate` to navigate to the primary documentation URL
   - Use `mcp__playwright__browser_take_screenshot` for visual context
   - Use `mcp__playwright__browser_snapshot` to extract ALL relevant documentation links from:
     - Navigation menus
     - Breadcrumbs  
     - Footer links
     - In-content "Related" or "Next Steps" sections
     - Table of contents
   - Map out the complete documentation structure from snapshot data
   - Prioritize pages based on the `purpose` field from planning_prp.md

4. **Page Validation**
   - Use `mcp__playwright__browser_navigate` to verify each discovered URL is accessible
   - Identify pages most relevant to the project's specific needs using snapshot content analysis
   - Flag any 404s or problematic pages for retry
   - Create comprehensive page list with URLs and relevance scores for systematic scraping

#### Phase 3: Content Harvesting (Jina-Led)
5. **Research Execution Strategy Assessment**
   - Analyze the total research scope (number of technologies × pages per technology)
   - **PARALLELIZATION CHECKPOINT:** If research scope is substantial (>20 pages total), pause and inform the user:
     *"I've mapped out [N] technologies with approximately [M] pages to research. This is a substantial research operation that would benefit from parallel processing. You can speed this up significantly by typing: 'Please spin up multiple subagents to speed up research' - this will allow me to deploy specialized research agents for each technology simultaneously."*
   - Wait for user decision on parallelization strategy
   - If user requests sub-agents, deploy multiple Task agents with specific technology assignments

6. **Smart Content Extraction** (Sequential or Parallel)
   - **Context-Aware Page Assessment**: For each page, use MCP Playwright tools to evaluate:
     - Use `mcp__playwright__browser_navigate` to load the page
     - Use `mcp__playwright__browser_snapshot` to analyze content structure
     - Assess content density vs navigation bloat ratio from snapshot YAML
     - Identify presence of code examples, API references, tutorials in snapshot
     - Compare with already scraped content to detect redundancy
     - Score project-specific relevance based on planning_prp.md purpose field
   
   - **Intelligent Filtering Strategy**: Apply context-preserving filters based on assessment:
     ```bash
     # AGGRESSIVE (navigation-heavy pages):
     # Note: $JINA_API_KEY is loaded from project .env file
     curl -X POST "https://r.jina.ai/" \
       -H "Authorization: Bearer $JINA_API_KEY" \
       -H "Content-Type: application/json" \
       -H "Accept: application/json" \
       -H "X-Remove-Selector: .ads,.cookie-banner,.newsletter-signup,footer.site-footer,nav.sidebar" \
       -H "X-Target-Selector: main,article,.documentation,.content,.tutorial" \
       -H "X-With-Links-Summary: true" \
       -H "X-Token-Budget: 3000" \
       -H "X-Return-Format: markdown" \
       -H "X-Timeout: 20" \
       -d '{"url":"[PAGE_URL]"}'
     
     # BALANCED (content-rich pages):
     curl -X POST "https://r.jina.ai/" \
       -H "Authorization: Bearer $JINA_API_KEY" \
       -H "Content-Type: application/json" \
       -H "Accept: application/json" \
       -H "X-Remove-Selector: .ads,.cookie-banner,.newsletter-signup" \
       -H "X-With-Links-Summary: true" \
       -H "X-With-Generated-Alt: true" \
       -H "X-Token-Budget: 4000" \
       -H "X-Return-Format: markdown" \
       -H "X-Timeout: 20" \
       -d '{"url":"[PAGE_URL]"}'
       
     # MINIMAL (essential/unique pages):
     curl -X POST "https://r.jina.ai/" \
       -H "Authorization: Bearer $JINA_API_KEY" \
       -H "Content-Type: application/json" \
       -H "Accept: application/json" \
       -H "X-With-Links-Summary: true" \
       -H "X-With-Generated-Alt: true" \
       -H "X-With-Iframe: true" \
       -H "X-Token-Budget: 5000" \
       -H "X-Return-Format: markdown" \
       -H "X-Timeout: 20" \
       -d '{"url":"[PAGE_URL]"}'
     ```
   
   - **Project-Specific Prioritization**: Based on planning_prp.md purpose, ensure critical sections get full token allocation:
     - Chat interfaces: `.chat-example`, `.state-management`, `.event-handlers`, `.websocket`
     - Database integration: `.database`, `.orm`, `.query`, `.connection`, `.async-db`
     - API integration: `.api`, `.authentication`, `.rate-limiting`, `.error-handling`
   
   - **Parallel Deployment**: If using sub-agents, provide context-aware instructions:
     ```
     Research [TECHNOLOGY_NAME] with intelligent token efficiency. Use MCP Playwright tools (mcp__playwright__browser_navigate, mcp__playwright__browser_snapshot) to assess each page's content density before scraping. Apply AGGRESSIVE filtering to navigation-heavy pages, BALANCED filtering to content-rich pages, and MINIMAL filtering to essential pages. Focus on [PROJECT_PURPOSE] related content. Use Bash tool with appropriate Jina curl format based on page assessment. Store results in /research/[technology]/pageN_[name].md format.
     ```
   
   - Store each page as: `/research/[technology]/page[N]_[descriptive_name].md`
   - Include metadata header with URL, purpose, filtering strategy used, and token count
   - Retry any failed scrapes with progressively less aggressive filtering

7. **Context Preservation Validation**
   - **Content Completeness Check**: Analyze scraped content for project-critical elements:
     - Code examples for implementation
     - API endpoints and parameters
     - Configuration and setup instructions
     - Error handling and troubleshooting info
     - Integration patterns specific to project purpose
   
   - **Context Gap Detection**: Compare scraped content against project requirements:
     ```python
     missing_concepts = detect_missing_implementation_details(
         scraped_content, 
         project_purpose_from_planning_prp
     )
     ```
   
   - **Rescue Scraping**: If critical context is missing, re-scrape specific pages with less aggressive filtering:
     - Escalate from AGGRESSIVE → BALANCED → MINIMAL filtering
     - Use targeted selectors for missing concept areas
     - Append additional context to existing files rather than full re-scrape
   
   - **Token Efficiency Report**: Track token usage vs context quality:
     - Total tokens used per technology
     - Context completeness percentage
     - Token savings compared to naive full-page scraping
   
   - **Parallel Coordination:** If using sub-agents, consolidate and cross-validate all findings before proceeding

#### Phase 4: Visual & Interactive Enhancement (MCP Playwright-Led)
8. **Visual Context Capture**
   - Use `mcp__playwright__browser_take_screenshot` for key UI components, demos, and interactive examples
   - Use `mcp__playwright__browser_console_messages` to capture browser console logs for debugging insights
   - Document any interactive elements that Jina couldn't capture using snapshot analysis
   - Store screenshots in `/research/[technology]/screenshots/`

9. **Interactive Content Documentation**
   - Use `mcp__playwright__browser_click`, `mcp__playwright__browser_type` to test live demos and interactive tutorials
   - Document form interactions, user flows, and dynamic content using MCP tools
   - Capture any JavaScript-heavy content that requires browser rendering with snapshots

#### Phase 5: Synthesis & Organization
10. **Knowledge Base Assembly**
   - Create comprehensive `research_summary.md` for each technology
   - Cross-reference findings with original project requirements
   - Identify implementation patterns and best practices
   - Flag any gaps or additional research needed

11. **Research Validation**
    - Verify all `research_targets` from planning_prp.md have been addressed
    - Ensure sufficient depth for implementation phase
    - Update the planning_prp.md status fields to "completed"
    - **Parallel Summary:** If sub-agents were used, compile and cross-reference all findings

## Output Structure
```
/research/
├── [technology1]/
│   ├── page1_introduction.md
│   ├── page2_getting_started.md
│   ├── page3_advanced_features.md
│   ├── screenshots/
│   │   ├── main_interface.png
│   │   └── demo_example.png
│   └── research_summary.md
├── [technology2]/
│   └── [similar structure]
└── research_complete.md (final report)
```

## Error Handling
- **404 Errors:** Retry with alternative URLs or skip with documentation
- **Jina API Failures:** Retry with different headers/timeout, fall back to Playwright text extraction
- **Rate Limiting:** Implement exponential backoff
- **Invalid Content:** Retry scrape or supplement with Playwright

## Success Criteria
- [ ] All research_targets from planning_prp.md have corresponding research directories
- [ ] Each technology has comprehensive coverage with intelligent token usage (typically 3000-6000 tokens per technology vs 15000+ with naive scraping)
- [ ] Context completeness >90% for project-critical concepts
- [ ] Visual context captured for complex concepts
- [ ] research_summary.md created for each technology with token efficiency metrics
- [ ] No 404s or failed scrapes remaining
- [ ] All findings directly applicable to project requirements
- [ ] Token usage optimized (target: 60-80% reduction vs full-page scraping)

## Quality Checklist
- [ ] Did AI navigate from provided URLs rather than improvising new starting points?
- [ ] Are all pages stored in organized `/research/[technology]/pageN.md` format?
- [ ] Were failed scrapes retried with progressively less aggressive filtering?
- [ ] Does each technology have sufficient depth for implementation while optimizing token usage?
- [ ] Are visual/interactive elements documented where Jina couldn't capture them?
- [ ] If research scope was substantial, was parallel processing offered to the user?
- [ ] If sub-agents were deployed, were all findings properly consolidated?
- [ ] Was content assessed for density vs bloat before applying filtering strategies?
- [ ] Were project-specific priorities (from planning_prp.md purpose) preserved in filtering decisions?
- [ ] Was context validation performed to ensure no critical implementation details were lost?
- [ ] Does the token efficiency report show meaningful savings without context compromise?

## Usage
```
/execute-research [path_to_planning_prp.md]
```

Example:
```
/execute-research projects/narrative-factory/PRPs/planning_prp.md
```

## Concrete Execution Example

**Step-by-step for Reflex.dev research:**

1. **Load Environment & Read planning_prp.md**:
   ```bash
   # Derive project directory from planning_prp.md path
   PROJECT_DIR="/workspaces/context-engineering-intro/projects/narrative-factory"
   
   # Check for .env file and load it
   if [ -f "$PROJECT_DIR/.env" ]; then
       export $(cat "$PROJECT_DIR/.env" | xargs)
       echo "Loaded JINA_API_KEY from $PROJECT_DIR/.env"
   fi
   ```
   → Extract: URL=`https://reflex.dev/docs/getting-started/introduction/`, Purpose="chat UI in pure Python"

2. **MCP Playwright Navigation:**
   ```
   mcp__playwright__browser_navigate → https://reflex.dev/docs/getting-started/introduction/
   mcp__playwright__browser_take_screenshot → reflex_intro.png  
   mcp__playwright__browser_snapshot → Extract all documentation links from YAML
   ```

3. **Content Assessment:** Analyze snapshot YAML to identify:
   - `/docs/getting-started/chatapp-tutorial/` → HIGH relevance (chat purpose)
   - `/docs/state/overview/` → HIGH relevance (state management)  
   - `/docs/library/` → MEDIUM relevance (components)

4. **Smart Jina Scraping:**
   ```bash
   # Chatapp tutorial (essential) → MINIMAL filtering
   curl -X POST "https://r.jina.ai/" -H "Authorization: Bearer $JINA_API_KEY" \
     -H "X-With-Links-Summary: true" -H "X-Token-Budget: 5000" \
     -d '{"url":"https://reflex.dev/docs/getting-started/chatapp-tutorial/"}'
   
   # Library docs (reference) → BALANCED filtering  
   curl -X POST "https://r.jina.ai/" -H "Authorization: Bearer $JINA_API_KEY" \
     -H "X-Remove-Selector: .ads,.newsletter-signup" -H "X-Token-Budget: 4000" \
     -d '{"url":"https://reflex.dev/docs/library/"}'
   ```

5. **File Output:**
   ```
   /research/reflex/page1_chatapp_tutorial.md (4,500 tokens, 95% context)
   /research/reflex/page2_state_overview.md (3,200 tokens, 92% context)  
   /research/reflex/page3_library_components.md (2,800 tokens, 88% context)
   ```

**Result:** 10,500 tokens vs 25,000+ with naive scraping, maintaining implementation-ready context!

## Token Efficiency Example

**Before (Naive Full-Page Scraping):**
```
Reflex.dev research:
- 10 pages × 2000 tokens avg = 20,000 tokens
- 80% navigation/header bloat = 16,000 wasted tokens
- Total useful content: 4,000 tokens
- Efficiency: 20%
```

**After (Smart Context-Preserving Filtering):**
```
Reflex.dev research:
- 10 pages assessed individually
- 3 pages: AGGRESSIVE filtering (800 tokens each) = 2,400 tokens  
- 5 pages: BALANCED filtering (1200 tokens each) = 6,000 tokens
- 2 pages: MINIMAL filtering (2000 tokens each) = 4,000 tokens
- Total: 12,400 tokens (38% reduction)
- Context preservation: 95%
- Useful content density: 90%
```

**Result: 60%+ token savings with better context quality!**

## Next Steps
After successful completion, the user should have a comprehensive research knowledge base ready for PRP implementation using the `/execute-prp` command.