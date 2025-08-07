---
description: Create comprehensive research foundation using Jina AI for implementation plans (ANTI-HALLUCINATION ENFORCED)
allowed-tools: Read, Glob, Grep, Bash, Write, mcp__filesystem__create_directory, mcp__filesystem__write_file, TodoWrite
argument-hint: [implementation_plan_path] - Path to implementation plan requiring research foundation
---

# Research Foundation Builder - Jina AI Integration

## Dynamic Project Context Discovery
!`pwd`
!`ls -la`

## Your Task - Complete Research Foundation Creation

**Objective:** Systematically build comprehensive research foundation for implementation plans using Jina AI Search Foundation API.

**CRITICAL**: This command uses actual Jina AI APIs with curl commands. Requires `JINA_API_KEY` environment variable.

## Phase 1: Discovery & Requirements Analysis

**Comprehensive project analysis through systematic discovery**

1. **Read Implementation Plan**: Use Read tool to analyze the provided implementation plan file
2. **Use TodoWrite**: Create systematic task tracking for each technology/dependency found
3. **Inventory Context Needs**: List EXACTLY what research context is needed
4. **Prioritize Research**: Categorize as CRITICAL, IMPORTANT, SUPPLEMENTAL
5. **Validate Environment**: Check that `JINA_API_KEY` is available

**Evidence validation**: All assumptions verified against actual codebase and implementation plan.

## Phase 2: Coverage Analysis & Gap Identification  

**Systematic assessment of existing research vs requirements**

1. **Check Existing Research**: Use Glob to find `/research/[technology]/` directories
2. **Analyze Current Coverage**: Use Read to examine existing research files  
3. **Search Implementation Details**: Use Grep to find specific patterns in existing research
4. **Identify Research Gaps**: Compare existing coverage against Phase 1 inventory
5. **Update TodoWrite**: Document findings and gaps for each technology

**Evidence validation**: All gap analysis based on actual file content and implementation needs.

## Phase 3: Jina AI Research Execution

**Direct Jina AI API integration for comprehensive content extraction**

For each missing research gap:

### Jina Reader API Implementation:
```bash
# Get your Jina AI API key for free: https://jina.ai/?sui=apikey
curl "https://r.jina.ai/$DOCUMENTATION_URL" \
  -H "Authorization: Bearer $JINA_API_KEY" \
  -H "Accept: application/json" \
  -H "X-Return-Format: markdown" \
  -H "X-With-Links-Summary: true" > /tmp/jina_response.json

# Parse JSON response to extract content
jq -r '.data.content' /tmp/jina_response.json > research_content.md
```

### Jina Search API for Discovery:
```bash
curl -X POST 'https://s.jina.ai/' \
  -H "Authorization: Bearer $JINA_API_KEY" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{"q":"[SEARCH_QUERY] official documentation"}' > /tmp/search_response.json

# Extract URLs from search results  
jq -r '.data[].url' /tmp/search_response.json
```

**Tool Integration**: Use Bash tool with curl commands, parse JSON responses with jq, validate content before documentation.

**Evidence validation**: All content extracted from official documentation sources with source attribution.

## Phase 4: Immediate Documentation & Validation

**CRITICAL**: Document each extraction IMMEDIATELY after successful API call

1. **For Each Successful Extraction**:
   - IMMEDIATELY create research file using Write tool
   - Use naming: `/research/[technology]/[page_name].md`
   - Include source URL, extraction date, and API response metadata
   - Update TodoWrite to mark extraction as completed

2. **Content Validation**:
   - Verify JSON parsing succeeded
   - Confirm markdown content is readable
   - Validate source URLs are accessible
   - Check for implementation-relevant details

3. **Research Directory Structure**:
```
/research/
├── [technology1]/
│   ├── 01_getting_started.md
│   ├── 02_api_reference.md
│   └── 03_implementation_guide.md
└── [technology2]/
    ├── 01_installation.md
    └── 02_configuration.md
```

**Evidence validation**: All documentation matches actual API responses and includes proper source attribution.

## Implementation Script Template

```bash
#!/bin/bash
# Research Foundation Builder with Jina AI Integration

# Check API key
if [ -z "$JINA_API_KEY" ]; then
    echo "ERROR: JINA_API_KEY environment variable not set"
    echo "Get your free API key: https://jina.ai/?sui=apikey"
    exit 1
fi

# Function to extract content with Jina Reader
jina_read() {
    local url="$1"
    local output_file="$2"
    
    echo "Extracting content from: $url"
    
    curl -s "https://r.jina.ai/$url" \
        -H "Authorization: Bearer $JINA_API_KEY" \
        -H "Accept: application/json" \
        -H "X-Return-Format: markdown" \
        -H "X-With-Links-Summary: true" > /tmp/jina_response.json
    
    # Check if request was successful
    if [ $? -eq 0 ] && [ -s /tmp/jina_response.json ]; then
        # Extract content from JSON response
        jq -r '.data.content' /tmp/jina_response.json > "$output_file"
        echo "SUCCESS: Content saved to $output_file"
        return 0
    else
        echo "ERROR: Failed to extract content from $url"
        return 1
    fi
}

# Function to search for documentation
jina_search() {
    local query="$1"
    
    echo "Searching for: $query"
    
    curl -s -X POST 'https://s.jina.ai/' \
        -H "Authorization: Bearer $JINA_API_KEY" \
        -H "Content-Type: application/json" \
        -H "Accept: application/json" \
        -d "{\"q\":\"$query official documentation\"}" > /tmp/search_response.json
    
    # Extract and display URLs
    if [ $? -eq 0 ] && [ -s /tmp/search_response.json ]; then
        echo "Found documentation URLs:"
        jq -r '.data[].url' /tmp/search_response.json
        return 0
    else
        echo "ERROR: Search failed for query: $query"
        return 1
    fi
}
```

## Anti-Hallucination Safeguards

### Tool Usage Enforcement:
- **Only approved tools** from allowed-tools list
- **Direct Jina AI integration** using curl commands via Bash tool
- **TodoWrite required** for systematic progress tracking
- **Immediate documentation** prevents batch-and-forget errors

### Process Validation:
- **API key validation** before any API calls
- **JSON response parsing** with error handling
- **Content verification** before file creation
- **Source attribution** for all extractions

### Quality Assurance:
- **Official documentation** sources only via Jina search
- **Implementation-focused** content extraction
- **Structured organization** in research directories
- **Progressive validation** throughout execution

## Success Criteria - Context Independent Operation

- ✅ **Phase 1 Complete**: Implementation plan analyzed, TodoWrite tracking initiated
- ✅ **Phase 2 Complete**: Existing research assessed, gaps systematically identified  
- ✅ **Phase 3 Complete**: Missing research gathered using Jina AI APIs directly
- ✅ **Phase 4 Complete**: All extractions documented immediately with source attribution
- ✅ **Anti-Hallucination Protected**: All content from verified official sources
- ✅ **Implementation Ready**: Research foundation complete for development phase
- ✅ **Environment Validated**: JINA_API_KEY confirmed and functional
- ✅ **Context Independence**: Works after /clear or /compact through systematic discovery

## Expected Output Structure

```
/research/[technology]/
├── 00_research_summary.md     # Overview and implementation guidance
├── 01_getting_started.md      # Installation and setup from official docs
├── 02_api_reference.md        # Core API documentation 
├── 03_authentication.md       # Security and auth patterns
├── 04_implementation_guide.md # Practical implementation examples
└── 05_troubleshooting.md     # Common issues and solutions
```

Each file includes:
- **Source URL**: Direct link to official documentation
- **Extraction Date**: When content was retrieved
- **Context**: Specific use case relevance
- **Implementation Notes**: Key details for development phase