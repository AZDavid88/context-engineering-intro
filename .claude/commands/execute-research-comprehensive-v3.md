# Execute Research - Comprehensive V3 (Multi-Vector Documentation Discovery)

---
allowed-tools: WebFetch, Read, Write, Glob, Grep, TodoWrite, Bash, mcp__brightdata__scrape_as_markdown
description: Advanced research command for GitHub repositories with complex API documentation structures
argument-hint: [path/to/planning_prp.md]
---

## Purpose
Research GitHub repositories with comprehensive documentation extraction using multi-vector discovery. Specifically designed for projects with structured API documentation, OpenAPI specs, and complex folder hierarchies that standard methods miss.

## Arguments
- `$ARGUMENTS`: Path to planning_prp.md file

## When to Use This Command
**ONLY use V3 when ALL of these conditions are met:**
1. Target is a GitHub repository (not general documentation sites)
2. Repository likely has `/api/`, `/docs/`, `/spec/` folders
3. Previous research methods missed critical API documentation  
4. Repository has >100 files (complex project structure)
5. Need complete REST endpoint/OpenAPI documentation

**DO NOT use V3 for:**
- Simple projects with just examples
- Non-GitHub documentation
- When standard `/execute-research` works fine

## Execution Process

### 1. **Load PRP and Environment Setup**
- Read planning_prp.md file at specified path
- Extract ALL research_targets with URLs from YAML frontmatter
- **VALIDATE**: All targets must be GitHub repository URLs
- Derive project directory from planning_prp.md path
- Create research directories: `{project_dir}/research/{technology}_v3/`
- Use TodoWrite to track multi-vector progress

### 2. **Multi-Vector Discovery Strategy**

#### Vector 1: Repository Structure Analysis
For each GitHub repository URL:
```bash
# Transform GitHub URLs to explore structured paths
base_url="$target_url"
# Example: https://github.com/user/repo → https://github.com/user/repo/tree/master/api

probe_paths=(
  "/tree/master/api"
  "/tree/master/docs" 
  "/tree/master/examples"
  "/tree/master/spec"
  "/tree/master/schema"
  "/tree/master/reference"
)
```

#### Vector 2: Direct API File Probing
```bash
# Probe for common API specification files
api_files=(
  "/blob/master/api/components.yaml"
  "/blob/master/api/openapi.yaml"
  "/blob/master/api/swagger.yaml"
  "/blob/master/spec/api.yaml"
  "/blob/master/docs/api.md"
)
```

#### Vector 3: Structured Folder Discovery
For discovered API folders, probe for:
```bash
# Common API documentation patterns
api_subpaths=(
  "/info/userstate.yaml"
  "/info/orders.yaml" 
  "/info/market.yaml"
  "/endpoints/"
  "/schemas/"
  "/examples/"
)
```

### 3. **Intelligent Content Extraction**

#### Phase A: Repository Structure Mapping
For each probe URL:
- Use WebFetch with specialized prompts:
  ```
  Extract the complete directory structure and file listings. 
  Focus on API-related files (.yaml, .json, .md), documentation files, 
  and example code. List all subdirectories and files with their purposes.
  Ignore general repository files (README, LICENSE, etc.) unless they contain API documentation.
  ```

#### Phase B: API Specification Extraction  
For discovered API files:
- Use `mcp__brightdata__scrape_as_markdown` for clean extraction
- If Brightdata fails (robots.txt), use WebFetch with enhanced prompts:
  ```
  Extract the complete API specification from this file. Focus on:
  1. All endpoint definitions and paths
  2. Request/response schemas and formats
  3. Authentication requirements
  4. Parameter definitions and constraints
  5. Error codes and responses
  Return as clean, structured markdown with properly formatted code blocks.
  ```

#### Phase C: Cross-Reference Validation
- Match discovered API endpoints to SDK examples
- Identify implementation gaps between specs and examples
- Extract missing integration patterns

### 4. **Quality Assurance and Synthesis**

#### File Naming Convention
```
research/{technology}_v3/
├── vector1_repo_structure.md      # Repository mapping
├── vector2_api_specs.md           # OpenAPI/YAML specifications  
├── vector3_examples.md            # SDK examples and patterns
├── vector4_integration.md         # Cross-referenced patterns
└── research_summary_v3.md         # Comprehensive analysis
```

#### Content Quality Standards
- **API Coverage**: 100% of discovered endpoints documented
- **Implementation Readiness**: Complete request/response examples
- **Cross-Validation**: API specs matched to SDK implementations
- **Zero Navigation**: <5% navigation content (Brightdata quality)

### 5. **Advanced Error Handling**

#### GitHub-Specific Issues
- **Robots.txt blocking**: Fall back to WebFetch with targeted prompts
- **Rate limiting**: Implement exponential backoff between requests
- **Private repositories**: Log warning and skip (cannot access)
- **Deleted/moved files**: Try alternative common paths

#### Tool Fallback Chain
1. **Primary**: `mcp__brightdata__scrape_as_markdown` (best quality)
2. **Fallback**: WebFetch with specialized GitHub prompts
3. **Emergency**: Manual URL construction with pattern matching

### 6. **Success Validation**

#### Completion Criteria
- All GitHub repositories processed through 3+ vectors
- API specifications extracted (if available)
- SDK examples cross-referenced with API specs
- Implementation gaps identified and documented
- Research summary includes quality metrics and completeness assessment

#### Quality Gates
- **API Completeness**: >90% of endpoints documented
- **Implementation Ready**: >95% useful content ratio
- **Cross-Validation**: API specs match SDK patterns
- **Zero Duplication**: No content overlap between vectors

## Output Structure
```
{project_dir}/research/
├── {technology}_v3/
│   ├── vector1_repo_structure.md      # Repository file structure
│   ├── vector2_api_specs.md           # Complete API documentation  
│   ├── vector3_examples.md            # SDK implementation patterns
│   ├── vector4_integration.md         # Cross-referenced usage
│   └── research_summary_v3.md         # Comprehensive analysis
├── research_comparison_v3.md           # V3 vs standard methods
└── extraction_report_v3.md             # Quality and completeness metrics
```

## Command Completion Signal
Upon successful completion, output:
```
✅ V3 COMPREHENSIVE RESEARCH COMPLETE
- GitHub repositories analyzed: {count}
- API vectors discovered: {count}
- Specification files extracted: {count}  
- Implementation patterns documented: {count}
- Cross-validation completeness: {percentage}%
- API coverage: {percentage}%
- Status: Enterprise-grade documentation ready
```

## Critical Limitations
- **GitHub repositories ONLY** - Will not work on general documentation sites
- **Requires structured API documentation** - Useless for simple projects
- **Higher resource usage** - More API calls and processing time
- **Overkill for basic needs** - Use standard methods when sufficient

## When V3 Succeeds vs Fails

### V3 SUCCESS scenarios:
- ✅ Large GitHub repos with `/api/` folders (like Hyperliquid SDK)
- ✅ Projects with OpenAPI/YAML specifications  
- ✅ Complex SDKs with separate API documentation
- ✅ When standard methods miss critical endpoints

### V3 FAILURE scenarios:
- ❌ Simple projects with only examples
- ❌ Non-GitHub documentation sites
- ❌ Projects without structured API docs
- ❌ When standard research already complete