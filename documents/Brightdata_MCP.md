# Brightdata MCP Documentation

## Overview
The Brightdata MCP (@brightdata/mcp) provides enterprise-grade web scraping capabilities through the Model Context Protocol. It offers advanced features like proxy rotation, bot detection evasion, and geo-unblocking for reliable data extraction from any public website.

## Installation & Configuration
```json
{
  "mcpServers": {
    "brightdata": {
      "command": "npx",
      "args": ["-y", "@brightdata/mcp"],
      "transport": "stdio",
      "env": {
        "API_TOKEN": "your-brightdata-api-token"
      }
    }
  }
}
```

## Core Capabilities

### Enterprise Web Scraping
- **Bypasses bot detection and CAPTCHAs**
- **Handles geo-restricted content**
- **Automatic proxy rotation and IP management**
- **JavaScript rendering for dynamic content**

### Structured Data Extraction
- **Direct access to popular platforms** (Amazon, LinkedIn, etc.)
- **Clean markdown/HTML output**
- **JSON-formatted structured data**
- **Real-time content retrieval**

### Browser Automation
- **Remote browser control**
- **Link discovery and navigation**
- **Form interactions and page manipulation**
- **Screenshot and content capture**

## Key Tools for Research Workflows

### Direct Content Scraping
```
mcp__brightdata__scrape_as_markdown(url)
```
- **PRIMARY tool for research workflows**
- Returns clean, LLM-ready markdown content
- Bypasses most bot detection automatically
- Handles JavaScript-rendered content

```
mcp__brightdata__scrape_as_html(url)
```
- Returns raw HTML when markdown isn't sufficient
- Useful for complex page structures
- Preserves original formatting and elements

```
mcp__brightdata__extract(url, extraction_prompt?)
```
- **POWERFUL for structured data extraction**
- Uses AI to extract specific information
- Custom prompts for targeted data mining
- Returns JSON-formatted results

### Browser Automation Tools
```
mcp__brightdata__scraping_browser_navigate(url)
```
- Navigate to URLs with full bot protection evasion
- Handles redirects and dynamic loading
- More reliable than standard browser automation

```
mcp__brightdata__scraping_browser_links()
```
- **PERFECT for documentation discovery**
- Returns all links on current page with text and selectors
- Essential for finding related documentation pages
- No need to write custom JavaScript

```
mcp__brightdata__scraping_browser_click(selector)
```
- Click elements using CSS selectors
- Navigate through multi-page documentation
- Access gated or menu-driven content

```
mcp__brightdata__scraping_browser_get_text()
mcp__brightdata__scraping_browser_get_html(full_page?)
```
- Extract text or HTML content from current page
- Optional full page HTML including scripts
- Clean content extraction without navigation noise

### Page Interaction Tools
```
mcp__brightdata__scraping_browser_wait_for(selector, timeout?)
```
- Wait for specific elements to load
- Essential for dynamic documentation sites
- Configurable timeout periods

```
mcp__brightdata__scraping_browser_scroll()
mcp__brightdata__scraping_browser_scroll_to(selector)
```
- Handle infinite scroll documentation
- Navigate to specific page sections
- Load additional content dynamically

```
mcp__brightdata__scraping_browser_screenshot(full_page?)
```
- Visual verification of page state
- Debug navigation and content issues
- Document page structure for analysis

## Research Workflow Patterns

### Pattern 1: Documentation Discovery & Extraction
```
1. Navigate to main documentation URL
2. Use scraping_browser_links() to discover all related pages
3. Filter links for documentation patterns
4. Use scrape_as_markdown() on each discovered URL
5. Generate clean research files from extracted content
```

### Pattern 2: Deep Content Mining
```
1. Navigate to target documentation section
2. Use extract() with custom prompts for specific data
3. Navigate through pagination or related sections
4. Combine structured extractions into comprehensive guides
```

### Pattern 3: Multi-Site Documentation Aggregation
```
1. Process multiple documentation sites in parallel
2. Use consistent extraction patterns across sites
3. Normalize content format through markdown extraction
4. Create unified research repositories
```

## Structured Data Platform Support

### Technology Documentation Platforms
- **GitHub Repositories**: Extract README, docs, code examples
- **API Documentation Sites**: Swagger, Postman, custom docs
- **Framework Documentation**: React, Vue, Angular, etc.
- **Cloud Provider Docs**: AWS, GCP, Azure documentation

### Specialized Extraction Tools
```javascript
// These are available but focused on specific platforms
mcp__brightdata__web_data_github_repository_file(url)
mcp__brightdata__web_data_linkedin_company_profile(url)
// ... many others for specific platforms
```

## Best Practices for Research

### Content Quality Optimization
- **Use scrape_as_markdown()** as primary extraction method
- **Filter discovered links** for documentation relevance
- **Implement retry logic** for network failures
- **Validate content quality** before saving

### Link Discovery Strategy
```javascript
// Effective link filtering for documentation
const relevantLinks = allLinks.filter(link => 
  /docs|guide|tutorial|api|reference|getting-started|examples/i.test(link.url) &&
  !link.url.includes('#') && // Skip anchor links
  link.text.length > 5 // Skip empty or very short links
);
```

### Error Handling
- **Network failures**: Implement exponential backoff
- **Bot detection**: Automatic retry with different approaches
- **Rate limiting**: Built-in handling through Brightdata infrastructure
- **Content validation**: Check for meaningful content extraction

### Performance Optimization
- **Batch URL processing** when possible
- **Use appropriate extraction method** for content type
- **Implement parallel processing** for multiple targets
- **Monitor API usage** and token consumption

## Integration with Jina AI

### Workflow: Brightdata → Jina Enhancement
```
1. Use Brightdata to extract initial content as markdown
2. Pass content through Jina Reader for additional cleaning
3. Use Jina's AI capabilities for targeted information extraction
4. Combine for highest quality research output
```

### Workflow: Brightdata Discovery → Jina Processing
```
1. Use Brightdata browser to discover all relevant URLs
2. Extract URLs list from documentation navigation
3. Pass each URL to Jina Reader for processing
4. Generate comprehensive research files
```

### Why This Combination Works
- **Brightdata handles bot protection** that blocks other scrapers
- **Brightdata discovers comprehensive URL lists** through navigation
- **Jina provides consistent content formatting** and AI-enhanced extraction
- **Together they handle any documentation site** regardless of protection

## Advanced Features

### Geo-Location Support
```
// Access region-specific documentation
mcp__brightdata__scraping_browser_navigate(url)
// Automatically uses optimal geographic location
```

### JavaScript Rendering
- **Full browser rendering** of modern documentation sites
- **Dynamic content loading** handled automatically
- **SPA (Single Page Application) support** for modern docs
- **AJAX content** extracted properly

### Anti-Bot Evasion
- **Automatic detection** of protection mechanisms
- **Adaptive strategies** for different site types
- **Proxy rotation** handled transparently
- **Browser fingerprinting** management

## Common Use Cases for Research

### Comprehensive API Documentation
1. Navigate to API documentation root
2. Discover all endpoint sections via link extraction
3. Extract each endpoint's documentation as markdown
4. Generate structured API reference materials

### Framework Learning Materials
1. Access framework documentation sites
2. Extract tutorials, guides, and examples
3. Process code samples and explanations
4. Create comprehensive learning resources

### Multi-Platform Documentation Aggregation
1. Process documentation from multiple related tools
2. Normalize content format through consistent extraction
3. Create unified knowledge base
4. Cross-reference related concepts

## Error Handling & Troubleshooting

### Common Issues & Solutions

#### Bot Detection Encountered
```
Problem: Site blocks automated access
Solution: Brightdata handles this automatically through proxy rotation
```

#### JavaScript Content Not Loading
```
Problem: Dynamic content missing from extraction
Solution: Use browser navigation tools with wait_for() before extraction
```

#### Rate Limiting
```
Problem: Too many requests to documentation site  
Solution: Built-in rate limiting and proxy management handles this
```

#### Content Quality Issues
```
Problem: Extracted content contains navigation/ads
Solution: Use scrape_as_markdown() which provides cleaner content
```

### Debugging Techniques
- **Use screenshots** to verify page state
- **Extract HTML** to understand page structure
- **Test navigation** step-by-step
- **Validate link discovery** before bulk extraction

## Performance Considerations

### Optimal Usage Patterns
- **Batch similar operations** when possible
- **Use appropriate extraction method** for content type
- **Implement smart retry logic** for transient failures
- **Monitor resource usage** and API limits

### Resource Management
- **Close browser sessions** when done
- **Limit concurrent operations** based on site capacity
- **Cache results** to avoid re-extraction
- **Implement progressive extraction** for large sites

### API Usage Optimization
- **Use markdown extraction** for cleaner results
- **Filter URLs** before processing to reduce API calls
- **Combine related extractions** in single sessions
- **Monitor usage dashboards** for optimization opportunities

## Security & Compliance

### Ethical Scraping
- **Respect robots.txt** when possible
- **Implement reasonable delays** between requests
- **Focus on public documentation** only
- **Avoid overwhelming target sites**

### Data Privacy
- **Only extract public information**
- **Respect site terms of service**
- **Handle personal data appropriately**
- **Consider data retention policies**

### Legal Considerations
- **Public documentation** is generally acceptable
- **Commercial use** may have different requirements
- **International compliance** handled by Brightdata infrastructure
- **Terms of service** should be reviewed per site

## Integration Examples

### Research Automation Pipeline
```python
# Pseudo-code for comprehensive research extraction
def research_technology(base_url, technology_name):
    # 1. Discover all relevant documentation URLs
    discovered_urls = discover_documentation_links(base_url)
    
    # 2. Extract content from each URL
    research_content = []
    for url in discovered_urls:
        content = brightdata_scrape_markdown(url)
        if validate_content_quality(content):
            research_content.append({
                'url': url,
                'content': content,
                'technology': technology_name
            })
    
    # 3. Generate research files
    generate_research_files(research_content, technology_name)
```

### Quality-First Extraction
```python
def high_quality_extraction(url):
    # Try markdown extraction first
    content = brightdata_scrape_markdown(url)
    
    # Enhance with Jina if needed
    if needs_enhancement(content):
        enhanced_content = jina_process(content)
        return enhanced_content
    
    return content
```

This documentation provides comprehensive guidance for using Brightdata MCP in research workflows, especially when combined with Jina AI for optimal content extraction and processing.