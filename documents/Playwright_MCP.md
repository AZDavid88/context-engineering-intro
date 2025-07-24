# Playwright MCP Documentation

## Overview
The Playwright MCP (@playwright/mcp) provides browser automation capabilities through the Model Context Protocol, enabling AI agents to interact with web pages programmatically. This is essential for dynamic content extraction, form interactions, and comprehensive web scraping.

## Installation & Configuration
```json
{
  "mcpServers": {
    "playwright": {
      "command": "npx",
      "args": ["-y", "@playwright/mcp@latest"],
      "transport": "stdio"
    }
  }
}
```

## Core Capabilities

### Browser Management
- **Launch and control browser instances**
- **Handle multiple tabs and windows**
- **Manage browser lifecycle and cleanup**
- **Resize browser windows for responsive testing**

### Page Navigation & Interaction
- **Navigate to URLs and handle redirects**
- **Click elements, fill forms, submit data**
- **Handle JavaScript-heavy sites and dynamic content**
- **Wait for specific elements or conditions**

### Content Extraction & Analysis
- **Capture page snapshots for structure analysis**
- **Extract HTML content and metadata**
- **Execute custom JavaScript for data extraction**
- **Take screenshots for visual verification**

## Key Tools for Research Workflows

### Navigation Tools
```
mcp__playwright__browser_navigate(url)
```
- Navigate to any URL
- Handles redirects automatically
- Returns page load status

```
mcp__playwright__browser_navigate_back()
mcp__playwright__browser_navigate_forward()
```
- Browser history navigation
- Useful for exploring documentation site structures

### Content Analysis Tools
```
mcp__playwright__browser_snapshot()
```
- **CRITICAL for research workflows**
- Returns accessibility-based page structure
- Better than screenshots for content analysis
- Provides text content and element hierarchy

```
mcp__playwright__browser_evaluate(function, element?, ref?)
```
- **POWERFUL for custom data extraction**
- Execute JavaScript in browser context
- Can target specific elements
- Perfect for extracting links and metadata

**Example: Extract all documentation links**
```javascript
() => {
  return Array.from(document.querySelectorAll('a[href]'))
    .map(link => ({
      text: link.textContent.trim(),
      url: new URL(link.href, window.location.href).href,
      isDocLink: /docs|guide|tutorial|api|reference|getting-started/i.test(link.href)
    }))
    .filter(link => link.isDocLink && link.text.length > 0)
    .slice(0, 50); // Limit results
}
```

### Interaction Tools
```
mcp__playwright__browser_click(element, ref, button?, doubleClick?)
```
- Click on elements (buttons, links, menus)
- Supports left/right/middle click
- Essential for navigating documentation sites

```
mcp__playwright__browser_wait_for(text?, textGone?, time?)
```
- Wait for content to appear/disappear
- Wait for specific time periods
- Critical for dynamic content loading

### Tab Management
```
mcp__playwright__browser_tab_new(url?)
mcp__playwright__browser_tab_select(index)
mcp__playwright__browser_tab_list()
```
- Open multiple documentation pages
- Switch between research targets
- Parallel content analysis

## Research Workflow Patterns

### Pattern 1: Documentation Discovery
```
1. Navigate to main documentation URL
2. Take snapshot to understand structure
3. Evaluate JavaScript to extract all relevant links
4. Filter links for documentation/API references
5. Navigate to each discovered link for content extraction
```

### Pattern 2: Dynamic Content Extraction
```
1. Navigate to target page
2. Wait for dynamic content to load
3. Execute JavaScript to extract structured data
4. Take snapshot for verification
5. Move to next page/section
```

### Pattern 3: Site Structure Mapping
```
1. Start from main documentation page
2. Extract navigation menu structure
3. Follow each major section
4. Build comprehensive URL list
5. Extract content from each discovered page
```

## Best Practices for Research

### Link Discovery
- Use `browser_evaluate()` with comprehensive CSS selectors
- Filter for documentation-specific patterns
- Convert relative URLs to absolute URLs
- Limit results to prevent overwhelming extraction

### Content Quality
- Always use `browser_snapshot()` before extraction
- Wait for dynamic content with `browser_wait_for()`
- Verify page load completion before processing
- Handle JavaScript-rendered content properly

### Error Handling
- Check navigation success before proceeding
- Handle timeout scenarios gracefully
- Verify element existence before interaction
- Implement retry logic for network issues

### Performance Optimization
- Use tabs for parallel processing when possible
- Close unnecessary tabs to free resources
- Take screenshots only when necessary (they're large)
- Limit JavaScript execution time

## Integration with Jina AI

### Workflow: Playwright → Jina
```
1. Use Playwright to navigate and discover URLs
2. Extract clean content URLs from page structure
3. Pass discovered URLs to Jina Reader API
4. Jina processes each URL for clean markdown output
5. Combine results into comprehensive research files
```

### Why This Combination Works
- **Playwright handles dynamic content** that static scrapers miss
- **Playwright discovers related documentation** through navigation
- **Jina provides clean, LLM-ready content** from discovered URLs
- **Together they create comprehensive research datasets**

## Common Use Cases for Research

### API Documentation Extraction
1. Navigate to API documentation root
2. Extract all endpoint URLs from navigation
3. Visit each endpoint page via Playwright
4. Use Jina to clean and structure the content
5. Generate comprehensive API reference

### Tutorial and Guide Collection
1. Discover all tutorial/guide links from main docs
2. Navigate through multi-page tutorials
3. Extract code examples and explanations
4. Create structured learning materials

### Framework Documentation Mining
1. Navigate to framework documentation
2. Extract all concept/feature pages
3. Follow cross-references and related links
4. Build complete implementation guides

## Limitations & Considerations

### Resource Usage
- Browser instances consume significant memory
- JavaScript execution can be CPU intensive
- Multiple tabs increase resource requirements
- Consider cleanup and limits for long-running tasks

### Site-Specific Challenges
- Some sites block automated browsing
- CAPTCHA and bot detection mechanisms
- Rate limiting on documentation sites
- JavaScript-heavy sites may require longer wait times

### Content Quality
- Playwright extracts raw content (needs Jina for cleaning)
- Dynamic content may require specific timing
- Advertisement and navigation content included
- Manual filtering may be needed for relevance

## Error Patterns & Solutions

### Navigation Failures
```
Problem: Page fails to load
Solution: Implement retry logic with exponential backoff
```

### Dynamic Content Issues
```
Problem: Content not fully loaded
Solution: Use wait_for() with specific selectors or time delays
```

### Memory Issues
```
Problem: Browser consumes too much memory
Solution: Close tabs regularly, limit concurrent operations
```

### JavaScript Execution Errors
```
Problem: Custom JavaScript fails
Solution: Add error handling in evaluate() functions
```

## Security Considerations

### Safe Browsing
- Only navigate to trusted documentation sites
- Avoid executing untrusted JavaScript
- Be cautious with file uploads/downloads
- Monitor for malicious redirects

### Data Privacy
- Be aware of cookies and tracking
- Consider using private/incognito mode
- Avoid submitting sensitive information
- Respect robots.txt and site policies

## Performance Tips

### Optimal Research Workflow
1. **Batch URL discovery** before content extraction
2. **Use snapshots** instead of full page screenshots
3. **Implement smart waiting** (element-based vs time-based)
4. **Clean up resources** (close tabs, browser instances)
5. **Parallel processing** where site allows it

### Memory Management
- Close unused tabs immediately
- Restart browser instance for long sessions  
- Limit concurrent browser operations
- Monitor system resources during execution

This documentation provides the foundation for building robust research workflows using Playwright MCP in combination with other tools like Jina AI for comprehensive documentation extraction and analysis.