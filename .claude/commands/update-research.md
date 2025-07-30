---
allowed-tools: WebFetch, WebSearch, Write, Read, Bash(git:*), Grep, Glob
description: Refresh outdated documentation and API research with latest information
argument-hint: [technology-name] | scan all research if no technology specified
---

# Research Update & Documentation Refresh

**Context**: You are using the CodeFarm methodology to keep research documentation current and accurate. This prevents AI hallucinations and ensures development decisions are based on up-to-date information.

## Research Audit & Discovery

### 1. Current Research Assessment
Identify existing research that needs updating:

- **Research directories**: Use Glob tool with patterns "*research*", "*docs*" to find research directories, use head_limit 5
- **Research files**: Use Glob tool with pattern "research/**/*.md" to find research markdown files, use head_limit 10
- **Documentation age**: Use Bash tool with command: `find . -name "*.md" -path "*/research/*" -exec stat -c "%Y %n" {} \; | sort -n | tail -5` to check file ages
- **Technology focus**: ${ARGUMENTS:-"[Scanning all research for updates]"}

### 2. Outdated Content Detection
Identify research that needs refreshing:

#### Version References
- **API versions**: Use Grep tool with pattern "v[0-9]|version.*[0-9]" and glob "research/**/*.md" to find version references, use head_limit 5
- **Package versions**: Use Grep tool with pattern "@|>=|<=|==" and glob "research/**/*.md" to find package versions, use head_limit 5
- **Date references**: Use Grep tool with pattern "202[0-4]" and glob "research/**/*.md" to find date references, use head_limit 5

#### Deprecated Patterns
- **Deprecated APIs**: Use Grep tool with pattern "deprecat|legacy|obsolete" and glob "research/**/*.md", case insensitive, to find deprecated content, use head_limit 3
- **TODO items**: Use Grep tool with pattern "TODO|FIXME|UPDATE" and glob "research/**/*.md" to find action items, use head_limit 5

### 3. Technology Priority Assessment
Determine update priority based on project usage:

#### High Priority (Active Dependencies)
Read current dependencies to understand what's actively used:
- **Python deps**: @requirements.txt (if exists)
- **Node deps**: @package.json (if exists)
- **Go deps**: @go.mod (if exists)

#### Medium Priority (Research References)
- Technologies mentioned in recent code
- APIs used in current implementation
- Frameworks referenced in documentation

## Research Update Execution

### 4. Current Documentation Analysis
For each technology requiring updates:

#### Read Existing Research
- **Current docs**: @research/[technology]/research_summary.md (if exists)
- **Implementation notes**: Look for project-specific patterns
- **Known issues**: Document current problems or limitations

#### Extract Update Requirements
- What specific information is outdated?
- Which APIs or features need current documentation?
- What new capabilities are relevant to the project?

### 5. Updated Research Collection
Systematically gather current information:

#### Official Documentation
For ${ARGUMENTS:-"each identified technology"}:

**Primary Sources**:
- Official documentation websites
- API reference materials  
- Release notes and changelogs
- Migration guides

**Search Strategy**:
- "[Technology] official documentation 2025"
- "[Technology] API reference latest"
- "[Technology] migration guide"
- "[Technology] best practices 2025"

#### Implementation Patterns
- Current best practices
- Updated code examples
- Performance optimizations
- Security recommendations

### 6. Research Documentation Update
Create updated research files:

#### Updated Summary Template
```markdown
# [Technology] Research Summary - Updated [Date]

## Current Status (2025)
- **Latest Version**: [Current version]
- **Release Date**: [Latest release date]
- **Stability**: [Production ready/Beta/Alpha]
- **Support Status**: [Active/Maintenance/Deprecated]

## Key Updates Since Last Research
### New Features
- [List significant new features]
- [Breaking changes to be aware of]
- [Performance improvements]

### Deprecated Features
- [Features marked for deprecation]
- [Migration paths for deprecated features]
- [Timeline for removal]

## Implementation for Our Project
### Current Usage
- [How we currently use this technology]
- [Specific features we depend on]
- [Configuration details]

### Recommended Updates
- [ ] [Specific upgrade recommendations]
- [ ] [Configuration changes needed]
- [ ] [New features to consider adopting]

### Migration Requirements
- [Breaking changes that affect our code]
- [Steps needed to upgrade]
- [Testing requirements for upgrade]

## Code Examples (Current Best Practices)
### Basic Setup
```[language]
[Updated code examples following current best practices]
```

### Advanced Usage
```[language]
[Updated advanced patterns relevant to our use case]
```

### Error Handling
```[language]
[Current recommended error handling patterns]
```

## Performance & Security
### Performance Considerations
- [Current performance best practices]
- [Benchmarking recommendations]
- [Optimization techniques]

### Security Updates
- [Security improvements in latest versions]
- [Vulnerability fixes]
- [Security configuration recommendations]

## Integration Patterns
### With Our Tech Stack
- [How this integrates with our current stack]
- [Compatibility matrix]
- [Known integration issues]

### Dependencies
- [Current dependency requirements]
- [Version compatibility]
- [Peer dependency considerations]

## Troubleshooting
### Common Issues
- [Updated list of common problems]
- [Solutions for known issues]
- [Debugging techniques]

### Community Resources
- [Active community forums]
- [GitHub repositories]
- [Stack Overflow tags]

## Next Review Date
**Schedule next update**: [Date + 6 months]
**Update triggers**: 
- Major version releases
- Security announcements
- Breaking changes in our dependencies
```

### 7. Validation & Quality Check
Ensure updated research is accurate:

#### Fact Verification
- Cross-reference information across multiple sources
- Verify version numbers and release dates
- Test code examples if possible
- Check for consistency with official documentation

#### Implementation Testing
```python
# Quick validation script for updated research
def validate_updated_info(technology, version):
    """Validate that updated research information is accurate"""
    
    try:
        # Test import/installation if applicable
        if technology.lower() == 'python':
            import importlib
            module = importlib.import_module(technology)
            print(f"✓ {technology} available: {getattr(module, '__version__', 'unknown')}")
            
        # Test API endpoints if applicable
        elif 'api' in technology.lower():
            import requests
            # Test basic connectivity (customize per API)
            response = requests.get(f"https://api.{technology}.com/v1/status", timeout=5)
            print(f"✓ {technology} API accessible: {response.status_code}")
            
    except Exception as e:
        print(f"⚠ {technology} validation failed: {e}")

# Run validation for updated technologies
technologies_updated = ["${ARGUMENTS}"] if "${ARGUMENTS}" != "full-scan" else []
for tech in technologies_updated:
    validate_updated_info(tech, "latest")
```

## Research Organization & Maintenance

### 8. Research Structure Standardization
Ensure consistent organization:

#### Directory Structure
```
research/
├── [technology]/
│   ├── research_summary.md         # Main summary (always current)
│   ├── api_reference.md           # API documentation
│   ├── implementation_guide.md    # How-to and examples
│   ├── performance_notes.md       # Performance considerations
│   ├── security_guide.md          # Security best practices
│   └── archive/
│       ├── 2024_summary.md        # Previous versions
│       └── migration_notes.md     # Upgrade information
```

#### Index File Update
Update main research index:

```markdown
# Research Index - Updated [Date]

## Active Technologies
| Technology | Version | Last Updated | Status | Priority |
|------------|---------|--------------|--------|----------|
| [Tech1] | [Version] | [Date] | ✅ Current | High |
| [Tech2] | [Version] | [Date] | ⚠ Needs Update | Medium |
| [Tech3] | [Version] | [Date] | ❌ Outdated | Low |

## Update Schedule
- **Quarterly Review**: [Next date]
- **Security Updates**: As announced
- **Major Version**: Before adoption

## Quick Reference
### High Priority Dependencies
- [List most critical technologies for immediate updates]

### Update Triggers
- [ ] New major version releases
- [ ] Security vulnerabilities announced
- [ ] Breaking changes in dependencies
- [ ] Project integration issues
```

### 9. Automated Update Monitoring
Set up monitoring for future updates:

#### Technology Watch List
Create monitoring configuration:

```yaml
# research_monitoring.yaml
technologies:
  - name: "FastAPI"
    type: "python_package"
    current_version: "0.104.1"
    watch_for: "major_releases"
    notification: "high"
    
  - name: "React"
    type: "npm_package"
    current_version: "18.2.0"
    watch_for: "security_updates"
    notification: "critical"

update_schedule:
  security_check: "weekly"
  version_check: "monthly"
  full_review: "quarterly"
```

#### Update Notification Script
```python
#!/usr/bin/env python3
"""
Research update monitoring script
Checks for updates to monitored technologies
"""
import requests
import json
from datetime import datetime, timedelta

def check_python_package_updates(package_name, current_version):
    """Check PyPI for package updates"""
    try:
        response = requests.get(f"https://pypi.org/pypi/{package_name}/json")
        data = response.json()
        latest_version = data['info']['version']
        
        if latest_version != current_version:
            return {
                'package': package_name,
                'current': current_version,
                'latest': latest_version,
                'update_available': True
            }
    except Exception as e:
        print(f"Error checking {package_name}: {e}")
    
    return {'update_available': False}

def generate_update_report():
    """Generate report of available updates"""
    packages_to_check = [
        # Add packages from your dependencies
    ]
    
    updates_available = []
    for package in packages_to_check:
        result = check_python_package_updates(package['name'], package['version'])
        if result['update_available']:
            updates_available.append(result)
    
    if updates_available:
        print("📦 Package updates available:")
        for update in updates_available:
            print(f"  - {update['package']}: {update['current']} → {update['latest']}")
    else:
        print("✅ All packages current")

if __name__ == "__main__":
    generate_update_report()
```

### 10. Success Criteria
Research update complete when:

- [ ] All high-priority technologies reviewed
- [ ] Documentation reflects current versions
- [ ] Code examples tested and working
- [ ] Migration requirements documented
- [ ] Security implications assessed
- [ ] Performance impacts noted
- [ ] Integration patterns validated
- [ ] Next review date scheduled

## Continuous Maintenance

### Research Health Monitoring
- Monthly version checks for critical dependencies
- Quarterly comprehensive research review
- Immediate updates for security announcements
- Annual technology strategy review

### Quality Gates
- All research includes version numbers and dates
- Code examples are tested before publishing
- Breaking changes are clearly documented
- Migration paths are provided for updates

---

This research update process ensures AI tools have access to current, accurate information for development decisions.