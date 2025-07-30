---
allowed-tools: Write, Read, Glob, Grep, Bash(find:*), Bash(git:*)
description: Create detailed specifications for new features using IndyDevDan's plan-first methodology
argument-hint: [feature-name] | interactive feature planning if no name provided
---

# Feature Planning & Specification Generator

**Context**: You are using the CodeFarm methodology to create detailed, executable feature specifications. This implements IndyDevDan's core principle: "The Plan IS the Prompt" - comprehensive planning that scales your development capabilities.

## Feature Discovery & Requirements

### 1. Feature Identification
**Feature Name**: ${ARGUMENTS:-"[Interactive Mode - Please describe the feature]"}

### 2. Current Project Context Analysis
Before planning, understand the existing system:

- **Project structure**: Use Glob tool with patterns "src", "lib", "app" to find main directories, use head_limit 5
- **Existing features**: Use Glob tool with patterns "**/*.py", "**/*.js", "**/*.ts" to find source files, use head_limit 10
- **Test patterns**: Use Glob tool with pattern "*test*" to find test files, use head_limit 5
- **Configuration setup**: Use Glob tool with patterns "*.json", "*.yaml", "*.toml" to find config files, use head_limit 5

### 3. Requirements Gathering
For the specified feature, analyze:

#### Business Requirements
- **Primary Purpose**: What problem does this feature solve?
- **User Stories**: Who will use this and how?
- **Success Criteria**: How do we measure success?
- **Priority Level**: Critical/High/Medium/Low

#### Technical Requirements
- **Integration Points**: How does this connect to existing code?
- **Data Requirements**: What data does this feature need?
- **Performance Requirements**: Any speed/scale constraints?
- **Security Requirements**: Authentication, authorization, data protection

#### Constraints & Dependencies
- **External Dependencies**: New packages, APIs, services needed
- **Internal Dependencies**: Existing modules this feature depends on
- **Compatibility Requirements**: Version constraints, platform support
- **Timeline Constraints**: Delivery expectations

## Architecture Planning

### 4. Design Approach
Based on existing project patterns, design the feature:

#### Module Structure
```
[feature-name]/
├── __init__.py
├── core.py           # Main business logic
├── api.py            # External interfaces (REST, GraphQL, etc.)
├── models.py         # Data models/schemas
├── services.py       # Business services
├── utils.py          # Utility functions
└── tests/
    ├── test_core.py
    ├── test_api.py
    └── test_integration.py
```

#### Integration Points
- **Entry Points**: Where users/systems access this feature
- **Data Flow**: How information moves through the system
- **Event Triggers**: What initiates feature operations
- **Output Interfaces**: What this feature produces/returns

### 5. Implementation Strategy

#### Phase Breakdown
**Phase 1: Foundation**
- [ ] Core data models
- [ ] Basic business logic
- [ ] Unit test framework

**Phase 2: Integration**
- [ ] API endpoints/interfaces
- [ ] Database integration
- [ ] External service connections

**Phase 3: Enhancement**
- [ ] Error handling & edge cases
- [ ] Performance optimization
- [ ] Comprehensive testing

**Phase 4: Production Readiness**
- [ ] Documentation
- [ ] Monitoring & logging
- [ ] Deployment configuration

#### Risk Assessment
- **High Risk**: [Identify complex/uncertain areas]
- **Medium Risk**: [Areas needing validation]
- **Low Risk**: [Standard implementation patterns]

## Detailed Implementation Plan

### 6. Code Structure Specification

#### Data Models
```python
# models.py
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class [FeatureName]Model(BaseModel):
    """Core data model for [feature]"""
    id: Optional[str] = None
    # [Define specific fields based on requirements]
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        # [Pydantic configuration]
        pass
```

#### Core Business Logic
```python
# core.py
from typing import List, Optional
from .models import [FeatureName]Model

class [FeatureName]Service:
    """Core business logic for [feature]"""
    
    def __init__(self):
        # [Initialize dependencies]
        pass
    
    async def create_[feature](self, data: [FeatureName]Model) -> [FeatureName]Model:
        """Create new [feature] instance"""
        # [Implementation logic]
        pass
    
    async def get_[feature](self, feature_id: str) -> Optional[[FeatureName]Model]:
        """Retrieve [feature] by ID"""
        # [Implementation logic]
        pass
    
    # [Additional methods based on requirements]
```

#### API Interface
```python
# api.py
from fastapi import APIRouter, HTTPException, Depends
from .core import [FeatureName]Service
from .models import [FeatureName]Model

router = APIRouter(prefix="/[feature]", tags=["[feature]"])

@router.post("/", response_model=[FeatureName]Model)
async def create_[feature](
    data: [FeatureName]Model,
    service: [FeatureName]Service = Depends()
):
    """Create new [feature]"""
    try:
        return await service.create_[feature](data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# [Additional endpoints]
```

### 7. Testing Strategy

#### Unit Tests
```python
# tests/test_core.py
import pytest
from unittest.mock import Mock, AsyncMock
from ..core import [FeatureName]Service
from ..models import [FeatureName]Model

class Test[FeatureName]Service:
    """Test core business logic"""
    
    @pytest.fixture
    def service(self):
        return [FeatureName]Service()
    
    @pytest.mark.asyncio
    async def test_create_[feature]_success(self, service):
        """Test successful [feature] creation"""
        # [Test implementation]
        pass
    
    @pytest.mark.asyncio
    async def test_create_[feature]_validation_error(self, service):
        """Test validation error handling"""
        # [Test implementation]
        pass
    
    # [Additional test methods]
```

#### Integration Tests
```python
# tests/test_integration.py
import pytest
from fastapi.testclient import TestClient
from ..api import router

class Test[FeatureName]Integration:
    """Test feature integration"""
    
    @pytest.fixture
    def client(self):
        # [Setup test client]
        pass
    
    def test_[feature]_api_flow(self, client):
        """Test complete API workflow"""
        # [Test implementation]
        pass
```

### 8. Configuration & Dependencies

#### New Dependencies
Add to requirements.txt:
```
# [List any new packages needed]
```

#### Configuration Changes
Add to config/settings.py:
```python
# [FeatureName] Configuration
[FEATURE]_ENABLED: bool = True
[FEATURE]_[SETTING]: str = "[value]"
```

#### Environment Variables
```bash
# .env additions
[FEATURE]_API_KEY=your_key_here
[FEATURE]_DATABASE_URL=connection_string
```

## Implementation Validation

### 9. Definition of Done
- [ ] Core functionality implemented and tested
- [ ] API endpoints functional and documented
- [ ] Unit test coverage >90%
- [ ] Integration tests passing
- [ ] Error handling comprehensive
- [ ] Performance benchmarks met
- [ ] Documentation complete
- [ ] Code review approved
- [ ] Production deployment tested

### 10. Success Metrics
- **Functional**: [Specific functionality measures]
- **Performance**: [Speed, memory, throughput requirements]
- **Quality**: [Error rates, test coverage, code quality]
- **User Experience**: [Usability, accessibility measures]

## Specification Output

### Create Specification File
Generate `specs/feature_[feature-name].md`:

```markdown
# Feature Specification: [Feature Name]

## Overview
[Feature description and purpose]

## Requirements
### Business Requirements
- [List business requirements]

### Technical Requirements  
- [List technical requirements]

## Architecture
### Module Structure
[Detailed module breakdown]

### Integration Points
[How this integrates with existing system]

## Implementation Plan
### Phase 1: Foundation
- [ ] [Specific tasks]

### Phase 2: Integration
- [ ] [Specific tasks]

### Phase 3: Enhancement
- [ ] [Specific tasks]

### Phase 4: Production Readiness
- [ ] [Specific tasks]

## Testing Strategy
[Comprehensive testing approach]

## Dependencies
[New dependencies and configuration changes]

## Definition of Done
[Specific completion criteria]

## Success Metrics
[Measurable success indicators]
```

## Next Steps

After creating this specification:

1. **Review & Validate**: Ensure all requirements captured
2. **Estimate Effort**: Time and resource requirements
3. **Get Approval**: Stakeholder sign-off on plan
4. **Execute**: Use `/implement-spec` to build the feature
5. **Validate**: Use `/validate-system` to verify implementation

---

This planning process ensures comprehensive, executable specifications that scale your development capabilities through detailed upfront planning.