---
allowed-tools: Write, MultiEdit, Read, Bash(mkdir:*), Bash(git:*), Bash(python:*), Bash(pytest:*), Glob, Grep
description: Execute detailed feature specifications with comprehensive validation and testing
argument-hint: [spec-file-path] | interactive mode if no spec provided
---

# Feature Implementation Executor

**Context**: You are using the CodeFarm methodology to execute detailed feature specifications. This implements IndyDevDan's principle that detailed plans become executable prompts that scale development capabilities.

## Pre-Implementation Setup

### 1. Specification Analysis
Load and analyze the feature specification:

- **Spec file**: @${ARGUMENTS:-"specs/"}
- **Project context**: Read project structure and existing patterns
- **Dependencies**: Verify all required dependencies are available
- **Git status**: Use Bash tool with command: `git status` to ensure clean working directory

### 2. Implementation Environment Validation
Before coding, verify environment:

- **Python environment**: Use Bash tool with command: `python --version` to check Python version
- **Virtual environment active**: Use Bash tool with command: `echo $VIRTUAL_ENV` to verify virtual environment
- **Dependencies installed**: Use Bash tool with command: `pip list | grep -E "(fastapi|pydantic|pytest)"` to check required packages
- **Test framework available**: Use Bash tool with command: `python -m pytest --version` to verify pytest installation

### 3. Create Implementation Branch
Safety protocol for implementation:

Use Bash tool with the following commands to create feature branch:

```bash
# Create feature branch
git checkout -b feature/[feature-name]

# Create initial commit
git add -A
git commit -m "Start implementation: [feature-name]"
```

## Systematic Implementation

### 4. Phase 1 - Data Models & Core Logic

#### Create Module Structure
Based on specification, create directory structure:

Use Bash tool with the following commands to create module structure:

```bash
# Create feature module
mkdir -p src/[feature-name]
mkdir -p src/[feature-name]/tests
mkdir -p src/[feature-name]/migrations  # if database changes needed
```

#### Implement Data Models
Create `src/[feature-name]/models.py`:

```python
"""
Data models for [feature-name]
Generated from specification: [spec-file]
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

# [Implement models based on specification]
class [FeatureName]Model(BaseModel):
    """Core data model for [feature-name]"""
    id: Optional[str] = Field(None, description="Unique identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    
    # [Add fields from specification]
    
    class Config:
        """Pydantic configuration"""
        orm_mode = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# [Additional models as specified]
```

#### Implement Core Business Logic
Create `src/[feature-name]/core.py`:

```python
"""
Core business logic for [feature-name]
Implements all business rules and operations
"""
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from .models import [FeatureName]Model
from .exceptions import [FeatureName]Error

logger = logging.getLogger(__name__)

class [FeatureName]Service:
    """Core business service for [feature-name]"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize service with configuration"""
        self.config = config or {}
        self._setup_dependencies()
    
    def _setup_dependencies(self):
        """Setup internal dependencies"""
        # [Initialize required dependencies]
        pass
    
    async def create_[feature](self, data: [FeatureName]Model) -> [FeatureName]Model:
        """
        Create new [feature] instance
        
        Args:
            data: [Feature] data model
            
        Returns:
            Created [feature] instance
            
        Raises:
            [FeatureName]Error: On validation or business rule violations
        """
        try:
            # [Implement creation logic from specification]
            logger.info(f"Creating [feature]: {data.dict()}")
            
            # Validation
            self._validate_[feature]_data(data)
            
            # Business logic
            result = await self._process_[feature]_creation(data)
            
            logger.info(f"[Feature] created successfully: {result.id}")
            return result
            
        except Exception as e:
            logger.error(f"Error creating [feature]: {str(e)}")
            raise [FeatureName]Error(f"Failed to create [feature]: {str(e)}")
    
    def _validate_[feature]_data(self, data: [FeatureName]Model) -> None:
        """Validate [feature] data according to business rules"""
        # [Implement validation logic]
        pass
    
    async def _process_[feature]_creation(self, data: [FeatureName]Model) -> [FeatureName]Model:
        """Process [feature] creation with business logic"""
        # [Implement core business logic]
        pass
    
    # [Additional methods from specification]
```

### 5. Phase 2 - API Layer & Integration

#### Create API Endpoints
Create `src/[feature-name]/api.py`:

```python
"""
API endpoints for [feature-name]
RESTful interface following OpenAPI standards
"""
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from typing import List, Optional

from .core import [FeatureName]Service
from .models import [FeatureName]Model
from .exceptions import [FeatureName]Error

router = APIRouter(
    prefix="/api/v1/[feature-name]",
    tags=["[feature-name]"],
    responses={404: {"description": "Not found"}}
)

def get_[feature]_service() -> [FeatureName]Service:
    """Dependency injection for [feature] service"""
    return [FeatureName]Service()

@router.post(
    "/",
    response_model=[FeatureName]Model,
    status_code=status.HTTP_201_CREATED,
    summary="Create [feature]",
    description="Create a new [feature] instance"
)
async def create_[feature](
    data: [FeatureName]Model,
    service: [FeatureName]Service = Depends(get_[feature]_service)
) -> [FeatureName]Model:
    """Create new [feature]"""
    try:
        return await service.create_[feature](data)
    except [FeatureName]Error as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

# [Additional endpoints from specification]

@router.get(
    "/health",
    summary="Health check",
    description="Check [feature] service health"
)
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "[feature-name]"}
```

### 6. Phase 3 - Testing Implementation

#### Unit Tests
Create `src/[feature-name]/tests/test_core.py`:

```python
"""
Unit tests for [feature-name] core business logic
Comprehensive test coverage following specification
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from ..core import [FeatureName]Service
from ..models import [FeatureName]Model
from ..exceptions import [FeatureName]Error

class Test[FeatureName]Service:
    """Test suite for [FeatureName]Service"""
    
    @pytest.fixture
    def service(self):
        """Create service instance for testing"""
        return [FeatureName]Service()
    
    @pytest.fixture
    def valid_[feature]_data(self):
        """Valid [feature] data for testing"""
        return [FeatureName]Model(
            # [Sample valid data from specification]
        )
    
    @pytest.mark.asyncio
    async def test_create_[feature]_success(self, service, valid_[feature]_data):
        """Test successful [feature] creation"""
        result = await service.create_[feature](valid_[feature]_data)
        
        assert result is not None
        assert result.id is not None
        assert result.created_at is not None
        # [Additional assertions from specification]
    
    @pytest.mark.asyncio
    async def test_create_[feature]_invalid_data(self, service):
        """Test [feature] creation with invalid data"""
        invalid_data = [FeatureName]Model(
            # [Invalid data scenarios from specification]
        )
        
        with pytest.raises([FeatureName]Error):
            await service.create_[feature](invalid_data)
    
    # [Additional test methods covering all specification requirements]
```

#### Integration Tests
Create `src/[feature-name]/tests/test_api.py`:

```python
"""
Integration tests for [feature-name] API endpoints
Test complete request/response flows
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from ..api import router
from ..models import [FeatureName]Model

class Test[FeatureName]API:
    """Test suite for [FeatureName] API"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)
    
    def test_create_[feature]_success(self, client):
        """Test successful [feature] creation via API"""
        test_data = {
            # [Valid API request data from specification]
        }
        
        response = client.post("/api/v1/[feature-name]/", json=test_data)
        
        assert response.status_code == 201
        result = response.json()
        assert "id" in result
        assert "created_at" in result
        # [Additional API response assertions]
    
    def test_create_[feature]_validation_error(self, client):
        """Test API validation error handling"""
        invalid_data = {
            # [Invalid request data scenarios]
        }
        
        response = client.post("/api/v1/[feature-name]/", json=invalid_data)
        
        assert response.status_code == 400
        # [Additional error response validation]
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/api/v1/[feature-name]/health")
        
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    # [Additional API test methods]
```

### 7. Phase 4 - Configuration & Integration

#### Update Project Configuration
Add to `src/config/settings.py`:

```python
# [FeatureName] Configuration
[FEATURE]_ENABLED: bool = Field(True, env="[FEATURE]_ENABLED")
[FEATURE]_[SETTING]: str = Field("default", env="[FEATURE]_[SETTING]")
# [Additional config from specification]
```

#### Update Dependencies
Add to `requirements.txt`:

```
# [Feature-name] dependencies
# [List new packages from specification]
```

#### Database Migrations (if needed)
Create `src/[feature-name]/migrations/001_initial.py`:

```python
"""
Initial migration for [feature-name]
Create database tables and indexes
"""
# [Database migration code if specified]
```

## Implementation Validation

### 8. Continuous Validation
After each phase, validate implementation:

Use Bash tool with the following commands for continuous validation:

```bash
# Run unit tests
python -m pytest src/[feature-name]/tests/test_core.py -v

# Run integration tests  
python -m pytest src/[feature-name]/tests/test_api.py -v

# Test imports work
python -c "from src.[feature-name] import [FeatureName]Service; print('Import successful')"

# Git commit progress
git add src/[feature-name]/
git commit -m "Implement [feature-name] Phase X: [description]"
```

### 9. Final Integration Testing
Complete system integration:

Use Bash tool with the following commands for final integration testing:

```bash
# Run full test suite
python -m pytest src/[feature-name]/tests/ -v --cov=src/[feature-name]

# Test API endpoints with real server
python -m uvicorn main:app --reload &
curl -X POST "http://localhost:8000/api/v1/[feature-name]/" \
     -H "Content-Type: application/json" \
     -d '[test-data]'

# Performance testing if specified
python -m pytest src/[feature-name]/tests/test_performance.py
```

### 10. Documentation & Completion
Generate final documentation:

```markdown
# Implementation Report: [Feature Name]

## Summary
[Brief summary of what was implemented]

## Components Implemented
- [ ] Data models ([list models])
- [ ] Core business logic ([list services])  
- [ ] API endpoints ([list endpoints])
- [ ] Unit tests ([coverage percentage])
- [ ] Integration tests ([test scenarios])
- [ ] Configuration ([config changes])

## Test Results
- Unit Tests: [X/Y passed]
- Integration Tests: [X/Y passed]
- Coverage: [percentage]

## Performance Metrics
[If performance requirements specified]

## Next Steps
[Any follow-up tasks or optimizations needed]
```

## Success Validation

### Definition of Done Checklist
- [ ] All specification requirements implemented
- [ ] Unit test coverage >90%
- [ ] Integration tests passing
- [ ] API endpoints functional and documented  
- [ ] Error handling comprehensive
- [ ] Performance benchmarks met (if specified)
- [ ] Code follows project conventions
- [ ] Documentation updated
- [ ] Git history clean with meaningful commits

---

This systematic implementation process ensures that detailed specifications become production-ready code with comprehensive validation and testing.