---
allowed-tools: Bash(pip:*), Bash(python:*), Bash(find:*), Grep, Read, Write, Glob, LS
description: Comprehensive security review and vulnerability assessment
argument-hint: [scope] | full security audit if no scope provided
---

# Comprehensive Security Audit

**Context**: You are using the CodeFarm methodology for comprehensive security auditing. This identifies and addresses security vulnerabilities while maintaining system functionality and following security best practices.

## Security Audit Scope

### 1. Audit Scope Definition
**Target**: ${ARGUMENTS:-"full-system"}

Available scopes:
- `code` - Source code security analysis
- `dependencies` - Third-party package vulnerabilities
- `config` - Configuration security review
- `api` - API security assessment
- `data` - Data protection and privacy
- `infrastructure` - Infrastructure security
- `full-system` - Complete security audit

### 2. Security Tools Setup
Install and configure security analysis tools using Bash tool:

```bash
# Install security scanning tools
pip install bandit safety semgrep

# Create security audit directory
mkdir -p security_audit
mkdir -p security_audit/reports
mkdir -p security_audit/fixes
```

### 3. Initial Security Assessment
Baseline security posture evaluation using Claude Code tools:

**Project analysis:**
- Use Glob tool with pattern "**/*.py" to find Python files for analysis
- Use Glob tool with patterns "**/*.env*", "**/*.conf", "**/*.ini" to find configuration files
- Use Glob tool with patterns "**/requirements*.txt", "**/package*.json" to find dependency files
- Use Bash tool with command: `git status --porcelain` to check uncommitted changes

## Source Code Security Analysis

### 4. Static Code Analysis
Create a custom security scanner:

```python
# security_scanner.py
import ast
import re
import os
from pathlib import Path
from typing import List, Dict, Any

class SecurityScanner:
    """Custom security vulnerability scanner"""
    
    def __init__(self):
        self.vulnerabilities = []
        self.security_patterns = {
            "hardcoded_secrets": [
                r"password\s*=\s*['\"][^'\"]+['\"]",
                r"api_key\s*=\s*['\"][^'\"]+['\"]",
                r"secret\s*=\s*['\"][^'\"]+['\"]",
                r"token\s*=\s*['\"][^'\"]+['\"]"
            ],
            "sql_injection": [
                r"execute\s*\([^)]*%[^)]*\)",
                r"query\s*\([^)]*\+[^)]*\)",
                r"\.format\s*\([^)]*\)" # String formatting in SQL
            ],
            "command_injection": [
                r"os\.system\s*\([^)]*\+",
                r"subprocess\.[^(]*\([^)]*shell\s*=\s*True",
                r"eval\s*\(",
                r"exec\s*\("
            ],
            "unsafe_deserialization": [
                r"pickle\.loads?\s*\(",
                r"yaml\.load\s*\(",
                r"json\.loads?\s*\([^)]*user"
            ],
            "weak_crypto": [
                r"md5\s*\(",
                r"sha1\s*\(",
                r"hashlib\.md5",
                r"hashlib\.sha1"
            ]
        }
    
    def scan_file(self, filepath: str) -> List[Dict[str, Any]]:
        """Scan a single file for security issues"""
        
        file_vulnerabilities = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\\n')
            
            # Check each security pattern
            for vuln_type, patterns in self.security_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        # Find line number
                        line_num = content[:match.start()].count('\\n') + 1
                        
                        file_vulnerabilities.append({
                            "type": vuln_type,
                            "file": filepath,
                            "line": line_num,
                            "code": lines[line_num - 1].strip() if line_num <= len(lines) else "",
                            "pattern": pattern,
                            "severity": self._get_severity(vuln_type)
                        })
            
            # Additional AST-based checks
            try:
                tree = ast.parse(content)
                ast_vulnerabilities = self._analyze_ast(tree, filepath)
                file_vulnerabilities.extend(ast_vulnerabilities)
            except SyntaxError:
                pass  # Skip files with syntax errors
        
        except Exception as e:
            print(f"Error scanning {filepath}: {e}")
        
        return file_vulnerabilities
    
    def _analyze_ast(self, tree: ast.AST, filepath: str) -> List[Dict[str, Any]]:
        """Analyze AST for security issues"""
        
        vulnerabilities = []
        
        for node in ast.walk(tree):
            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', 'compile']:
                        vulnerabilities.append({
                            "type": "dangerous_function",
                            "file": filepath,
                            "line": node.lineno,
                            "code": f"Use of {node.func.id}()",
                            "severity": "HIGH"
                        })
            
            # Check for assert statements (removed in production)
            elif isinstance(node, ast.Assert):
                vulnerabilities.append({
                    "type": "assert_statement",
                    "file": filepath,
                    "line": node.lineno,
                    "code": "Assert statement (removed in production -O)",
                    "severity": "MEDIUM"
                })
        
        return vulnerabilities
    
    def _get_severity(self, vuln_type: str) -> str:
        """Determine vulnerability severity"""
        
        severity_map = {
            "hardcoded_secrets": "CRITICAL",
            "sql_injection": "HIGH",
            "command_injection": "HIGH",
            "unsafe_deserialization": "HIGH",
            "weak_crypto": "MEDIUM"
        }
        
        return severity_map.get(vuln_type, "LOW")
    
    def scan_project(self, project_path: str = ".") -> List[Dict[str, Any]]:
        """Scan entire project for security vulnerabilities"""
        
        all_vulnerabilities = []
        
        # Find all Python files
        for py_file in Path(project_path).rglob("*.py"):
            if "venv" not in str(py_file) and ".git" not in str(py_file):
                file_vulns = self.scan_file(str(py_file))
                all_vulnerabilities.extend(file_vulns)
        
        return all_vulnerabilities
    
    def generate_report(self, vulnerabilities: List[Dict[str, Any]]) -> str:
        """Generate security audit report"""
        
        # Group by severity
        by_severity = {"CRITICAL": [], "HIGH": [], "MEDIUM": [], "LOW": []}
        for vuln in vulnerabilities:
            severity = vuln.get("severity", "LOW")
            by_severity[severity].append(vuln)
        
        report = f"""
# Security Audit Report
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- **Total vulnerabilities found**: {len(vulnerabilities)}
- **Critical**: {len(by_severity['CRITICAL'])}
- **High**: {len(by_severity['HIGH'])}
- **Medium**: {len(by_severity['MEDIUM'])}
- **Low**: {len(by_severity['LOW'])}

"""
        
        for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            if by_severity[severity]:
                report += f"\\n## {severity} Severity Issues\\n"
                for vuln in by_severity[severity][:10]:  # Limit to top 10 per severity
                    report += f"- **{vuln['type']}** in {vuln['file']}:{vuln['line']}\\n"
                    report += f"  Code: `{vuln['code']}`\\n\\n"
        
        return report

if __name__ == "__main__":
    import time
    
    scanner = SecurityScanner()
    vulnerabilities = scanner.scan_project()
    
    print(f"Security scan completed. Found {len(vulnerabilities)} potential issues.")
    
    if vulnerabilities:
        report = scanner.generate_report(vulnerabilities)
        
        # Save report
        with open("security_audit/custom_scan_report.md", "w") as f:
            f.write(report)
        
        print("Report saved to security_audit/custom_scan_report.md")
```

### 5. Bandit Security Analysis
Use Bash tool to run Bandit security scanner:

```bash
# Run Bandit security scanner
echo "=== Bandit Security Analysis ==="
python -m bandit -r src/ -f json -o security_audit/reports/bandit_report.json 2>/dev/null || echo "Bandit scan completed with issues"

# Generate human-readable report
python -m bandit -r src/ -f txt -o security_audit/reports/bandit_report.txt 2>/dev/null || echo "Bandit text report generated"

# High severity issues only
python -m bandit -r src/ -ll -i 2>/dev/null || echo "High severity issues scan completed"
```

### 6. Dependency Vulnerability Scan
Use Bash tool to check for vulnerable dependencies:

```bash
# Safety check for known vulnerabilities
echo "=== Dependency Security Check ==="
python -m safety check --json --output security_audit/reports/safety_report.json 2>/dev/null || echo "Safety scan completed"

# Generate readable safety report
python -m safety check --output security_audit/reports/safety_report.txt 2>/dev/null || echo "Safety text report generated"

# Audit specific requirements file
if [ -f "requirements.txt" ]; then
    python -m safety check -r requirements.txt 2>/dev/null || echo "Requirements security check completed"
fi
```

## Configuration Security Review

### 7. Environment and Configuration Security
Use Read and Grep tools to review configuration files for security issues:

**Configuration file analysis:**
- Use Glob tool with patterns "**/.env*", "**/*.conf", "**/*.ini", "**/*.yaml", "**/*.yml", "**/*.json" to find config files
- Use Grep tool to search for patterns like "password.*=", "secret.*=", "api_key.*=" in configuration files
- Use Grep tool to search for "debug.*=.*true", "ssl_verify.*=.*false" for insecure settings

### 8. API Security Assessment
Use Read and Grep tools to review API endpoints for security vulnerabilities:

**API endpoint analysis:**
- Use Glob tool with patterns "**/api.py", "**/routes.py", "**/views.py", "**/endpoints.py" to find API files
- Use Grep tool to search for route decorators without authentication
- Use Grep tool to search for "CORS.*allow_origins.*\*" for CORS misconfigurations
- Use Grep tool to search for "request.args.get", "request.form.get" for input validation issues

## Security Fixes and Recommendations

### 9. Automated Security Fixes
Create security fixer script to implement common security fixes:

```python
# security_fixer.py
import re
import os
from pathlib import Path
from typing import List, Dict, Any

class SecurityFixer:
    """Automated security issue fixes"""
    
    def __init__(self):
        self.fixes_applied = []
    
    def fix_hardcoded_secrets(self, filepath: str) -> bool:
        """Replace hardcoded secrets with environment variables"""
        
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            original_content = content
            
            # Replace common hardcoded patterns
            replacements = {
                r'password\s*=\s*[\'"]([^\'"]+)[\'"]': 'password = os.getenv("PASSWORD", "")',
                r'api_key\s*=\s*[\'"]([^\'"]+)[\'"]': 'api_key = os.getenv("API_KEY", "")',
                r'secret\s*=\s*[\'"]([^\'"]+)[\'"]': 'secret = os.getenv("SECRET", "")',
                r'token\s*=\s*[\'"]([^\'"]+)[\'"]': 'token = os.getenv("TOKEN", "")'
            }
            
            for pattern, replacement in replacements.items():
                content = re.sub(pattern, replacement, content)
            
            # Add import if needed and changes were made
            if content != original_content:
                if 'import os' not in content:
                    content = 'import os\\n' + content
                
                # Create backup
                backup_path = f"{filepath}.backup"
                with open(backup_path, 'w') as f:
                    f.write(original_content)
                
                # Write fixed content
                with open(filepath, 'w') as f:
                    f.write(content)
                
                self.fixes_applied.append({
                    "type": "hardcoded_secrets",
                    "file": filepath,
                    "backup": backup_path
                })
                
                return True
        
        except Exception as e:
            print(f"Error fixing {filepath}: {e}")
        
        return False

if __name__ == "__main__":
    fixer = SecurityFixer()
    
    # Apply automated fixes to Python files
    for py_file in Path(".").rglob("*.py"):
        if "venv" not in str(py_file) and ".git" not in str(py_file):
            fixer.fix_hardcoded_secrets(str(py_file))
    
    print(f"Applied {len(fixer.fixes_applied)} automated security fixes")
    for fix in fixer.fixes_applied:
        print(f"  {fix['type']}: {fix['file']}")
```

### 10. Security Best Practices Implementation
Generate security configuration templates:

```python
# security_config_generator.py
def generate_security_configs():
    """Generate security configuration files"""
    
    # .env.example with secure defaults
    env_example = """
# Security Configuration
DEBUG=False
SECRET_KEY=${GENERATE_RANDOM_SECRET}
DATABASE_URL=${DATABASE_CONNECTION_STRING}

# API Keys (set in production)
API_KEY=${YOUR_API_KEY}
JWT_SECRET=${GENERATE_JWT_SECRET}

# Security Settings
ALLOWED_HOSTS=localhost,127.0.0.1
CORS_ORIGINS=https://yourdomain.com
SESSION_COOKIE_SECURE=True
CSRF_COOKIE_SECURE=True

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60
MAX_CONTENT_LENGTH=16777216  # 16MB
"""
    
    # Write configuration files
    configs = {'.env.example': env_example}
    
    for filepath, content in configs.items():
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(content.strip())
    
    return configs.keys()

if __name__ == "__main__":
    created_files = generate_security_configs()
    print("Security configuration files created:")
    for file in created_files:
        print(f"  {file}")
```

## Security Audit Report

### 11. Comprehensive Security Report
Generate detailed security audit report using collected data from scans.

### 12. Success Criteria
Security audit complete when:

- [ ] All critical vulnerabilities addressed
- [ ] High-priority issues have remediation plan
- [ ] Security configuration implemented
- [ ] Dependencies updated and secured
- [ ] Security headers configured
- [ ] Authentication and authorization reviewed
- [ ] Input validation implemented
- [ ] Security monitoring established
- [ ] Documentation updated
- [ ] Security training plan created

---

This comprehensive security audit ensures system protection against common vulnerabilities while establishing ongoing security practices.