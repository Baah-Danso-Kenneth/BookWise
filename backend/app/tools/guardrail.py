import re
import logging
from typing import List, Dict, Any

from app.tools.base import MCPToolServer, MCPToolDescriptor, MCPToolResult, MCPToolStatus


class GuardrailServer(MCPToolServer):
    """
    MCP-compliant guardrail tool.
    Scans content for violations and injects disclaimers.
    """
    
    # Patterns to block
    INAPPROPRIATE_PATTERNS = [
        r"hate\s+speech",
        r"violence\s+against",
        r"kill\s+yourself",
        r"suicide",
        r"self-harm",
        r"racist",
        r"sexist",
    ]
    
    # Patterns that trigger warnings
    BIAS_PATTERNS = [
        r"all\s+[a-z]+s\s+are",
        r"always\s+wrong",
        r"never\s+right",
        r"everyone\s+knows",
    ]
    
    # Overconfident claims to flag
    OVERCONFIDENT_PATTERNS = [
        r"guarantee",
        r"best\s+book\s+ever",
        r"you\s+will\s+love",
        r"100%",
        r"perfect\s+for\s+everyone",
    ]
    
    REQUIRED_DISCLAIMER = """
DISCLAIMER: Book recommendations are based on algorithmic analysis and web search.
Taste is deeply personal. Your experience may vary.
Always preview books before purchasing to ensure they match your preferences.
"""
    
    def _register(self) -> MCPToolDescriptor:
        return MCPToolDescriptor(
            name="Guardrail",
            version="1.0.0",
            description="Scans content for safety violations, bias, and injects disclaimers",
            input_schema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Content to scan"},
                    "strict_mode": {"type": "boolean", "default": True}
                },
                "required": ["content"]
            }
        )
    
    def execute(self, content: str, strict_mode: bool = True) -> MCPToolResult:
        """Scan content and return sanitized version"""
        logging.info("Guardrail: scanning content")
        
        violations = []
        warnings = []
        disclaimer_added = False
        
        # Check for inappropriate content
        for pattern in self.INAPPROPRIATE_PATTERNS:
            if re.search(pattern, content.lower()):
                violations.append({
                    "type": "INAPPROPRIATE_CONTENT",
                    "pattern": pattern,
                    "action": "BLOCKED"
                })
        
        # Check for bias
        for pattern in self.BIAS_PATTERNS:
            if re.search(pattern, content.lower()):
                warnings.append({
                    "type": "BIAS_INDICATOR",
                    "pattern": pattern,
                    "action": "FLAGGED"
                })
        
        # Check for overconfident claims
        for pattern in self.OVERCONFIDENT_PATTERNS:
            if re.search(pattern, content.lower()):
                warnings.append({
                    "type": "OVERCONFIDENT_CLAIM",
                    "pattern": pattern,
                    "action": "MODIFIED"
                })
                # Remove the overconfident phrase
                content = re.sub(pattern, "[recommended]", content, flags=re.IGNORECASE)
        
        # Determine if content passes
        hard_violations = [v for v in violations if v["type"] != "WARNING"]
        
        if strict_mode:
            passed = len(hard_violations) == 0
        else:
            passed = True
        
        # Always add disclaimer
        sanitized_content = f"{content}\n\n{self.REQUIRED_DISCLAIMER}"
        disclaimer_added = True
        
        status = MCPToolStatus.SUCCESS if passed else MCPToolStatus.WARNING
        
        logging.info(f"Guardrail: passed={passed}, violations={len(violations)}, warnings={len(warnings)}")
        
        return MCPToolResult(
            tool_name="Guardrail",
            status=status,
            content={
                "passed": passed,
                "sanitized_content": sanitized_content,
                "disclaimer_added": disclaimer_added,
                "violations": violations,
                "warnings": warnings
            },
            metadata={
                "original_length": len(content),
                "sanitized_length": len(sanitized_content),
                "strict_mode": strict_mode
            }
        )