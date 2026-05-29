import json
import re
import logging
from typing import Dict, Any, List, Tuple, Optional
from jsonschema import Draft7Validator

logger = logging.getLogger("synthmed.schema")

class JSONRepairer:
    """
    Multi-strategy JSON repairer for clinical metadata.
    Implements layered repair approach:
    1. Structural repair (add missing fields)
    2. Type repair (fix type mismatches)
    3. Constraint repair (fix value constraints)
    """
    
    def __init__(self, schema: Dict[str, Any], max_iterations: int = 3):
        self.schema = schema
        self.max_iterations = max_iterations
        self.validator = Draft7Validator(schema)
        self.repair_stats = {"attempted": 0, "successful": 0}
    
    def repair(self, record: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """
        Attempt to repair a record to conform to schema.
        Returns (repaired_record, success).
        """
        self.repair_stats["attempted"] += 1
        repaired = record.copy()
        
        for iteration in range(self.max_iterations):
            errors = list(self.validator.iter_errors(repaired))
            
            if not errors:
                self.repair_stats["successful"] += 1
                return repaired, True
            
            # Apply repair strategies sequentially
            repaired = self._repair_structural(repaired, errors)
            repaired = self._repair_types(repaired, errors)
            repaired = self._repair_constraints(repaired, errors)
            
            logger.debug(f"Iteration {iteration + 1}: {len(errors)} errors remaining")
        
        # Final check
        errors = list(self.validator.iter_errors(repaired))
        if not errors:
            self.repair_stats["successful"] += 1
            return repaired, True
        
        return repaired, False
    
    def _repair_structural(
        self, record: Dict[str, Any], errors: List[Any]
    ) -> Dict[str, Any]:
        """Add missing required fields with defaults."""
        repaired = record.copy()
        required = self.schema.get("required", [])
        
        for field in required:
            if field not in repaired:
                default_value = self._get_default_value(field)
                repaired[field] = default_value
                logger.debug(f"Added missing field: {field} = {default_value}")
        
        return repaired
    
    def _repair_types(
        self, record: Dict[str, Any], errors: List[Any]
    ) -> Dict[str, Any]:
        """Fix type mismatches."""
        repaired = record.copy()
        properties = self.schema.get("properties", {})
        
        for field, prop_schema in properties.items():
            if field not in repaired:
                continue
            
            expected_type = prop_schema.get("type")
            value = repaired[field]
            
            # Type coercion
            if expected_type == "integer" and isinstance(value, (str, float)):
                try:
                    repaired[field] = int(float(value))
                    logger.debug(f"Coerced {field} to integer: {repaired[field]}")
                except (ValueError, TypeError):
                    repaired[field] = self._get_default_value(field)
            
            elif expected_type == "number" and isinstance(value, str):
                try:
                    repaired[field] = float(value)
                except ValueError:
                    repaired[field] = self._get_default_value(field)
            
            elif expected_type == "boolean" and isinstance(value, str):
                repaired[field] = value.lower() in ("true", "1", "yes")
            
            elif expected_type == "string" and not isinstance(value, str):
                repaired[field] = str(value)
        
        return repaired
    
    def _repair_constraints(
        self, record: Dict[str, Any], errors: List[Any]
    ) -> Dict[str, Any]:
        """Fix value constraint violations."""
        repaired = record.copy()
        properties = self.schema.get("properties", {})
        
        for field, prop_schema in properties.items():
            if field not in repaired:
                continue
            
            value = repaired[field]
            
            # Integer constraints
            if prop_schema.get("type") == "integer":
                minimum = prop_schema.get("minimum")
                maximum = prop_schema.get("maximum")
                
                if minimum is not None and isinstance(value, (int, float)) and value < minimum:
                    repaired[field] = minimum
                if maximum is not None and isinstance(value, (int, float)) and value > maximum:
                    repaired[field] = maximum
            
            # Enum constraints
            enum_values = prop_schema.get("enum")
            if enum_values and value not in enum_values:
                repaired[field] = enum_values[0]
                logger.debug(f"Fixed enum {field}: {value} -> {repaired[field]}")
            
            # Pattern constraints
            pattern = prop_schema.get("pattern")
            if pattern and isinstance(value, str):
                if not re.match(pattern, value):
                    repaired[field] = self._generate_pattern_match(pattern)
                    logger.debug(f"Fixed pattern {field}: {value} -> {repaired[field]}")
        
        return repaired
    
    def _get_default_value(self, field: str) -> Any:
        """Get sensible default value for a field."""
        defaults = {
            "patient_id": "P00000",
            "age": 50,
            "sex": "M",
            "dr_grade": 0,
            "image_quality": 0.5,
            "left_eye": True,
            "anatomical_findings": {
                "microaneurysms": "none",
                "hemorrhages": "none",
                "exudates": "none",
            }
        }
        return defaults.get(field, None)
    
    def _generate_pattern_match(self, pattern: str) -> str:
        """Generate a string matching regex pattern."""
        if pattern == "^P[0-9]{5}$":
            import random
            return f"P{random.randint(0, 99999):05d}"
        return ""
    
    @property
    def repair_success_rate(self) -> float:
        """Get repair success rate."""
        if self.repair_stats["attempted"] == 0:
            return 0.0
        return self.repair_stats["successful"] / self.repair_stats["attempted"]
    
    def reset_stats(self):
        """Reset repair statistics."""
        self.repair_stats = {"attempted": 0, "successful": 0}