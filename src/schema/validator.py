import json
import logging
from typing import Dict, Any, Tuple, List
from jsonschema import Draft7Validator, ValidationError

logger = logging.getLogger("synthmed.schema")

class SchemaValidator:
    """Validate JSON records against clinical metadata schema."""
    
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
        self.validator = Draft7Validator(schema)
        self.total_checked = 0
        self.total_valid = 0
    
    def validate(self, record: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a record against the schema.
        Returns (is_valid, list_of_errors).
        """
        self.total_checked += 1
        errors = list(self.validator.iter_errors(record))
        
        if not errors:
            self.total_valid += 1
            return True, []
        
        error_messages = [
            f"{' -> '.join(str(p) for p in err.path)}: {err.message}"
            for err in errors
        ]
        
        return False, error_messages
    
    def validate_batch(
        self, records: List[Dict[str, Any]]
    ) -> List[Tuple[bool, List[str]]]:
        """Validate multiple records."""
        return [self.validate(record) for record in records]
    
    @property
    def validity_rate(self) -> float:
        """Get the rate of valid records."""
        if self.total_checked == 0:
            return 0.0
        return self.total_valid / self.total_checked
    
    def reset_stats(self):
        """Reset validation statistics."""
        self.total_checked = 0
        self.total_valid = 0