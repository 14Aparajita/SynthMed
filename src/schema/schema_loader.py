import json
from pathlib import Path
from jsonschema import Draft7Validator
from typing import Dict, Any

def load_schema(schema_path: str) -> Dict[str, Any]:
    """Load and validate a JSON schema."""
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    
    # Verify it's a valid JSON Schema
    Draft7Validator.check_schema(schema)
    
    return schema