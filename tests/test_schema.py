"""Test schema validation and repair."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.schema import SchemaValidator, JSONRepairer, load_schema

@pytest.fixture
def schema():
    return load_schema("config/schema/clinical_metadata.json")

@pytest.fixture
def validator(schema):
    return SchemaValidator(schema)

@pytest.fixture
def repairer(schema):
    return JSONRepairer(schema)

def test_valid_record(validator):
    record = {
        "patient_id": "P12345",
        "age": 55,
        "sex": "F",
        "dr_grade": 2,
        "image_quality": 0.85,
        "left_eye": True,
        "anatomical_findings": {
            "microaneurysms": "moderate",
            "hemorrhages": "few",
            "exudates": "few"
        }
    }
    is_valid, errors = validator.validate(record)
    assert is_valid
    assert len(errors) == 0

def test_invalid_record_missing_field(validator):
    record = {
        "patient_id": "P12345",
        "age": 55,
        # Missing sex
        "dr_grade": 2,
        "image_quality": 0.85,
        "left_eye": True,
    }
    is_valid, errors = validator.validate(record)
    assert not is_valid

def test_repair_missing_field(repairer):
    record = {
        "patient_id": "P12345",
        "age": 55,
        "dr_grade": 2,
        "image_quality": 0.85,
        "left_eye": True,
    }
    repaired, success = repairer.repair(record)
    assert "sex" in repaired

def test_repair_type_coercion(repairer):
    record = {
        "patient_id": "P12345",
        "age": "55",  # String instead of int
        "sex": "F",
        "dr_grade": 2,
        "image_quality": 0.85,
        "left_eye": True,
        "anatomical_findings": {
            "microaneurysms": "none",
            "hemorrhages": "none",
            "exudates": "none"
        }
    }
    repaired, success = repairer.repair(record)
    assert isinstance(repaired["age"], int)
    assert repaired["age"] == 55

def test_validity_rate(validator):
    records = [
        {
            "patient_id": "P00001", "age": 50, "sex": "M",
            "dr_grade": 0, "image_quality": 0.7, "left_eye": True,
            "anatomical_findings": {
                "microaneurysms": "none", "hemorrhages": "none", "exudates": "none"
            }
        },
        {"patient_id": "P00002"},  # Invalid
    ]
    validator.validate_batch(records)
    assert validator.validity_rate == 0.5