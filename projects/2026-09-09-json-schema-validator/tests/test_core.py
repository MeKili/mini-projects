"""Tests for JSON schema validation."""

import pytest

from json_schema_validator.core import SchemaValidator, ValidationError


def test_type_null_valid() -> None:
    validator = SchemaValidator({"type": "null"})
    validator.validate(None)


def test_type_null_invalid() -> None:
    validator = SchemaValidator({"type": "null"})
    with pytest.raises(ValidationError) as exc_info:
        validator.validate(0)
    assert "expected null" in str(exc_info.value)


def test_type_boolean_valid() -> None:
    validator = SchemaValidator({"type": "boolean"})
    validator.validate(True)
    validator.validate(False)


def test_type_boolean_invalid() -> None:
    validator = SchemaValidator({"type": "boolean"})
    with pytest.raises(ValidationError) as exc_info:
        validator.validate(1)
    assert "expected boolean" in str(exc_info.value)


def test_type_number_valid() -> None:
    validator = SchemaValidator({"type": "number"})
    validator.validate(42)
    validator.validate(3.14)
    validator.validate(-1.5)


def test_type_number_invalid() -> None:
    validator = SchemaValidator({"type": "number"})
    with pytest.raises(ValidationError):
        validator.validate("not a number")


def test_type_integer_valid() -> None:
    validator = SchemaValidator({"type": "integer"})
    validator.validate(42)
    validator.validate(-1)


def test_type_integer_rejects_float() -> None:
    validator = SchemaValidator({"type": "integer"})
    with pytest.raises(ValidationError) as exc_info:
        validator.validate(3.14)
    assert "expected integer" in str(exc_info.value)


def test_type_string_valid() -> None:
    validator = SchemaValidator({"type": "string"})
    validator.validate("hello")
    validator.validate("")


def test_type_string_invalid() -> None:
    validator = SchemaValidator({"type": "string"})
    with pytest.raises(ValidationError) as exc_info:
        validator.validate(123)
    assert "expected string" in str(exc_info.value)


def test_string_min_length() -> None:
    validator = SchemaValidator({"type": "string", "minLength": 3})
    validator.validate("hello")
    with pytest.raises(ValidationError) as exc_info:
        validator.validate("hi")
    assert "minLength" in str(exc_info.value)


def test_string_max_length() -> None:
    validator = SchemaValidator({"type": "string", "maxLength": 3})
    validator.validate("hi")
    with pytest.raises(ValidationError) as exc_info:
        validator.validate("hello")
    assert "maxLength" in str(exc_info.value)


def test_number_minimum() -> None:
    validator = SchemaValidator({"type": "number", "minimum": 0.0})
    validator.validate(5.0)
    with pytest.raises(ValidationError) as exc_info:
        validator.validate(-1.0)
    assert "minimum" in str(exc_info.value)


def test_number_maximum() -> None:
    validator = SchemaValidator({"type": "number", "maximum": 100.0})
    validator.validate(50.0)
    with pytest.raises(ValidationError) as exc_info:
        validator.validate(101.0)
    assert "maximum" in str(exc_info.value)


def test_integer_bounds() -> None:
    validator = SchemaValidator({"type": "integer", "minimum": 1, "maximum": 10})
    validator.validate(5)
    with pytest.raises(ValidationError):
        validator.validate(0)
    with pytest.raises(ValidationError):
        validator.validate(11)


def test_type_array_valid() -> None:
    validator = SchemaValidator({"type": "array"})
    validator.validate([])
    validator.validate([1, "two", None])


def test_type_array_invalid() -> None:
    validator = SchemaValidator({"type": "array"})
    with pytest.raises(ValidationError) as exc_info:
        validator.validate("not an array")
    assert "expected array" in str(exc_info.value)


def test_array_items_schema() -> None:
    validator = SchemaValidator({"type": "array", "items": {"type": "integer"}})
    validator.validate([1, 2, 3])
    with pytest.raises(ValidationError) as exc_info:
        validator.validate([1, "two", 3])
    assert "[1]" in str(exc_info.value)


def test_type_object_valid() -> None:
    validator = SchemaValidator({"type": "object"})
    validator.validate({})
    validator.validate({"a": 1, "b": "two"})


def test_type_object_invalid() -> None:
    validator = SchemaValidator({"type": "object"})
    with pytest.raises(ValidationError) as exc_info:
        validator.validate([1, 2, 3])
    assert "expected object" in str(exc_info.value)


def test_object_required_properties() -> None:
    validator = SchemaValidator(
        {
            "type": "object",
            "required": ["name", "age"],
        }
    )
    validator.validate({"name": "Alice", "age": 30})
    with pytest.raises(ValidationError) as exc_info:
        validator.validate({"name": "Bob"})
    assert "missing required property" in str(exc_info.value)


def test_object_property_schemas() -> None:
    validator = SchemaValidator(
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0},
            },
        }
    )
    validator.validate({"name": "Alice", "age": 30})
    with pytest.raises(ValidationError):
        validator.validate({"name": "Bob", "age": -5})


def test_object_additional_properties_allowed() -> None:
    validator = SchemaValidator(
        {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": True,
        }
    )
    validator.validate({"name": "Alice", "extra": "data"})


def test_object_additional_properties_forbidden() -> None:
    validator = SchemaValidator(
        {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": False,
        }
    )
    validator.validate({"name": "Alice"})
    with pytest.raises(ValidationError) as exc_info:
        validator.validate({"name": "Bob", "extra": "data"})
    assert "unexpected property" in str(exc_info.value)


def test_nested_objects() -> None:
    validator = SchemaValidator(
        {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                }
            },
        }
    )
    validator.validate({"user": {"name": "Alice"}})
    with pytest.raises(ValidationError) as exc_info:
        validator.validate({"user": {"age": 30}})
    assert "user" in str(exc_info.value)


def test_nested_arrays() -> None:
    validator = SchemaValidator(
        {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number"},
            },
        }
    )
    validator.validate([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValidationError) as exc_info:
        validator.validate([[1.0, "two"], [3.0, 4.0]])
    assert "[0][1]" in str(exc_info.value)


def test_error_path_tracking() -> None:
    validator = SchemaValidator(
        {
            "type": "object",
            "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
        }
    )
    with pytest.raises(ValidationError) as exc_info:
        validator.validate({"tags": ["good", 42]})
    assert "tags[1]" in str(exc_info.value)


def test_complex_user_schema() -> None:
    schema = {
        "type": "object",
        "required": ["id", "email"],
        "properties": {
            "id": {"type": "integer", "minimum": 1},
            "email": {"type": "string", "minLength": 5},
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0, "maximum": 150},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": False,
    }
    validator = SchemaValidator(schema)

    valid_user = {
        "id": 1,
        "email": "alice@example.com",
        "name": "Alice",
        "age": 30,
        "tags": ["developer", "python"],
    }
    validator.validate(valid_user)

    with pytest.raises(ValidationError):
        validator.validate({"id": 0, "email": "bob@example.com"})

    with pytest.raises(ValidationError):
        validator.validate({"id": 1, "email": "x"})

    with pytest.raises(ValidationError):
        validator.validate({"id": 1, "email": "test@example.com", "extra": True})
