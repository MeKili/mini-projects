"""json-schema-validator — a simple, strict JSON schema validator.

Validates JSON data (dicts, lists, strings, numbers, booleans, null)
against a schema with support for types, required properties, length
constraints, numeric bounds, array items, and object structure validation.
"""

from json_schema_validator.core import SchemaValidator, ValidationError

__all__ = ["SchemaValidator", "ValidationError"]
__version__ = "0.1.0"
