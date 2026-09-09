"""JSON schema validation with detailed error reporting."""

from __future__ import annotations

from typing import Any

type Schema = dict[str, Any]
type JSON = None | bool | int | float | str | list[Any] | dict[str, Any]


class ValidationError(Exception):
    """Raised when validation fails; includes path to error and details."""

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")


class SchemaValidator:
    """Validates JSON data against a schema."""

    def __init__(self, schema: Schema) -> None:
        self.schema = schema

    def validate(self, data: JSON) -> None:
        """Raise ValidationError if data does not match schema."""
        self._validate_impl(data, self.schema, "$")

    def _validate_impl(self, data: JSON, schema: Schema, path: str) -> None:
        """Internal validation with path tracking."""
        schema_type = schema.get("type")

        if schema_type == "null":
            if data is not None:
                raise ValidationError(path, f"expected null, got {type(data).__name__}")

        elif schema_type == "boolean":
            if not isinstance(data, bool):
                raise ValidationError(path, f"expected boolean, got {type(data).__name__}")

        elif schema_type == "number":
            if not isinstance(data, (int, float)) or isinstance(data, bool):
                raise ValidationError(path, f"expected number, got {type(data).__name__}")
            if "minimum" in schema and data < schema["minimum"]:
                raise ValidationError(
                    path, f"value {data} is less than minimum {schema['minimum']}"
                )
            if "maximum" in schema and data > schema["maximum"]:
                raise ValidationError(
                    path, f"value {data} is greater than maximum {schema['maximum']}"
                )

        elif schema_type == "integer":
            if not isinstance(data, int) or isinstance(data, bool):
                raise ValidationError(path, f"expected integer, got {type(data).__name__}")
            if "minimum" in schema and data < schema["minimum"]:
                raise ValidationError(
                    path, f"value {data} is less than minimum {schema['minimum']}"
                )
            if "maximum" in schema and data > schema["maximum"]:
                raise ValidationError(
                    path, f"value {data} is greater than maximum {schema['maximum']}"
                )

        elif schema_type == "string":
            if not isinstance(data, str):
                raise ValidationError(path, f"expected string, got {type(data).__name__}")
            if "minLength" in schema and len(data) < schema["minLength"]:
                raise ValidationError(
                    path,
                    f"string length {len(data)} is less than minLength {schema['minLength']}",
                )
            if "maxLength" in schema and len(data) > schema["maxLength"]:
                raise ValidationError(
                    path,
                    f"string length {len(data)} is greater than maxLength {schema['maxLength']}",
                )

        elif schema_type == "array":
            if not isinstance(data, list):
                raise ValidationError(path, f"expected array, got {type(data).__name__}")
            items_schema = schema.get("items")
            if items_schema:
                for i, item in enumerate(data):
                    self._validate_impl(item, items_schema, f"{path}[{i}]")

        elif schema_type == "object":
            if not isinstance(data, dict):
                raise ValidationError(path, f"expected object, got {type(data).__name__}")
            required = schema.get("required", [])
            for prop in required:
                if prop not in data:
                    raise ValidationError(path, f"missing required property: {prop}")
            properties = schema.get("properties", {})
            for prop, prop_schema in properties.items():
                if prop in data:
                    self._validate_impl(data[prop], prop_schema, f"{path}.{prop}")
            allow_additional = schema.get("additionalProperties", True)
            if not allow_additional:
                for prop in data:
                    if prop not in properties:
                        raise ValidationError(
                            path, f"unexpected property: {prop} (additionalProperties: false)"
                        )
