"""Recursive descent parser and evaluator for arithmetic expressions with +, -, *, /, ()."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    """Token types in expression grammar."""

    NUMBER = auto()
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    LPAREN = auto()
    RPAREN = auto()
    EOF = auto()


@dataclass
class Token:
    """A single token with type and value."""

    type: TokenType
    value: str


class Tokenizer:
    """Break expression string into tokens."""

    def __init__(self, expr: str) -> None:
        self.expr = expr
        self.pos = 0

    def tokenize(self) -> list[Token]:
        """Return all tokens from expression."""
        tokens: list[Token] = []
        while self.pos < len(self.expr):
            self._skip_whitespace()
            if self.pos >= len(self.expr):
                break

            ch = self.expr[self.pos]
            if ch.isdigit():
                tokens.append(self._read_number())
            elif ch == "+":
                tokens.append(Token(TokenType.PLUS, "+"))
                self.pos += 1
            elif ch == "-":
                tokens.append(Token(TokenType.MINUS, "-"))
                self.pos += 1
            elif ch == "*":
                tokens.append(Token(TokenType.MULTIPLY, "*"))
                self.pos += 1
            elif ch == "/":
                tokens.append(Token(TokenType.DIVIDE, "/"))
                self.pos += 1
            elif ch == "(":
                tokens.append(Token(TokenType.LPAREN, "("))
                self.pos += 1
            elif ch == ")":
                tokens.append(Token(TokenType.RPAREN, ")"))
                self.pos += 1
            else:
                raise ValueError(f"Unexpected character: {ch}")

        tokens.append(Token(TokenType.EOF, ""))
        return tokens

    def _skip_whitespace(self) -> None:
        while self.pos < len(self.expr) and self.expr[self.pos].isspace():
            self.pos += 1

    def _read_number(self) -> Token:
        start = self.pos
        while self.pos < len(self.expr) and self.expr[self.pos].isdigit():
            self.pos += 1
        if self.pos < len(self.expr) and self.expr[self.pos] == ".":
            self.pos += 1
            while self.pos < len(self.expr) and self.expr[self.pos].isdigit():
                self.pos += 1
        return Token(TokenType.NUMBER, self.expr[start : self.pos])


class Parser:
    """Recursive descent parser for expressions."""

    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    def parse(self) -> float:
        """Parse and evaluate expression."""
        result = self._parse_expr()
        if self.tokens[self.pos].type != TokenType.EOF:
            raise ValueError("Unexpected token after expression")
        return result

    def _parse_expr(self) -> float:
        result = self._parse_term()
        while self.tokens[self.pos].type in (TokenType.PLUS, TokenType.MINUS):
            op = self.tokens[self.pos].type
            self.pos += 1
            right = self._parse_term()
            if op == TokenType.PLUS:
                result += right
            else:
                result -= right
        return result

    def _parse_term(self) -> float:
        result = self._parse_factor()
        while self.tokens[self.pos].type in (TokenType.MULTIPLY, TokenType.DIVIDE):
            op = self.tokens[self.pos].type
            self.pos += 1
            right = self._parse_factor()
            if op == TokenType.MULTIPLY:
                result *= right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                result /= right
        return result

    def _parse_factor(self) -> float:
        token = self.tokens[self.pos]
        if token.type == TokenType.NUMBER:
            self.pos += 1
            return float(token.value)
        elif token.type == TokenType.LPAREN:
            self.pos += 1
            result = self._parse_expr()
            if self.tokens[self.pos].type != TokenType.RPAREN:
                raise ValueError("Expected closing parenthesis")
            self.pos += 1
            return result
        else:
            raise ValueError(f"Unexpected token: {token.value}")


def evaluate(expr: str) -> float:
    """Evaluate an arithmetic expression and return the result."""
    tokenizer = Tokenizer(expr)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens)
    return parser.parse()
