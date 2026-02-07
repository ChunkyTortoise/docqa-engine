# Python Programming Guide

## Variables and Data Types

Python is a dynamically typed language, meaning you do not need to declare variable types explicitly. The interpreter infers the type at runtime based on the assigned value.

The core data types include integers (`int`), floating-point numbers (`float`), strings (`str`), and booleans (`bool`). Python also provides complex numbers with the `complex` type. Strings can be defined with single quotes, double quotes, or triple quotes for multiline text. Type conversion is straightforward using built-in functions like `int()`, `float()`, and `str()`.

## Control Flow

Conditional logic uses `if`, `elif`, and `else` blocks. Python relies on indentation rather than braces to define code blocks, which enforces readable formatting. Comparison operators include `==`, `!=`, `<`, `>`, `<=`, and `>=`. Logical operators `and`, `or`, and `not` combine conditions.

Loops come in two forms. The `for` loop iterates over sequences such as lists, tuples, dictionaries, and ranges. The `while` loop repeats as long as a condition remains true. The `break` statement exits a loop early, `continue` skips to the next iteration, and `else` on a loop runs when the loop completes without hitting `break`.

## Functions

Functions are defined with the `def` keyword. They support positional arguments, keyword arguments, default values, and variable-length argument lists using `*args` and `**kwargs`. Functions are first-class objects, so you can pass them as arguments, return them from other functions, and assign them to variables.

Lambda functions provide a concise syntax for simple, single-expression functions. Decorators wrap functions to add behavior without modifying the original function body. Common built-in decorators include `@staticmethod`, `@classmethod`, and `@property`.

## Classes and Object-Oriented Programming

Classes are defined with the `class` keyword. The `__init__` method serves as the constructor. Instance attributes are set on `self`, and class attributes are shared across all instances. Python supports single and multiple inheritance. Method resolution order determines which parent method is called when there are conflicts.

Special methods (dunder methods) like `__str__`, `__repr__`, `__len__`, and `__eq__` customize how objects behave with built-in operations and functions.

## List Comprehensions and Generators

List comprehensions provide a compact way to create lists from existing iterables. The syntax is `[expression for item in iterable if condition]`. Dictionary and set comprehensions follow similar patterns.

Generator expressions use parentheses instead of brackets and produce values lazily, one at a time, which is memory-efficient for large datasets. Generator functions use `yield` to produce a series of values over multiple calls.

## Error Handling

Python uses `try`, `except`, `else`, and `finally` blocks for exception handling. You can catch specific exception types or use a bare `except` as a fallback. The `raise` keyword throws exceptions explicitly. Custom exceptions inherit from `Exception` or its subclasses.

Context managers, defined with `__enter__` and `__exit__` methods or the `@contextmanager` decorator, ensure resources like file handles and database connections are properly cleaned up, even when exceptions occur.
