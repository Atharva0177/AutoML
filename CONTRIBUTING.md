# Contributing to AutoML

Thank you for your interest in contributing to AutoML! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and constructive in all interactions.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in Issues
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs. actual behavior
   - Environment details (OS, Python version, etc.)

### Suggesting Features

1. Check if the feature has been suggested
2. Create a new issue with:
   - Clear description of the feature
   - Use cases and benefits
   - Proposed implementation (optional)

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following our code style
4. Add/update tests as needed
5. Ensure all tests pass (`pytest`)
6. Commit with descriptive messages
7. Push to your fork
8. Create a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/automl.git
cd automl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt
pip install -e .

# Install pre-commit hooks
pre-commit install
```

## Code Style

- Follow PEP 8
- Use Black for formatting (line length: 88)
- Use type hints where appropriate
- Write docstrings for all public functions/classes
- Keep functions focused and testable

## Testing

- Write tests for all new features
- Maintain >80% code coverage
- Run tests before submitting PR:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=automl --cov-report=html

# Run specific tests
pytest tests/unit/test_data_loaders.py
```

## Commit Messages

Follow conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Example:
```
feat(data): add Excel file loader

- Implement ExcelLoader class
- Add support for multiple sheets
- Include unit tests

Closes #42
```

## Documentation

- Update documentation for new features
- Add docstrings to new functions/classes
- Update README.md if needed

## Questions?

Feel free to open an issue for questions or join our discussions.

Thank you for contributing! 🎉
