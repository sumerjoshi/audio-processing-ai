# Contributing to Audio Processing AI

Thank you for your interest in contributing to Audio Processing AI! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/yourusername/audio-processing-ai.git
cd audio-processing-ai
```

2. Create and activate a virtual environment:
```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows

# Using pip
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

3. Install the package in development mode with dev dependencies:
```bash
# Using uv
uv pip install -e ".[dev]"

# Using pip
pip install -e ".[dev]"
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

## Code Style

This project follows Python best practices and uses several tools to ensure code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pytest**: Testing

### Running Code Quality Checks

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type check
mypy src/audio_processing_ai/

# Run tests
pytest tests/ -v
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src/audio_processing_ai --cov-report=html

# Run specific test file
pytest tests/test_predict.py -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Test files should start with `test_`
- Use descriptive test names
- Aim for good test coverage

## Pull Request Process

1. Create a feature branch from `main-copy`:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and ensure they follow the code style guidelines

3. Run tests and quality checks:
```bash
pytest tests/ -v
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/audio_processing_ai/
```

4. Commit your changes with a descriptive message:
```bash
git add .
git commit -m "feat: add new feature description"
```

5. Push to your fork:
```bash
git push origin feature/your-feature-name
```

6. Create a Pull Request with:
   - Clear description of changes
   - Reference to any related issues
   - Screenshots if applicable

## Commit Message Format

Use conventional commits format:

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

## Issue Reporting

When reporting issues, please include:

- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages or logs

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.
