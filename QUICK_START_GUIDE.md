# AutoML Project - Quick Start Guide

**For**: Development Team  
**Purpose**: Get up and running quickly with the project  
**Last Updated**: 2026-02-14

---

## 📋 Prerequisites Checklist

Before you begin, ensure you have:

- [ ] Python 3.10 or 3.11 installed
- [ ] Git installed and configured
- [ ] GitHub account with access to the repository (once created)
- [ ] Code editor (VS Code recommended)
- [ ] At least 16GB RAM (32GB recommended for DL work)
- [ ] 20GB free disk space
- [ ] (Optional) NVIDIA GPU with CUDA 11.8+ for deep learning

---

## 🚀 Quick Setup (5 Minutes)

### Step 1: Clone the Repository

```bash
# Replace with actual repo URL once created
git clone https://github.com/your-org/automl.git
cd automl
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install development dependencies
pip install -r requirements-dev.txt

# Install package in editable mode
pip install -e .
```

### Step 4: Verify Installation

```bash
# Run tests
pytest tests/ -v

# Should see: "X passed in Y seconds"
```

### Step 5: Setup Pre-commit Hooks

```bash
pre-commit install
```

✅ **You're ready to code!**

---

## 📂 Project Structure Overview

```
automl/
├── automl/          # Main source code
│   ├── data/        # Data loading & validation
│   ├── eda/         # Exploratory data analysis
│   ├── preprocessing/  # Data preprocessing
│   ├── models/      # ML/DL models
│   ├── training/    # Training pipeline
│   └── ...
├── tests/           # All tests
├── docs/            # Documentation
├── examples/        # Example scripts & notebooks
└── scripts/         # Utility scripts
```

**Key Files**:
- `automl/core/automl.py` - Main AutoML class
- `automl/config/settings.yaml` - Configuration
- `requirements.txt` - Production dependencies
- `requirements-dev.txt` - Development dependencies

---

## 🛠️ Common Development Tasks

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=automl --cov-report=html

# Run specific test file
pytest tests/unit/test_data_loaders.py

# Run tests in parallel (faster)
pytest -n auto
```

### Code Formatting & Linting

```bash
# Format code with black
black automl/ tests/

# Sort imports
isort automl/ tests/

# Run linter
flake8 automl/ tests/

# Type checking
mypy automl/

# Or run all checks at once
pre-commit run --all-files
```

### Running Examples

```bash
# CLI example (once implemented)
python -m automl.ui.cli --input data/sample.csv --output results/

# Python API example
python examples/quickstart.py
```

### Building Documentation

```bash
cd docs/
make html
# Open docs/_build/html/index.html in browser
```

---

## 🎯 Your First Contribution

### Option 1: Fix a Beginner Issue

1. Check GitHub Issues labeled `good-first-issue`
2. Comment that you're working on it
3. Create a branch: `git checkout -b fix/issue-123`
4. Make your changes
5. Run tests: `pytest`
6. Commit: `git commit -m "fix: description"`
7. Push: `git push origin fix/issue-123`
8. Create Pull Request

### Option 2: Write a Test

1. Find a module with <80% coverage
2. Create branch: `git checkout -b test/module-name`
3. Add test in `tests/unit/test_module_name.py`
4. Verify it runs: `pytest tests/unit/test_module_name.py`
5. Submit PR

### Option 3: Improve Documentation

1. Create branch: `git checkout -b docs/topic-name`
2. Edit files in `docs/` or add docstrings
3. Build docs: `cd docs && make html`
4. Submit PR

---

## 📖 Development Workflow

### Standard Workflow

```bash
# 1. Update main branch
git checkout main
git pull origin main

# 2. Create feature branch
git checkout -b feature/your-feature-name

# 3. Make changes
# ... edit files ...

# 4. Run tests
pytest

# 5. Format code (pre-commit will do this too)
black automl/ tests/
isort automl/ tests/

# 6. Commit
git add .
git commit -m "feat: your feature description"

# 7. Push
git push origin feature/your-feature-name

# 8. Create PR on GitHub
# Go to GitHub and create Pull Request
```

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting (no code change)
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Build/tooling changes

**Example**:
```
feat(data): add parquet file loader

- Implement ParquetLoader class with chunked reading
- Add support for partitioned parquet datasets
- Include comprehensive unit tests
- Update documentation

Closes #42
```

---

## 🧪 Testing Guidelines

### Test Structure

```python
# tests/unit/test_example.py
import pytest
from automl.module import ClassName

class TestClassName:
    """Test suite for ClassName."""
    
    def test_basic_functionality(self):
        """Test basic use case."""
        obj = ClassName()
        result = obj.method()
        assert result == expected
    
    def test_edge_case(self):
        """Test edge case."""
        obj = ClassName()
        with pytest.raises(ValueError):
            obj.method(invalid_input)
```

### Test Coverage Goals

- Minimum: 80% overall
- Critical modules: 90%+
- All public APIs: 100%

### Running Specific Tests

```bash
# By module
pytest tests/unit/test_data_loaders.py

# By class
pytest tests/unit/test_data_loaders.py::TestCSVLoader

# By test
pytest tests/unit/test_data_loaders.py::TestCSVLoader::test_load_basic

# By marker (once configured)
pytest -m "not slow"
```

---

## 🐛 Debugging Tips

### Using Debugger

```python
# Add breakpoint in code
import pdb; pdb.set_trace()

# Or use built-in breakpoint (Python 3.7+)
breakpoint()

# Or in VS Code: Click left of line number to add breakpoint
```

### Verbose Logging

```python
# Set environment variable
export AUTOML_LOG_LEVEL=DEBUG

# Or in code
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Common Issues

**Issue**: Import errors after installing package
```bash
# Solution: Reinstall in editable mode
pip install -e .
```

**Issue**: Tests fail with module not found
```bash
# Solution: Make sure you're in virtual environment
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
```

**Issue**: Pre-commit hooks fail
```bash
# Solution: Let them auto-fix, then retry
git add .
git commit -m "your message"
```

---

## 📚 Key Resources

### Documentation
- [Implementation Plan](IMPLEMENTATION_PLAN.md) - Detailed roadmap
- [Architecture Doc](doc.md) - System design
- [Technical Decisions](TECHNICAL_DECISIONS.md) - Decision log

### External Resources
- [scikit-learn docs](https://scikit-learn.org/)
- [PyTorch tutorials](https://pytorch.org/tutorials/)
- [Optuna documentation](https://optuna.readthedocs.io/)
- [MLflow docs](https://mlflow.org/docs/latest/index.html)

### Communication
- GitHub Issues - Bug reports & feature requests
- Pull Requests - Code reviews
- [Slack/Discord] - (To be setup) Daily communication
- Weekly sync meetings - (To be scheduled)

---

## 🎓 Learning Paths

### For Python Developers New to ML

1. Complete scikit-learn tutorial
2. Read "Hands-On Machine Learning" book
3. Work through `examples/01_getting_started.ipynb`
4. Start with data loading module

### For ML Engineers New to Software Engineering

1. Review Python best practices (PEP 8)
2. Learn pytest basics
3. Understand Git workflow
4. Start with adding tests to existing modules

### For Everyone

1. Read the [Implementation Plan](IMPLEMENTATION_PLAN.md)
2. Understand the architecture from [doc.md](doc.md)
3. Run all examples in `examples/`
4. Pick an issue labeled `good-first-issue`

---

## ⚡ Pro Tips

1. **Use VS Code**: Configured settings in `.vscode/settings.json`
2. **Let pre-commit fix things**: Don't manually format, let hooks do it
3. **Write tests first**: TDD makes better code
4. **Small commits**: Easier to review and revert
5. **Ask questions**: No question is stupid
6. **Read error messages**: They usually tell you what's wrong
7. **Check CI before pushing**: Run `pytest` and `pre-commit` locally first

---

## 🆘 Getting Help

**Stuck?**

1. Check documentation and examples
2. Search GitHub Issues
3. Ask in team chat
4. Create a new issue with `question` label

**Found a bug?**

1. Check if already reported
2. Create issue with:
   - Steps to reproduce
   - Expected behavior
   - Actual behavior
   - Environment details
3. Label as `bug`

**Have an idea?**

1. Check roadmap in Implementation Plan
2. Create issue describing:
   - Problem it solves
   - Proposed solution
   - Alternatives considered
3. Label as `enhancement`

---

## ✅ Ready to Start?

### Phase 1 Quick Tasks (Good for First PRs)

- [ ] Add a new data loader (Excel, JSON)
- [ ] Improve error messages in existing loaders
- [ ] Add more unit tests to increase coverage
- [ ] Write docstrings for undocumented functions
- [ ] Create example notebook for a use case
- [ ] Fix a `good-first-issue`

### Check Your Setup

Run this to verify everything:

```bash
# Should all pass ✓
python --version          # 3.10 or 3.11
pytest --version          # Latest
black --version           # Latest
git --version             # Any recent

# Should show tests passing
pytest tests/ -v

# Should show no issues
pre-commit run --all-files
```

---

**Welcome to the AutoML project! 🎉**

Questions? Check [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) or ask the team!
