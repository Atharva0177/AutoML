# Getting Started with AutoML

Welcome to the AutoML system! This guide will help you get up and running quickly.

## Installation

### Prerequisites

- Python 3.10 or 3.11
- pip (Python package manager)
- Git (for development)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/your-org/automl.git
cd automl

# Create and activate virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux/Mac:
source venv/bin/activate

# Install the package
pip install -e .
```

For development (includes testing and code quality tools):

```bash
pip install -r requirements-dev.txt
pre-commit install
```

## Basic Usage

### Python API

```python
from automl import AutoML

# Initialize AutoML
aml = AutoML()

# Load your data
aml.load_data("path/to/your/data.csv", target_column="target")

# Get data information
info = aml.get_data_info()
print(f"Data shape: {info['shape']}")
print(f"Quality score: {info['quality_score']:.1f}/100")

# Train models (coming in Phase 1, Month 3)
# results = aml.train()
# best_model = aml.get_best_model()
```

### Command Line Interface

```bash
# Train on a dataset
automl train --input data.csv --target target_column --output results/

# View results
automl results --path results/

# Get system info
automl info

# Validate data only (skip training  
automl train --input data.csv --target target --validate-only
```

## Running Examples

We provide several example scripts to help you get started:

```bash
# Quick start example
python examples/quickstart.py

# Advanced example with data quality issues
python examples/advanced_pipeline.py
```

## Project Structure

```
automl/
├── automl/              # Main package
│   ├── data/           # Data loading and validation
│   ├── eda/            # Exploratory data analysis (coming soon)
│   ├── preprocessing/  # Data preprocessing (coming soon)
│   ├── models/         # ML models (coming soon)
│   ├── training/       # Training pipeline (coming soon)
│   ├── config/         # Configuration management
│   ├── utils/          # Utilities
│   ├── ui/             # User interfaces (CLI, web)
│   └── core/           # Core AutoML logic
├── tests/              # Tests
├── examples/           # Example scripts
└── docs/               # Documentation
```

## Current Features (MVP - Phase 1, Weeks 1-2)

✅ **Data Loading**
- CSV files with automatic encoding detection
- Parquet files
- Automatic schema inference

✅ **Data Validation**
- Dimension checks
- Missing value analysis
- Duplicate detection
- Constant column identification
- Target column validation

✅ **Data Quality Assessment**
- Quality scoring (0-100)
- Missing value analysis
- Outlier detection
- Data type distribution
- Actionable recommendations

✅ **Configuration Management**
- YAML-based configuration
- Environment variable support
- Easy customization

✅ **Logging & Monitoring**
- Structured logging
- Configurable log levels
- File and console output

## Coming Soon

🔄 **Phase 1, Month 1-2** (Weeks 3-8)
- Exploratory Data Analysis (EDA)
- Problem type detection
- Data preprocessing pipeline
- Feature engineering

🔄 **Phase 1, Month 3** (Weeks 9-12)
- Model training (5 ML algorithms)
- Cross-validation
- Model comparison
- Performance metrics

## Configuration

Edit `automl/config/settings.yaml` or create a `.env` file:

```bash
# Copy example environment file
cp .env.example .env

# Edit with your settings
nano .env
```

Key settings:
- `AUTOML_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `AUTOML_DATA_DIR`: Data directory path
- `AUTOML_RESULTS_DIR`: Results output directory
- `AUTOML_N_JOBS`: Number of parallel jobs (-1 for all cores)

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=automl --cov-report=html

# Run specific test file
pytest tests/unit/test_data_loaders.py

# Run tests in parallel (faster)
pytest -n auto
```

## Code Quality

```bash
# Format code
black automl/ tests/

# Sort imports
isort automl/ tests/

# Lint
flake8 automl/ tests/

# Type check
mypy automl/

# Or run all checks with pre-commit
pre-commit run --all-files
```

## Troubleshooting

### Import Errors

If you get import errors, make sure you've installed the package:
```bash
pip install -e .
```

### Missing Dependencies

Install all dependencies:
```bash
pip install -r requirements.txt
```

For development:
```bash
pip install -r requirements-dev.txt
```

### Permission Errors (Windows)

Run your terminal/PowerShell as Administrator when installing packages.

### Encoding Issues with CSV

The CSV loader automatically detects encoding, but you can specify manually:
```python
aml.load_data("data.csv", encoding="utf-8")
```

## Getting Help

- **Documentation**: See the [docs/](docs/) directory
- **Examples**: Check [examples/](examples/) directory
- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Ask questions on GitHub Discussions

## Next Steps

1. **Try the quick start example**:
   ```bash
   python examples/quickstart.py
   ```

2. **Load your own data**:
   ```python
   from automl import AutoML
   aml = AutoML()
   aml.load_data("your_data.csv", target_column="your_target")
   ```

3. **Explore the configuration**:
   - Read `automl/config/settings.yaml`
   - Customize for your needs

4. **Star the repository** and watch for updates!

## Roadmap

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the complete development roadmap.

**Current Status**: Phase 1, Month 1 (Weeks 1-2) ✅  
**Next Milestone**: EDA Module (Weeks 3-4)

---

Happy AutoML-ing! 🚀
