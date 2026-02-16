"""Setup configuration for AutoML package."""

from pathlib import Path

from setuptools import find_packages, setup

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (
    (this_directory / "README.md").read_text(encoding="utf-8")
    if (this_directory / "README.md").exists()
    else ""
)

setup(
    name="automl",
    version="0.1.0",
    author="AutoML Team",
    author_email="team@automl.dev",
    description="Automated Machine Learning system for intelligent model selection and training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/automl",
    packages=find_packages(exclude=["tests", "tests.*", "docs", "examples"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
        "catboost>=1.2.0",
        "pyarrow>=10.0.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.0",
        "joblib>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.17.0",
        "optuna>=3.4.0",
        "mlflow>=2.8.0",
        "click>=8.1.0",
        "rich>=13.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "pytest-xdist>=3.4.0",
            "black>=23.10.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.6.0",
            "pre-commit>=3.5.0",
        ],
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
        "deep-learning": [
            "torch>=2.1.0",
            "torchvision>=0.16.0",
            "transformers>=4.35.0",
            "opencv-python>=4.8.0",
        ],
        "api": [
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
            "pydantic>=2.5.0",
        ],
        "ui": [
            "streamlit>=1.28.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "automl=automl.ui.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
