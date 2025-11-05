"""Setup configuration for Orion Data Analysis Agent."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="orion-data-analyst",
    version="1.0.6",
    author="Gavriel Hannuna",
    author_email="gavriel.hannuna@gmail.com",
    description="AI-powered BigQuery data analysis agent with natural language interface",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gavrielhan/orion-data-analyst",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Database",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "langgraph>=0.2.0",
        "langchain>=0.3.0",
        "google-cloud-bigquery>=3.25.0",
        "google-generativeai>=0.3.0",
        "db-dtypes",
        "pandas>=2.0.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "typing-extensions>=4.12.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    entry_points={
        "console_scripts": [
            "orion=src.cli:main",
        ],
    },
    include_package_data=True,
    keywords="data-analysis bigquery ai nlp sql gemini analytics",
    project_urls={
        "Bug Reports": "https://github.com/gavrielhan/orion-data-analyst/issues",
        "Source": "https://github.com/gavrielhan/orion-data-analyst",
        "Documentation": "https://github.com/gavrielhan/orion-data-analyst#readme",
    },
)

