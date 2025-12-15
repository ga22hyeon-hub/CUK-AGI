#!/usr/bin/env python3
"""
AGI Framework Setup Script
Multi-LLM Agent 기반 AGI 프레임워크
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="agi-framework",
    version="0.1.0",
    author="CU NLP Lab",
    author_email="nlp@example.com",
    description="Multi-LLM Agent 기반 사용자 상호작용 및 능동학습 AGI 프레임워크",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cuknlp/agi-framework",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.11.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "agi-framework=agi_framework.cli:main",
        ],
    },
)

