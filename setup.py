"""
Setup configuration for miso-onnx package.
"""

from setuptools import setup, find_packages
from pathlib import Path

def read_requirements(filename):
    """
    Read requirements from a requirements file.
    Filters out comments, empty lines, and git URLs.
    """
    requirements_file = Path(__file__).parent / filename
    if not requirements_file.exists():
        return []
    
    requirements = []
    with open(requirements_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith('#'):
                requirements.append(line)
    return requirements

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="miso_onnx",
    version="0.1.5",
    description="ONNX-based inference for MISO microfossil classification models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ross Marchant",
    author_email="ross.g.marchant@gmail.com",
    url="https://github.com/microfossil/miso-onnx",
    packages=find_packages(),
    python_requires="==3.11.*",
    install_requires=read_requirements("requirements.txt"),
    entry_points={
        "console_scripts": [
            "miso-onnx=miso_onnx.__main__:cli",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Programming Language :: Python :: 3.11",
    ],
)
