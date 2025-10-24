"""Setup script for cubic_graph_optimizer package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cubic-graph-optimizer",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Optimize spanning trees in cubic graphs using Whitehead flips",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cubic_graph_optimizer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "networkx>=2.8.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "automorphisms": ["pynauty>=2.8.0"],
        "plotting": ["matplotlib>=3.5.0"],
        "all": ["pynauty>=2.8.0", "matplotlib>=3.5.0"],
    },
    entry_points={
        "console_scripts": [
            "cubic-optimizer=cubic_graph_optimizer.main:main",
        ],
    },
)