from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="supertrend-trading-bot",
    version="2.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Automated SuperTrend trading bot for NSE using Zerodha Kite",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/supertrend-trading-bot",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=[
        "kiteconnect>=4.1.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "python-dotenv>=1.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    entry_points={
        "console_scripts": [
            "trading-bot=cli:main",
        ],
    },
)
