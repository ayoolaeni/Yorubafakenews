from setuptools import find_packages, setup

setup(
    name="yoruba-fake-news-detection",
    version="0.1.0",
    description="Classifies Yoruba-language news text as GENUINE or FAKE.",
    author="Fatunwase Micheal",
    python_requires=">=3.10",
    packages=find_packages(include=["src", "src.*"]),
)
