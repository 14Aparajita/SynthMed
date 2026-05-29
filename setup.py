from setuptools import setup, find_packages

setup(
    name="synthmed",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "pydantic>=2.0.0",
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.4",
        "pyyaml>=6.0",
        "scikit-learn>=1.3.0",
    ],
    python_requires=">=3.9",
)