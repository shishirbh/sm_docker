from setuptools import setup, find_packages

setup(
    name="unified_sagemaker_framework",
    version="1.0.0",
    description="Unified SageMaker framework for custom training and serving",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "sagemaker-training>=4.4.0",
        "sagemaker-inference>=1.10.0",
        "numpy>=1.20.0",
        "boto3>=1.20.0",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)