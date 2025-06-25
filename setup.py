from setuptools import setup, find_packages

setup(
    name="hsi",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "pandas",
        "tqdm"
    ]
) 