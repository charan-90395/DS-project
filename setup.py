from setuptools import setup, find_packages

setup(
    name='DSProject',
    version='0.1.0',
    author='Charan Banda',
    author_email="b.charanprakash178@gmail.com",
    description='A Data Science Project Template',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "requests",
        "flask"
    ],
)
