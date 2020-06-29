import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("VERSION", "r") as fh:
    version = fh.read()


setuptools.setup(
    name="SAILenv-SAILab",
    version=version,
    author="Enrico Meloni",
    author_email="meloni@diism.unisi.it",
    description="Python API for interfacing with SAILenv",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sailab-code/SAILenv",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.6'
)
