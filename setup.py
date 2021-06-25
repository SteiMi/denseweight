import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="denseweight",
    version="0.1.0",
    author="Michael Steininger",
    author_email="steininger@informatik.uni-wuerzburg.de",
    description="The imbalanced regression method DenseWeight",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/steimi/denseweight",
    project_urls={
        "Bug Tracker": "https://github.com/steimi/denseweight/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "denseweight"},
    packages=setuptools.find_packages(where="denseweight"),
    install_requires=["numpy>=1.20.0", "KDEpy", "scikit-learn"],
    python_requires=">=3.6",
)
