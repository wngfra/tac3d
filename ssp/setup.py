from setuptools import setup

setup(
    name="ssp",
    version="0.2",
    description="Spatial Semantic Pointers",
    url="https://gl.appliedbrainresearch.com/abr/ssp",
    author="Applied Brain Research",
    author_email="info@appliedbrainresearch.com",
    maintainer="wngfra",
    maintainer_email="wngfra@gmail.com",
    packages=["ssp"],
    install_requires=[
        "nengo",
        "seaborn",
        "click",
        "tensorflow",
    ],
    extras_require={
        "tests": ["pytest", "pytest-rng"],
    },
    zip_safe=False,
)
