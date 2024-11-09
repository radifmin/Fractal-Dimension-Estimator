from setuptools import setup, find_packages

setup(
    name="Fractal-Dimension-Estimator",
    version="0.1",
    description="Fractal Dimension Estimator CNN",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "torchvision"
    ],
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers, Education',
    ],
    author="radifmin",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/radifmin/Fractal-Dimension-Estimator",
    python_requires='>=3.9',
)