from setuptools import setup
setup(
    name="demotivational_policy_descent",
    packages=["demotivational_policy_descent"],
    version="0.0.1.dev1",
    description="Reinforcement Learning",
    author="Fabio Colella",  # packager 
    author_email="fabio.colella@aalto.fi",
    url="https://github.com/fcole90/demotivational-policy-descent.git",
    download_url="https://github.com/fcole90/demotivational-policy-descent.git",
    keywords=["reinforcement", "learning"],
    license="MIT License",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research"
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    long_description="""\
Final project for the 2018 edition of the course of Reinforcement Learning
at Aalto University, Finland.
""",
    install_requires=['gym', 'numpy', 'Pygame', 'matplotlib', 'torch'],
    python_requires='>=3.4'
)
