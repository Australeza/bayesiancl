# Installation instructions
This repository contains the python package `{{ cookiecutter.project_name|lower }}`. Here we describe how to install it.

## Installation with pyenv
For `pyenv` users, the script
```shell
./init_repo.sh
```
can be called to set up a local environment called `{{ cookiecutter.project_name|lower }}` and install the requirements.
Users of `conda` have to create an environment manually, and can then and follow the developer installation instructions below.

## Manual developer installation
To set up the environment exactly as it was tested, go to the root dir of the repository and run:
```shell
pip install -r requirements.txt
pip install -r requirements_dev.txt
```
in your current (virtual) python environment. Then, install the `{{ cookiecutter.project_name|lower }}` Python package using
```shell
pip install -e .
```
when in the root directory of the repository. You can also use `pip install -e {{ cookiecutter.project_name|lower }}`, although this does not seem to always work on Windows machines.

### Pre-commit hooks (Git only)
It is recommended to install pre-commit hooks using
```shell
pre-commit install
```
within your virtual environment, which perform code integrity checks and style checks before committing. Note that you do need the `git` command line for that.

### Jupyter Notebooks
We do not store jupyter notebooks (`*.ipynb`) in the repository (they are blocked by the `.gitignore`). The reason is that
jupyter files are harder to version control as they are not plain text, but also because run outcomes often
contain privacy-sensitive data such as images, serial numbers or other personal data.

There, the module `jupytext` is used to create a script out of notebooks. If you install the development requirements
before starting jupyter, you can just open them like any other notebook. It then synchronizes the `*.ipynb` with a `*.py`
file of the same name, and you can commit the `*.py` version.

## Regular installation
If you do not plan to develop, but simply want to use tools from the package, you can run
```shell
pip install .
```
in the root folder of the repository. You can also use `pip install {{ cookiecutter.project_name|lower }}`, although this does not seem to always work on Windows machines.
