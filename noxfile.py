"""Configuration file for nox routines"""

from pathlib import Path as pathlib_Path

import nox

_PYTHON_VERSIONS = ["3.9", "3.8", "3.7"]
_NOXFILE = pathlib_Path(__file__)
_HERE = _NOXFILE.parent

# Pylint-exe conversion table for return codes (can be a combination of it)
# Pylint code 	Message 	            Final return code (pylint-exit)
# 1 	        Fatal message issued 	    1
# 2 	        Error message issued 	    0
# 4 	        Warning message issued 	    0
# 8 	        Refactor message issued	    0
# 16 	        Convention message issued   0
# 32 	        Usage error 	            1
_PYLINT_VALID_RETURN_CODES = range(2, 32)


def _run_lint_session(session, location):
    pylintrc_path = f"{location}/.pylintrc"
    with open(f"{location}/pylint.log", "w") as pylint_out:
        session.run(
            "pylint",
            "--output-format=parseable",
            "--reports=no",
            f"--rcfile={pylintrc_path}",
            "--jobs=0",
            location,
            stdout=pylint_out,
            success_codes=_PYLINT_VALID_RETURN_CODES,
        )


@nox.session(python=_PYTHON_VERSIONS)
def lint(session):
    """Runs linting commands"""
    session.install("flake8")
    session.run("flake8", "--extend-exclude", ".nox,experiments", f"{_HERE}")


@nox.session(python=_PYTHON_VERSIONS)
def format_code(session):
    """Runs formatter command"""
    session.install("black")
    session.run('black', '--line-length', 80, r'*.py')
