"""
Consolidate all env-related functions into this module.

common point to prevent messy path finding

    check_duplicate_keys
    check_environment_variable_integrity
    get_env_bool
    get_env_list
    load_dotenv
    load_dotenvs
        it's nice but not necessary for the following variables to be set already:
            CHECK_ENVIRONMENT_INTEGRITY
            AUDIT_PATH
            DOT_ENVS

load_dotenvs() is all you need in most cases
"""

import json
import logging
import os
import pathlib
import sys
from collections import Counter

import dotenv

# These warnings are very verbose and annoying, so we'll skip them for now
WARN_MISSING_PATHS = False


# We need this to capture the logging warning
# "Python-dotenv could not parse statement starting at line 1"
class CaptureLogsHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def check_duplicate_keys(dotenv_path: str) -> None:
    # Load key-value pairs from the .env file
    # env_dict = dotenv.dotenv_values(dotenv_path)  # Not currently used but may be useful for debugging

    # Read the file again as a plain text file
    with open(dotenv_path, "r") as f:
        lines = f.readlines()

    keys = [
        line.split("=")[0]
        for line in lines
        if line.strip() and not line.startswith("#")
    ]

    # Check for duplicates
    counter = Counter(keys)
    duplicates = [key for key, count in counter.items() if count > 1]

    if duplicates:
        raise AssertionError(
            f"Duplicate keys detected in {dotenv_path}: {', '.join(duplicates)}"
        )


def check_environment_variable_integrity(env_filepath: str) -> None:
    # Validate that the .env file has all the latest environment variables
    assert env_filepath.endswith(".env")

    # Make the path nice (removing the relative ..s if any)
    env_filepath = str(pathlib.Path(env_filepath).resolve())

    example_env_filepath = env_filepath + ".example"

    # Check that both files exist
    if not os.path.isfile(env_filepath):
        raise AssertionError(
            f"Please define a `.env` file at {pathlib.Path(env_filepath).parent}\n"
        )

    if not os.path.isfile(example_env_filepath):
        raise AssertionError(
            "Your repo is missing a `.env.example` file in "
            f"{pathlib.Path(env_filepath).parent}\n"
        )

    # We need this to capture the logging warning
    # "Python-dotenv could not parse statement starting at line 1"
    # Create the custom logging handler
    dotenv_handler = CaptureLogsHandler()

    # Get the logger for 'dotenv' and attach the handler
    dotenv_logger = logging.getLogger("dotenv")
    dotenv_logger.addHandler(dotenv_handler)
    dotenv_logger.setLevel(logging.WARNING)

    # LOAD the files to check for parsing errors
    for fp in [env_filepath, example_env_filepath]:
        dotenv.dotenv_values(fp)
        parsed_errors = [
            msg
            for msg in dotenv_handler.messages
            if "could not parse statement starting" in msg
        ]
        if len(parsed_errors) > 0:
            raise AssertionError(f"{fp} contains errors {parsed_errors}")

    # Check for duplicate keys
    check_duplicate_keys(env_filepath)
    check_duplicate_keys(example_env_filepath)

    # Check for DIFFERENCES in keys between the two files
    current_env = dotenv.dotenv_values(env_filepath)
    example_env = dotenv.dotenv_values(example_env_filepath)
    current_keys = set(current_env.keys())
    example_keys = set(example_env.keys())

    # Check for missing keys in either file
    # Check the symmetric difference; if it's nonempty, a key is missing in
    # one or the other of the files
    if len(current_keys ^ example_keys) > 0:
        raise AssertionError(
            f"For .env {env_filepath}:\n"
            "Your .env file and .env.example files have different variables defined.  Please fix this; \n "
            f"in .env but not in .env.example we have: {sorted(current_keys.difference(example_keys))}\n"
            f"in .env.example but not in .env we have: {sorted(example_keys.difference(current_keys))}\n"
        )

    # Check that all variables ending in _PATH or _DIRECTORY are valid paths
    path_keys = [
        k
        for k in current_keys
        if any(k.endswith(j) for j in ["_PATH", "_DIRECTORY", "_FILEPATH"])
    ]
    for current_key in path_keys:
        k_prefix = f".env {env_filepath} has variable {current_key}"
        current_path = current_env[current_key]

        if WARN_MISSING_PATHS:
            if current_path is None or current_path == "":
                print(f"WARNING: {k_prefix} which is blank or None.")
                continue
            if not (os.path.isdir(current_path) or os.path.isfile(current_path)):
                print(
                    "WARNING: "
                    f"For {k_prefix} with path {current_path} which is not a valid path on your machine."
                )


def get_env_bool(key: str) -> bool:
    """Handles the case of a boolean environment variable"""
    if key not in os.environ:
        raise KeyError(f"No environment variable '{key}' was found.")

    if os.environ[key] not in ("True", "False"):
        raise AssertionError(
            f"Environment variable '{key}' is recorded "
            f"as '{os.environ[key]}', which is not a proper boolean. "
            "It must be either 'True' or 'False'."
        )

    return os.environ[key] == "True"


def get_env_list(key: str) -> list[str]:
    """Parse a list of strings from an environment variable

    Returns:
        If it's just a blank string, return []

        Otherwise, if it's not a list of nonempty strings, an exception
        will be raised.

        Otherwise, it will return the list of strings.
    """
    if key not in os.environ:
        raise KeyError(f"No environment variable {key}")

    v = os.environ[key]

    if v is None:
        v = ""

    assert isinstance(v, str)

    # Get the string ready to be parsed
    v = v.strip().replace("'", '"')

    if v == "":
        return []

    # Parse the list of strings
    try:
        v_list = json.loads(v)
    except json.decoder.JSONDecodeError as e:
        raise Exception(
            "Expected environment variable to be a list, and for all Windows "
            "paths to be escaped e.g. \\\\ instead of a single \\ ; "
            f'e.g. ["C:\\\\Zebra\\\\.env", "Camel\\\\other.env"], '
            f"rather than: {os.environ[key]}.  {e}"
        )

    # Confirm it's a list of nonempty strings
    if not isinstance(v_list, list):
        raise ValueError(
            f"Environment variable {key} is "
            f"expected to be a list, but it is not: {os.environ[key]}"
        )

    # Convert all elements to strings
    v_list = [str(v) for v in v_list]

    if not all([isinstance(x, str) and len(x) > 0 for x in v_list]):
        raise ValueError(
            f"Environment variable {key} is "
            f"expected to be a list of strings, but it is "
            f"not: {os.environ[key]}"
        )

    return v_list


def load_dotenvs() -> list[str]:
    """
    A more powerful version of `dotenv.load_dotenv`.  It will:
        - Load the .env file in the script's path, if any
        - Also, load any .env files listed in DOT_ENVS
        - Also, check these files against their .env.example files for omissions
    """
    # Load .env wherever it may be
    dotenv.load_dotenv(override=True)
    # Also, the .env file in the script's path, if any
    base_env_path = os.path.join(sys.path[0], ".env")
    if os.path.isfile(base_env_path):
        print(f"load_dotenvs: loading base env {base_env_path}")
        dotenv.load_dotenv(base_env_path, override=True)
    else:
        print(
            f"load_dotenvs: WARNING: base env {base_env_path} .env file does not exist."
        )

    if "DOT_ENVS" in os.environ:
        # Get every .env we are supposed to load
        env_paths_raw = get_env_list("DOT_ENVS")
    elif os.path.isfile(base_env_path):
        env_paths_raw = [base_env_path]
        print(
            f"load_dotenvs: WARNING: No `DOT_ENVS` in os.environ, "
            f"so defaulting to `DOT_ENVS` = {env_paths_raw}"
        )
    else:
        print(
            "load_dotenvs: WARNING: No `DOT_ENVS` in os.environ, and no base env file."
        )
        env_paths_raw = []

    # In some contexts, like docker on the server, it's impractical
    # to check environment integrity so let's do it only optionally
    cs = "CHECK_ENVIRONMENT_INTEGRITY"
    do_check = cs in os.environ and get_env_bool(cs)

    # Prepend the script's path if the provided env path is relative
    env_paths = [
        p if os.path.isabs(p) else os.path.join(sys.path[0], p) for p in env_paths_raw
    ]

    # Resolve the path to remove any pesky ".."s
    env_paths = [str(pathlib.Path(p).resolve()) for p in env_paths]

    # This is too verbose:
    # env_paths_str = "\n".join(env_paths)
    # print(f"`DOT_ENVS` are:\n{env_paths_str}")

    # Load and check these files against their .env.example files for omissions
    for env_path in env_paths:
        if not os.path.isfile(env_path):
            print(f"load_dotenvs: WARNING: env path {env_path} does not exist")
            continue
        print(f"load_dotenvs: loading env_path {env_path}")
        dotenv.load_dotenv(env_path, override=True)
        if do_check:
            # This is too verbose:
            # print(f"load_dotenvs: checking env_path {env_path}")
            # Check environment variables are consistent between .env and .env.example
            check_environment_variable_integrity(env_path)

    return env_paths
