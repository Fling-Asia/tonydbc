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
            DOT_ENVS

load_dotenvs() is all you need in most cases
"""

import code
import os
import sys
import dotenv
from dotenv import load_dotenv
import json
import pathlib
from collections import Counter


def check_duplicate_keys(dotenv_path):
    # Load key-value pairs from the .env file
    env_dict = dotenv.dotenv_values(dotenv_path)

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


def check_environment_variable_integrity(env_filepath):
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

    # Check for duplicate keys
    check_duplicate_keys(env_filepath)
    check_duplicate_keys(example_env_filepath)

    # Check for missing keys in either file
    current_env = dotenv.dotenv_values(env_filepath)
    example_env = dotenv.dotenv_values(example_env_filepath)

    current_keys = set(current_env.keys())
    example_keys = set(example_env.keys())

    # Check the symmetric difference; if it's nonempty, a key is missing in
    # one or the other of the files
    if len(current_keys ^ example_keys) > 0:
        raise AssertionError(
            "Your .env file and .env.example files have different variables defined.  Please fix this; \n "
            f"in .env but not in .env.example we have: {sorted(current_keys.difference(example_keys))}\n"
            f"in .env.example but not in .env we have: {sorted(example_keys.difference(current_keys))}\n"
        )


def get_env_bool(key):
    """Handles the case of a boolean environment variable"""
    if not key in os.environ:
        raise KeyError(f"No environment variable {key}")

    if not os.environ[key] in ("True", "False"):
        raise AssertionError(f"Key {key} is not proper boolean: {os.environ[key]}")

    return os.environ[key] == "True"


def get_env_list(key):
    """Parse a list of strings from an environment variable

    Returns:
        If it's just a blank string, return []

        Otherwise, if it's not a list of nonempty strings, an exception
        will be raised.

        Otherwise, it will return the list of strings.
    """
    if not key in os.environ:
        raise KeyError(f"No environment variable {key}")

    v = os.environ[key]

    if v is None:
        v = ""

    assert type(v) == str

    # Get the string ready to be parsed
    v = v.strip().replace("'", '"')

    if v == "":
        return []

    # Parse the list of strings
    try:
        v_list = json.loads(v)
    except json.decoder.JSONDecodeError as e:
        raise Exception(
            "Expected environment variable to be a list. "
            f' e.g. ["Zebra", "Camel"]. {e}'
        )

    # Confirm it's a list of nonempty strings
    if not type(v_list) == list:
        raise ValueError(
            f"Environment variable {key} is "
            f"expected to be a list, but it is not: {os.environ[key]}"
        )

    # Convert all elements to strings
    v_list = [str(v) for v in v_list]

    if not all([type(x) == str and len(x) > 0 for x in v_list]):
        raise ValueError(
            f"Environment variable {key} is "
            f"expected to be a list of strings, but it is "
            f"not: {os.environ[key]}"
        )

    return v_list


def load_dotenvs():
    """
    A more powerful version of `dotenv.load_dotenv`.  It will:
        - Load the .env file in the script's path, if any
        - Also, load any .env files listed in DOT_ENVS
        - Also, check these files against their .env.example files for omissions
    """
    # Load .env wherever it may be
    load_dotenv()
    # Also, the .env file in the script's path, if any
    base_env_path = os.path.join(sys.path[0], ".env")
    if not os.path.isfile(base_env_path):
        print(f"WARNING: {base_env_path} .env file does not exist.")

        # Last resort: try seeing if it's just already there in os.environ anyway
        if not "DOT_ENVS" in os.environ:
            print(f"WARNING: No `DOT_ENVS` present.")
            return
    else:
        load_dotenv(base_env_path)

    if "DOT_ENVS" in os.environ:
        # Get every .env we are supposed to load
        paths = get_env_list("DOT_ENVS")
    else:
        paths = [base_env_path]

    # In some contexts, like docker on the server, it's impractical
    # to check environment integrity so let's do it only optionally
    cs = "CHECK_ENVIRONMENT_INTEGRITY"
    do_check = cs in os.environ and get_env_bool(cs)

    # Prepend the script's path if the provided env path is relative
    paths = [p if os.path.isabs(p) else os.path.join(sys.path[0], p) for p in paths]

    # Resolve the path to remove any pesky ".."s
    paths = [str(pathlib.Path(p).resolve()) for p in paths]

    # Load and check these files against their .env.example files for omissions
    for p in paths:
        if not os.path.isfile(p):
            print(f"WARNING: env path {p} does not exist")
            continue
        dotenv.load_dotenv(p)
        if do_check:
            print(f"Checking environment integrity on {p}")
            # Check environment variables are consistent between .env and .env.example
            check_environment_variable_integrity(p)
