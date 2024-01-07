"""
tonydbc

A context manager for mariadb data connections.

On import, also checks that in the env filepath and example env filepath are consistent.

"""
import code
import os
import sys
import dotenv
import pathlib
from collections import Counter

__version__ = "1.0.0"

# Needs an extra .. for some reason (I guess if this library is imported...)
DEFAULT_ENV_FILEPATH = os.path.join(os.path.abspath(__file__), "..", "..", "..", ".env")


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


def check_environment_variable_integrity(env_filepath=DEFAULT_ENV_FILEPATH):
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


def get_env_bool2(key):
    """Handles the case of a boolean environment variable"""
    if not key in os.environ:
        raise KeyError(f"No environment variable {key}")

    if not os.environ[key] in ("True", "False"):
        raise AssertionError(f"Key {key} is not proper boolean: {os.environ[key]}")

    return os.environ[key] == "True"


# Load the environment variables
dotenv.load_dotenv()

# In some contexts, like docker on the server, it's annoying to check environment integrity
# so let's do it only optionally
if "CHECK_ENVIRONMENT_INTEGRITY" in os.environ and get_env_bool2(
    "CHECK_ENVIRONMENT_INTEGRITY"
):
    # Check environment variables are good.
    check_environment_variable_integrity()

from .tonydbc import (
    TonyDBC,
    set_MYSQL_DATABASE,
    get_current_time,
    get_current_time_string,
    deserialize_table,
    get_env_bool,
)
from .mqtt_client import MQTTClient
from .dataframe_fast import DataFrameFast

from .create_test_database import create_test_database
