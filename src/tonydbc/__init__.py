"""
tonydbc

    TonyDBC: A context manager for mariadb data connections.

    MQTTClient: A context manager for an MQTT client

    create_test_database: function to create a complete test database

Note: you should define the following environment variables before using this library, e.g.

    CHECK_ENVIRONMENT_INTEGRITY = True
    DOT_ENVS                    = ["..\\..\\.env", ".env"]
    USE_PRODUCTION_DATABASE     = True
    PRODUCTION_DATABASES        = ["master_db", "master2_db", "etc"]
    MYSQL_TEST_DATABASE         = test_db
    MYSQL_PRODUCTION_DATABASE   = master_db
    INTERACT_AFTER_ERROR        = False
    AUDIT_PATH                  = database
    # Full list is at pytz.all_timezones
    DEFAULT_TIMEZONE            = Asia/Singapore
    MEDIA_BASE_PATH_PRODUCTION  = C:\\

e.g.
    import tonydbc
    tonydbc.load_dotenvs()

"""

__version__ = "1.3.9"

# Public API - explicitly declare what should be available when importing this package
__all__ = [
    "load_dotenv",
    "get_env_bool",
    "get_env_list",
    "load_dotenvs",
    "set_MYSQL_DATABASE",
    "get_current_time",
    "deserialize_table",
    "list_to_SQL",
    "list_to_SQL2",
    "TonyDBC",
    "MQTTClient",
    "DataFrameFast",
    "create_test_database",
]

# Include vanialla dotenv.load_dotenv to be comprehensive, but it's not really needed
# since the version people should use is load_dotenvs
from dotenv import load_dotenv

from .create_test_database import create_test_database
from .dataframe_fast import DataFrameFast
from .env_utils import get_env_bool, get_env_list, load_dotenvs
from .mqtt_client import MQTTClient
from .tony_utils import (
    deserialize_table,
    get_current_time,
    list_to_SQL,
    list_to_SQL2,
    set_MYSQL_DATABASE,
)
from .tonydbc import TonyDBC
