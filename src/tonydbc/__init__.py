"""
tonydbc

    TonyDBC: A context manager for mariadb data connections.

    MQTTClient: A context manager for an MQTT client

    create_test_database: function to create a complete test database

Note: you should define the following environment variables before using this library:

    USE_PRODUCTION_DATABASE
    MYSQL_TEST_DATABASE
    MYSQL_PRODUCTION_DATABASE
    DEFAULT_TIMEZONE
    DEFAULT_TIME_OFFSET
    MEDIA_BASE_PATH_PRODUCTION

e.g. 
    import dotenv
    dotenv.load_dotenv(os.path.join(sys.path[0], "..", "..", ".env"))
    sys.path.insert(1, os.path.join(sys.path[0], ".."))
    import tonydbc

"""
__version__ = "1.0.3"

from .tonydbc import TonyDBC
from .tony_utils import (
    set_MYSQL_DATABASE,
    get_current_time,
    get_current_time_string,
    deserialize_table,
    get_env_bool,
)
from .mqtt_client import MQTTClient
from .dataframe_fast import DataFrameFast
from .create_test_database import create_test_database
