# TonyDBC

Latest version: 1.0.2

Available on PyPI: https://pypi.org/project/tonydbc/

2024-01-07: [Release announcement on Medium](https://mcurrie-59915.medium.com/introducing-tonydbc-a-high-level-database-connector-for-mariadb-python-and-pandas-8600676fbf88)

Supports high-level database operations within Python.  TonyDBC is short for Tony’s Database Connector, named for Maria’s lover in West Side Story.

TonyDBC uses the MariaDB/Connector but adds these features:

* Support for the object-oriented context manager design pattern, which automatically closes connections.
* A way to convert between pandas DataFrames and MariaDB tables, including serialization of complex data types like dictionaries and numpy arrays.
* A way to autoresume when the connection is lost.
* Support for fast INSERT and UPDATE from DataFrame.to_sql. Currently pandas’ implemention using sqlalchemy is 300x slower.
* A way to quickly and automatically clone a database from a DDL script and copy the contents from the production database.
* Batch INSERT and UPDATE offline and resume when Internet is available.
* Logging database actions using the standard logging module.
* In a CI/CD context, automatically connecting to the correct production or test database.
* Support for the np.Int64Dtype data type.
* Support for pandas timestamps.
* Able to bulk insert data containing NULLs

### Installation

To install TonyDBC from PyPI:

```python
pip install tonydbc
```

### Usage

A typical use case:

```python
import dotenv
from tonydbc import TonyDBC

dotenv.load_dotenv()

sql_params = {
    "host": os.environ["MYSQL_HOST"],
    "user": os.environ["MYSQL_USER"],
    "password": os.environ["MYSQL_PASSWORD"],
    "database": os.environ["MYSQL_DATABASE"],
    "port": int(os.environ["MYSQL_PORT"]),
}

with TonyDBC(**sql_params) as db:
    # Do stuff
    pass
```
