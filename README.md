# TonyDBC

Latest version: 1.2.14

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

To install TonyDBC from PyPI on Windows:

```bash
pip install tonydbc
```

To install TonyDBC on Ubuntu or Debian, first the MariaDB Connector / C must be installed, then the MariaDB Connector / Python must be installed, because neither can be installed via `pip3`:

```bash
cd ~

sudo apt-get install -y wget curl python3-packaging gcc

# Install mariadb CS Package Repository so apt-get can find the right package
# See https://mariadb.com/docs/skysql/connect/programming-languages/c/install/#Installation_via_Package_Repository_(Linux)
wget https://r.mariadb.com/downloads/mariadb_repo_setup || wget https://downloads.mariadb.com/MariaDB/mariadb_repo_setup
chmod +x mariadb_repo_setup
sudo ./mariadb_repo_setup  --mariadb-server-version="mariadb-1.2.13"
sudo apt-get install -y libmariadb3 libmariadb-dev

pip3 install tonydbc
```

### Usage

A typical use case:

```python
from tonydbc import TonyDBC, load_dotenvs
import pandas as pd

load_dotenvs()

sql_params = {
    "host": os.environ["MYSQL_HOST"],
    "user": os.environ["MYSQL_USER"],
    "password": os.environ["MYSQL_PASSWORD"],
    "database": os.environ["MYSQL_DATABASE"],
    "port": int(os.environ["MYSQL_PORT"]),
}

with TonyDBC(**sql_params) as db:
    # Do stuff
    contoso_users_df = db.query_table("user", f"""
        SELECT *
        FROM `user`
        WHERE
            `user`.`company` = 'Contoso'
        LIMIT 10;
        """)

    print(contoso_users_df.first_name)
```
