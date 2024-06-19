"""
A very convenient library, which contains:

    TonyDBC: A context manager class for MariaDB

To protect certain databases against accidental deletion, please set in your .env:

    PRODUCTION_DATABASES = ["important_db", "other_important_db"]

"""
import os
import io
import sys
import code
import json
import pytz
import zoneinfo
import tzlocal
import typing
import pickle
import numpy as np
import pandas as pd
import datetime
import dateutil
import copy
import shutil
import pathlib
import queue

# Use the official mariadb Python connection library
import mariadb
from mariadb.constants.CLIENT import MULTI_STATEMENTS
from .dataframe_fast import DataFrameFast, read_sql_table, get_data, DATATYPE_MAP
from .tony_utils import (
    serialize_table,
    deserialize_table,
    get_current_time_string,
    get_tz_offset,
)
from .env_utils import get_env_list

NULL = mariadb.constants.INDICATOR.NULL

# Max number of times to re-try a command if connection is lost
MAX_RECONNECTION_ATTEMPTS = 3


def check_connection(fn):
    def conn_wrapper(self, *args, **kwargs):
        try:
            self._mariatonydbcn.ping()
        except mariadb.Error:
            _ping_str = "Ping failed: Restarting mariadb connection"
            if self._l is None:
                print(_ping_str)
            else:
                self._l.info(_ping_str)
            self.__enter__()
        result = fn(self, *args, **kwargs)
        return result

    return conn_wrapper


class __TonyDBCOnlineOnly:
    """
    Generic context manager for a database connection and for several
    common queries and commands made against a database.

    Creates a persistent database connection, using the mariadb connector,
        the official Python connection library for mariadb

    You can change the database it points to by running .use(new_database)

    """

    def __init__(
        self,
        host,
        user,
        password,
        database,
        port=3306,
        media_to_deserialize=[],
        autocommit=True,
        l=None,
        prefix="",
        lost_connection_callback=None,
        session_timezone=None,
    ):
        """
        Parameters:
            lost_internet_callback: this function will be called if the
                                    connection to the database is lost
                                    while running an SQL command.
        """
        required_fields = [host, user, password, database, port]
        if any(k is None or (type(k) == str and k == "") for k in required_fields):
            raise AssertionError("TonyDBC: Not all credentials provided.")

        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.port = port
        self.media_to_deserialize = media_to_deserialize
        self.autocommit = autocommit
        self._l = l
        self._lost_connection_callback = lost_connection_callback
        self.using_temp_conn = False

        # Used to preface all logging statements
        self.prefix = prefix

        self.prelim_session_timezone = session_timezone

    def __enter__(self):
        self.log(f"Connecting to database {self.database}.")
        try:
            # DOCS:
            # https://github.com/mariadb-corporation/mariadb-connector-python/blob/f26934540d9506b6079ad92f603b697c761622de/mariadb/mariatonydbcnection.c#L301
            self._mariatonydbcn = mariadb.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                client_flag=MULTI_STATEMENTS,
                autocommit=self.autocommit,
                # connect_timeout=30,  # Default is 0, in seconds (have not yet tried this)
                read_timeout=3600,  # 15,  # in seconds
                write_timeout=3600,  # 20  # in seconds
            )
        except mariadb.InterfaceError as e:
            raise Exception(
                f"Could not connect to database. {e}\n"
                f"Possible causes: \n"
                f"(1) Your local Internet connection is down\n"
                f"(2) The server {self.host} is down\n"
                f"(3) The mariadb docker container on the server is down\n"
                f"(4) You are using a VPN.  Disable VPN please."
            )
        except mariadb.OperationalError as e:
            # e.g. mariadb.OperationalError: Can't connect to server on 'fling.ninja' (10060)
            self.log(f"Got mariadb.OperationalError {e}")
            if self._lost_connection_callback is not None:
                self.log(
                    "mariadb.OperationalError during initial connect(): "
                    "Calling the callback provided for this situation."
                )
                self._lost_connection_callback()

            raise Exception(
                f"Database server is down?  Got mariadb.OperationalError {e}"
            )

        # If the connection fails, attempt to reconnect (cool!)
        self._mariatonydbcn.auto_reconnect = True

        if not self._mariatonydbcn.open:
            if self._lost_connection_callback is not None:
                self.log("not self._mariatonydbcn.open so calling the callback")
                self._lost_connection_callback()

            raise AssertionError("Database could not open via mariadb")

        # Set the timezone for the first time
        self.set_timezone(self.prelim_session_timezone)

    def set_timezone(self, session_timezone=None):
        """
        Set the IANA session time zone for display and input purposes for TIMEZONE fields
        e.g. session_timezone = "Asia/Bangkok" => session_time_offset = "+07:00"

        if None specified, default to os.environ["DEFAULT_TIMEZONE"]
        """
        if session_timezone is None:
            self.session_timezone = os.environ["DEFAULT_TIMEZONE"]
        elif session_timezone == self.session_timezone:
            self.log(f"Time zone unchanged ({self.session_timezone})")
            return
        else:
            self.session_timezone = session_timezone

        if not self.session_timezone in zoneinfo.available_timezones():
            raise AssertionError(
                f"The session timezone specified, {self.session_timezone}, "
                "is not a valid IANA time zone (for a complete list,"
                " see zoneinfo.available_timezones())."
            )

        # Require that the session timezone be the same as the system timezone
        system_timezone = str(tzlocal.get_localzone())

        if self.session_timezone != system_timezone:
            self.log(
                f"WARNING: The session timezone specified, {self.session_timezone}, "
                f", is not the same as the system time "
                f"zone {system_timezone}.  This might be fine if you are rendering data for another time zone, "
                "but it WILL cause problems if you are APPENDING or INSERTING data if you are not careful."
            )

        self.session_time_offset = get_tz_offset(
            iana_tz=self.session_timezone
        )  # e.g. "+07:00"

        # Set the database session to be this time zone
        # any INSERT or UPDATE commands executed on TIMEZONE data types will assume
        # the data was expressed in this time zone.
        self.execute(f"SET @@session.time_zone:='{self.session_time_offset}';")
        self.default_tz = dateutil.tz.gettz(self.session_timezone)
        assert self.default_tz is not None

        self.log(
            f"TonyDBC: @@session.time_zone set to {self.session_time_offset}, "
            f"which is IANA timezone {self.session_timezone}."
        )

        return self

    def __exit__(self, exit_type, value, traceback):
        # Commit all pending transactions if necessary
        if not self.autocommit:
            self.log(f"TonyDBC commit pending transactions.")
            self.commit()
        self._mariatonydbcn.close()
        self.log(f"TonyDBC mariadb connection closed.")

        if exit_type == SystemExit:
            self.log(f"TonyDBC: user typed exit() in interpreter.")
        elif not exit_type is None:
            self.log(
                f"TonyDBC: exception triggered __exit__: \n"
                f"exit_type: {exit_type}\nvalue: {value}\ntraceback: {traceback}"
            )
            return False  # Do not handle the exception; propagate it up
        else:
            pass
            self.log(f"TonyDBC: normal __exit__ successful.")
            # (No need to return a value since `exit_type` is None.)

    def start_temp_conn(self):
        """Start using a temporary database connection"""
        if self.using_temp_conn:
            raise AssertionError("You are already using your temporary connection.")
        self.using_temp_conn = True
        self._mariatonydbcn_old = self._mariatonydbcn
        # Create a new connection
        self.__enter__()

    def close_temp_conn(self):
        """Stop using the temporary database connection"""
        if not self.using_temp_conn:
            raise AssertionError("You are not using your temporary connection.")
        # Close the temporary connection
        self.__exit__(exit_type=None, value=None, traceback=None)
        # Restore the original connection
        self._mariatonydbcn = self._mariatonydbcn_old
        self.using_temp_conn = False

    def cursor(self):
        return self._mariatonydbcn.cursor()

    def begin_transaction(self):
        """This must be followed by a commit later.  Turns off autocommit."""
        # We must ensure autocommit is not on to be able to create a transaction
        if self.autocommit:
            self.execute("SET AUTOCOMMIT=0;")

        self._mariatonydbcn.begin()

    def commit(self):
        try:
            self._mariatonydbcn.commit()
        except mariadb.InterfaceError as e:
            # e.g. Lost connection to server during query
            if self._lost_connection_callback is not None:
                self.log(
                    "mariadb.InterfaceError during commit: "
                    "Calling the callback provided for this situation."
                )
                self._lost_connection_callback()

            raise Exception(f"Database server is down?  Got mariadb.InterfaceError {e}")

        # Turn autocommit back on if that's what was originally requested
        if self.autocommit:
            self.execute("SET AUTOCOMMIT=1;")

    def log(self, msg: str):
        """Internal logging; use normal print if a logger was not provided"""
        if self.prefix != "":
            msg = f"{self.prefix} | {msg}"

        if self._l is None:
            print(msg)
        else:
            self._l.info(msg)

    def column_info(self, table_name=None):
        """Returns the column information.
        Parameters:
            table_name: string.  If None, returns column info for ALL tables.
        """
        clauses = [f"TABLE_SCHEMA = '{self.database}'"]
        if not table_name is None:
            clauses.append(f"TABLE_NAME = '{table_name}'")

        r = self.get_data(
            f"SELECT * FROM INFORMATION_SCHEMA.COLUMNS "
            f"WHERE {'AND '.join(clauses)};"
        )

        return pd.DataFrame(r)

    @property
    def last_insert_id(self):
        # No need to do a commit here necessarily...
        return self.get_data("SELECT LAST_INSERT_ID() AS id;")[0]["id"]

    def use(self, new_database: str):
        """Change databases"""
        self.database = new_database
        # Reset current connections
        self.__exit__()
        # Make new connections
        self.__enter__()

    def iso_timestamp_to_session_time_zone(self, iso_timestamp_string):
        """e.g. converts an ISO-formatted string to the database session's time zone,
        self.DEFAULT_TIME_OFFSET, which is useful when adding to a mariadb TIMESTAMP
        which assumes the time is formatted as the session time zone.
        e.g. self.DEFAULT_TIME_OFFSET = "+07:00",
            "2023-03-03T09:00:00+07:00" ->
            "2023-03-03 09:00:00"
                (since 9-7+7 = 9)

            or

            "2023-03-03T09:00:00+03:00" ->
            "2023-03-03 13:00:00"
                (since 9-3+7 = 13)
        """
        try:
            ts = dateutil.parser.parse(iso_timestamp_string)
        except dateutil.parser._parser.ParserError as e:
            raise ValueError(f"Timestamp '{iso_timestamp_string}' is invalid. {e}")
        else:
            iso_ts = ts.astimezone(self.default_tz)  # dateutil.tz.UTC
            return iso_ts.strftime("%Y-%m-%d %H:%M:%S")

    def write_dataframe(self, df, table_name, if_exists="replace", index=False):
        """Write all rows of a dataframe to the database
        Assumes the fields are the same in number and data type
        """
        df0 = DataFrameFast(df)
        df0.to_sql(
            name=table_name, con=self._mariatonydbcn, if_exists=if_exists, index=index
        )

    def update_table(self, table_name: str, df):
        """Perform an UPDATE with appropriate serialization.

        This is the spiritual sister of `append_to_table` (i.e. it uses the same fancy semantics
        to serialize before inserting, which is useful for JSON-ifying columns like the `OCR` column)

        e.g. if one of the fields is OCR, it will serialize as a string: "['hello', 'world']"

        Parameters:
            table_name is the name of the table
            df: is a dataframe containing the correct indexes to update, and
                containing the correct columns to be updated

        """
        if len(df) == 0:
            l.log(f"UPDATE 0 rows in {self.database}.{table_name}")
            return

        pk = self.primary_keys[table_name]
        # For vanity's sake, give the dataframe index the correct name
        df.index.name = pk
        # We don't try to cast the primary key (which is always an int)
        col_dtypes = {
            k: v for k, v in self.column_datatypes[table_name].items() if k != pk
        }

        if table_name in self.media_to_deserialize:
            columns_to_serialize = self.media_to_deserialize[table_name]
        else:
            columns_to_serialize = []

        df_serialized = serialize_table(df, col_dtypes, columns_to_serialize)

        # Append our values to the actual database table
        self.log(
            f"UPDATE {len(df_serialized)} rows in {self.database}.{table_name} "
            f"for columns: {df_serialized.columns}"
        )

        # e.g. 'UPDATE roi SET OCR = ? WHERE `id` = ?'
        update_command = f"""
            UPDATE {table_name}
            SET {', '.join([v+' = ?' for v in df_serialized.columns])}
            WHERE `id` = ?
            """

        # Make a list of tuples of all the values to be changed.
        # Put the index LAST in each tuple
        update_command_values = [
            tuple(row) + (i,) for i, row in df_serialized.iterrows()
        ]

        with self.cursor() as cursor:
            cursor.executemany(update_command, update_command_values)

    def update_blob(self, table_name, blob_column, id_value, filepath, max_size_MB=16):
        """Add a BLOB (Binary Large OBject) to the database, from a file
        This requires that the table row already exists; we are just updating it

        Note, to check for max size, you have to know the actual blob type and then specify
        the size manually since we don't actually check the blob type.  TODO: check automatically

        For now, we assume MEDIUMBLOB (16 MB)

        max size for a BLOB is 64 KB
        max size for a MEDIUMBLOB is 16 MB
        max size for a LONGBLOB is 4 GB
        """
        assert os.path.isfile(filepath)

        # Check the file size of the blob
        file_size = os.path.getsize(filepath)

        # Convert size to MB
        file_size_MB = file_size // (1024**2)

        if file_size_MB > max_size_MB:
            raise AssertionError(
                f"The file {filepath} to be uploaded as a blob to {table_name}.{blob_column} is too large. "
                f" It is {file_size_MB} MB, but the maximum allowed size is {max_size_MB} MB."
            )

        # Open your file in binary mode
        with open(filepath, "rb") as blob_file:
            blob = blob_file.read()

        # Update the relevant row, adding the BLOB
        blob_mdb_obj = mariadb.Binary(blob)
        self.execute(
            command=f"UPDATE {table_name} SET {blob_column} = ? WHERE `id` = {id_value}",
            command_values=(blob_mdb_obj,),
        )

    def read_dataframe_from_table(
        self, table_name, columns_to_deserialize=[], query=None
    ):
        """Read the dataframe but don't set the index"""
        full_name = ".".join([self.database, table_name])
        df = read_sql_table(name=full_name, con=self._mariatonydbcn, query=query)

        try:
            primary_key = self.primary_keys[table_name]
        except KeyError as e:
            # KLUDGE
            primary_key = "id"

        # If the primary key was selected, then use it, otherwise leave the table with no index
        if primary_key in df.columns:
            df.set_index(primary_key, inplace=True)
            df.sort_index(inplace=True)

        # DESERIALIZE!
        # (I wrote this but it's not needed; only arrays need special deserialization
        # and this cannot be inferred from the SQL datatype which will be varchar)  :(
        # Query INFORMATION_SCHEMA to get column data types
        # col_df = self.column_info()
        # nonstring_filter = (col_df['DATA_TYPE'] != 'varchar')
        # Deserialize all columns in tables which are NOT varchar
        # table_filter = (col_df['TABLE_NAME'] == table)
        # cols = list(col_df.loc[
        #    table_filter & nonstring_filter,
        #    'COLUMN_NAME'])
        df = deserialize_table(
            df,
            columns_to_deserialize=columns_to_deserialize,
            session_timezone=self.session_timezone,
        )

        return df

    def get_data(self, query: str):
        return get_data(self._mariatonydbcn, query)

    @property
    def databases(self):
        return [d["Database"] for d in self.get_data("SHOW DATABASES;")]

    @property
    def users(self):
        return [
            (d["Host"], d["User"]) for d in self.get_data("SELECT * FROM mysql.user")
        ]

    def show_grants(self, username: str, host="%"):
        return self.get_data(f"SHOW GRANTS FOR '{username}'@'{host}';")

    @property
    def production_databases(self):
        try:
            return self.__production_databases
        except AttributeError:
            # Protect these databases - test harness will check to ensure
            # it doesn't try to drop these databases (will raise AssertionError)
            if "PRODUCTION_DATABASES" in os.environ:
                dbs = get_env_list("PRODUCTION_DATABASES")
            else:
                dbs = []

            dbs += [
                "information_schema",
                "mysql",
                "performance_schema",
                "sys",
            ]
            # Remove duplicates
            self.__production_databases = sorted(set(dbs))

            return self.__production_databases

    def drop_database(self, database):
        if database in self.production_databases:
            raise AssertionError(
                f"DANGER DANGER!  You are trying to drop {database}, "
                "a production database. Please talk to your manager for advice on what "
                " to do.  Do NOT bypass this error."
            )

        if database in self.databases:
            self.log(f"DROP {database} since it already exists")
            # First unlock the tables so they won't prevent us from dropping the database
            self.execute(f"USE {database}; UNLOCK TABLES;")
            self.execute(f"DROP DATABASE {database};")

    def post_data(self, query: str):
        with self.cursor() as cursor:
            record = cursor.execute(query)

        return record

    def post_datalist(self, query: str, insert_data: list):
        with self.cursor() as cursor:
            record = cursor.executemany(query, insert_data)
        return record

    def execute(self, command: str, command_values=None, before_retry_cmd=None):
        """Parameters:
        command_values: a tuple of values [optional]
        """

        attempts_remaining = MAX_RECONNECTION_ATTEMPTS
        while attempts_remaining > 0:
            with self.cursor() as cursor:
                try:
                    # TODO: Consider adding
                    # cursor.execute("SET time_zone = '+00:00'")
                    # every time though???
                    # https://stackoverflow.com/questions/1136437/
                    if (
                        not before_retry_cmd is None
                        and attempts_remaining < MAX_RECONNECTION_ATTEMPTS
                    ):
                        cursor.execute(before_retry_cmd)
                    if command_values is not None:
                        cursor.execute(command, command_values)
                    else:
                        cursor.execute(command)
                except mariadb.InterfaceError as e:
                    # Try reconnecting by explicitly calling the TonyDBC enter method
                    # (not any child class overridden version) by doing this instead
                    # of saying self.__enter__()
                    self.log(
                        f"Reconnecting to mariadb; attempting command {command} again. "
                        f"Attempts remaining {attempts_remaining} BEFORE this attempt."
                    )
                    __TonyDBCOnlineOnly.__enter__(self)
                    attempts_remaining -= 1
                except Exception as e:
                    self.log(
                        f"mariadb execute command failed: {command} with error {e}"
                    )
                    code.interact(local=locals(), banner=f"{e}")
                else:
                    # if False and cursor.lastrowid is None:
                    #    raise AssertionError(
                    #        f"An error occurred with one of the commands {command}; lastrowid is None"
                    #    )
                    break

            # mariadb.InterfaceError: Lost connection to server during query
            # mariadb.OperationalError: Can't connect to server on 'fling.ninja' (10060)
            # mariadb.InterfaceError: Server has gone away

    def execute_script(
        self, script_path: str, get_return_values=False, cur_database=None
    ):
        return_values = []
        # Read the SQL schema file
        with open(script_path, "r") as file:
            script_string = file.read()

        if cur_database is None:
            cur_database = self.database

        # KLUDGE
        if "DELIMITER" in script_string:
            print(
                "Sorry, the python Mariadb connector doesn't seem to be "
                "able to handle the DELIMITER directive.  Can you please go "
                "to HeidiSQL or another MySQL terminal and run this script "
                "manually.  Once it's completed, come back here and press Ctrl-Z."
                f"\nFirst run: USE {cur_database};"
                f"\nScript to run: {script_path}"
            )
            # Wait for user command
            code.interact(local=locals(), banner=f"{e}")

            if get_return_values:
                return []
            else:
                return

        with self.cursor() as cursor:
            # Get a list of each separate command
            commands = [c + ";" for c in script_string.split(";") if len(c) > 0]
            for command in commands:
                # Ensure command is not simply empty
                command = command.strip()
                if len(command) == 0 or command == ";":
                    continue
                cursor.execute(command)

                if get_return_values:
                    if cursor.description is None:
                        # Handle the case of a command with no result set;
                        # such as `USE fling_db;` in this case fetchall() will
                        # raise a ProgrammingError so we should check for that first
                        return_values.append(None)
                    else:
                        records0 = cursor.fetchall()
                        # Get the field names
                        fields = cursor.description
                        # Get a list of dicts with proper field names
                        # (i.e. records in the pandas sense)
                        records1 = [
                            {
                                fields[i][0]: field_value
                                for i, field_value in enumerate(v)
                            }
                            for v in records0
                        ]
                        return_values.append(records1)

        if get_return_values:
            return return_values

    @property
    def primary_keys(self):
        """Returns a dict of all tables in the database
        and the field which is the primary key  e.g. {'drone': 'id'}
        NOTE: assumes only one primary key
        """
        try:
            return self._primary_keys
        except AttributeError:
            r = self.get_data(
                query=f"""
                SELECT TABLE_NAME, COLUMN_NAME
                FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                WHERE 
                    TABLE_SCHEMA = '{self.database}' AND
                    CONSTRAINT_NAME = 'PRIMARY'
                """
            )
            self._primary_keys = {v["TABLE_NAME"]: v["COLUMN_NAME"] for v in r}
            return self._primary_keys

    @property
    def column_datatypes(self):
        """Returns a dict of dicts with Python data types of each column, e.g.
        {
            "warehouse" : {
                ('warehouse_name', str),
                ('warehouse_version', int)
            },
            ...
        }
        """
        try:
            return self._column_datatypes
        except AttributeError:
            r = self.get_data(
                query=f"""
                SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
                FROM INFORMATION_SCHEMA.`COLUMNS`
                WHERE
                    TABLE_SCHEMA = '{self.database}'
                """
            )
            tables = {v["TABLE_NAME"] for v in r}
            self._column_datatypes = {
                t: {
                    v["COLUMN_NAME"]: DATATYPE_MAP[v["DATA_TYPE"]]
                    for v in r
                    if v["TABLE_NAME"] == t
                }
                for t in tables
            }
            return self._column_datatypes

    def non_primary_keys(self, table):
        """Returns a list of all non-primary keys for a table"""
        pk = self.primary_keys[table]
        cols = self.column_datatypes[table].keys()
        return [c for c in cols if c != pk]

    @property
    def connection_params(self):
        """Useful for making a new connection.  A dict of all you need.

        e.g.
        db = TonyDBC(...)  # First connection
        params = db.connection_params
        db2 = TonyDBC(**params)  # Second connection
        """
        return {
            "host": self.host,
            "database": self.database,
            "user": self.user,
            "password": self.password,
            "port": self.port,
            "media_to_deserialize": self.media_to_deserialize,
        }

    def insert_row_all_string(self, table, row_dict):
        """Insert a single row into a table, as all string
        Note that the row_dict is required to be all string or stringifyable
        Note that it does not autocommit
        """
        if len(row_dict) == 0:
            return

        stringified_row_dict = {
            k: str(v)
            for k, v in row_dict.items()
            # Skip any columns which are to be NULL and rely on DEFAULT VALUE because it won't accept
            # NULL aka the special mariadb NULL indicator for NULL values
            if (v is not None) and (str(v) != "None")
        }
        # We must sanitize the values to avoid things like single quotes breaking the INSERT
        # so we will pass a list of values to self.execute instead
        command = f"""
            INSERT INTO {table} 
                       ({', '.join(stringified_row_dict.keys())}) 
                VALUES ({", ".join(['%s' for _ in stringified_row_dict])})
            """
        command_values = list(map(str, stringified_row_dict.values()))
        self.execute(command=command, command_values=command_values)

    def append_to_table(self, table, df, return_reindexed=False):
        pk = self.primary_keys[table]
        # For vanity's sake, give the dataframe index the correct name
        # but really we never actually use this index since we let the
        # database autoincrement during the insert
        df.index.name = pk
        # We don't try to cast the primary key (which is always an int)
        col_dtypes = {k: v for k, v in self.column_datatypes[table].items() if k != pk}

        if table in self.media_to_deserialize:
            columns_to_serialize = self.media_to_deserialize[table]
        else:
            columns_to_serialize = []

        df_serialized = serialize_table(df, col_dtypes, columns_to_serialize)

        # Append our values to the actual database table
        self.log(f"INSERT {len(df_serialized)} rows in {self.database}.{table}")
        self.write_dataframe(df_serialized, table, if_exists="append", index=False)

        # Return the dataframe with the actual index AUTOINCREMENTED with the correct numbers
        # This is actually GUARANTEED to work, by the way mariadb does consecutive inserts!
        # cool right?
        if return_reindexed:
            # From testing on 2023-11-24, THIS is correct:
            df.index = list(range(self.last_insert_id, self.last_insert_id + len(df)))
            # NOT this:
            # df.index = list(range(self.last_insert_id - len(df) + 1, self.last_insert_id))
            df.index.name = pk
            return df

    def query_table(self, table, query=None):
        """Query a single table and deserialize if necessary"""
        if table in self.media_to_deserialize:
            cols_to_deserialize = self.media_to_deserialize[table]
        else:
            cols_to_deserialize = []

        cur_df = self.read_dataframe_from_table(table, cols_to_deserialize, query=query)

        return cur_df

    def refresh_table(self, table):
        """Load the ENTIRE table which might be highly inefficient
        but for small scales this should be fine.
        """
        cur_df = self.query_table(table)
        setattr(self, f"{table}_df", cur_df)

    def log_to_db(self, log: dict):
        """Save log to database
        This module will get the dictionary of log to print put and add
        to the database 'server_log' table. ALL of the keys need past
        to the function with db_credentials. Dictionary format :
            log_message = {
                'log_module' : log_module,
                'log_state' : "STARTED/ERROR/INFO/WARNING/COMPLETED,
                'log_event' : 'event of the log or action',
                'log_message' : "log message",
                'log_hostname' : hostname
            }
        ps. to get the log_module : log_module = str(os.path.basename(__file__))
            to get the log_hostname : hostname = socket.gethostname()
        """
        log_template = [
            "log_module",
            "log_state",
            "log_event",
            "log_message",
            "log_hostname",
        ]
        if all(k in log_template for k in log):
            with self.cursor() as cursor:
                log["saveTime"] = get_current_time_string()
                query = f"""
                    INSERT INTO server_log (module,state,log_event,message,saveTime,_host)
                    VALUES ('{str(log['log_module'])}','{str(log['log_state'])}','{str(log['log_event'])}','{str(log['log_message'])}','{log['saveTime']}','{str(log['log_hostname'])}');
                """
                query = query.replace("'None'", "null")
                record = cursor.execute(query)
                self.log(
                    f"{log['saveTime']}| {log['log_state']} | {log['log_module']} : {log['log_event']} {log['log_message']} {log['log_hostname']}"
                )
        else:
            if "log_module" in log.keys():
                failed_module = str(log["log_module"])
            else:
                failed_module = "unknown"
            self.log(
                f"ERROR | central_log warning : missing log parameters from - {failed_module}"
            )


class TonyDBC(__TonyDBCOnlineOnly):
    """Public exposed class offering all base functions plus an offline mode

    Note: this is not a true offline mode; it still needs to be in online mode at the beginning, and end.
    But it will make it easy to enqueue many writes, even across multiple sessions.

    Use TonyDBC context manager as normal.  But if you set self.is_online = False, then it will enqueue
    any db updates until you set it back to self.is_online = True.

    If the TonyDBC exits before you set is_online back to True, then the updates will be pickled.
    If you load TonyDBC again, it will load briefly in online mode then flip to offline mode.

    Warning: Like TonyDBC itself, this is intended to be run in a single thread and not to be run in multiple threads

    Usage example:
        with TonyDBC(**connect_kwargs) as db:
            # now you can read from DB
            # ...

            db.is_online = False
            # write a lot of stuff into the queue...
            # ....

            db.is_online = True  # Flush updates into the database

        with TonyDBC(**connect_kwargs) as db:
            # Automatically, TonyDBC will flush all updates not
            # saved to DB from last connection (if db.is_online = True was not done)
            pass

    """

    def __init__(self, *args, **kwargs):
        mb = "MEDIA_BASE_PATH_PRODUCTION"
        if mb in os.environ:
            pickle_base_path = os.environ[mb]
        else:
            # Default to the script path if no pickle path was provided
            pickle_base_path = sys.path[0]

        self.__offline_status = "online"
        self.__offline_pickle_path = os.path.join(
            pickle_base_path, "dbcon_pickle.PICKLE"
        )
        self.__update_queue = queue.Queue()
        super().__init__(*args, **kwargs)

    @property
    @check_connection
    def is_online(self):
        return self.__offline_status != "offline"

    @is_online.setter
    def is_online(self, value: bool):
        """If the user sets is_online to True, and it was not before, then flush updates."""
        assert type(value) == bool
        if self.__offline_status == "offline" and value:
            self.__offline_status = "flushing"
            self.flush_updates()
            self.__offline_status = "online"
        elif self.__offline_status == "online" and not value:
            self.__offline_status = "offline"

    def flush_updates(self):
        """Make all the updates to the tables that we have been saving up"""
        # Pickle our updates in case of error
        try:
            if not self.__update_queue.empty():
                self.log(f"Pickling before flushing to be safe")
                self.pickle_updates()
                self.log(f"Flushing {self.__update_queue.qsize()} database updates")
                while not self.__update_queue.empty():
                    method, kwargs = self.__update_queue.get()
                    getattr(self, method)(**kwargs)
                self.log("Flushing database updates - DONE")
                self.log("Backing up temp pickle")
                shutil.move(
                    self.__offline_pickle_path, self.__offline_pickle_path + ".BAK"
                )
        except AttributeError as e:
            code.interact(banner=f"Bad TonyDBC {e}", local=locals())

    def __enter__(self):
        super().__enter__()

        if not os.path.isfile(self.__offline_pickle_path):
            return self

        self.is_online = False
        # Try loading unsaved stuff if it exists
        try:
            with open(self.__offline_pickle_path, "rb") as pickle_file:
                update_list = pickle.load(pickle_file)
        except EOFError as e:
            self.log(
                f"Deleting corrupt pickle file {self.__offline_pickle_path} that is 0 bytes"
            )
            os.remove(self.__offline_pickle_path)
        else:
            self.log(
                f"Finished loading DB updates pickle {self.__offline_pickle_path}."
            )
            # Now get rid of the pickle so we don't use it again
            shutil.move(self.__offline_pickle_path, self.__offline_pickle_path + ".BAK")

        for v in update_list:
            self.__update_queue.put(v)

        # Now flush the pickled updates
        self.is_online = True

        return self

    def pickle_updates(self):
        # Pickle our queue
        if not self.__update_queue.empty():
            self.log(
                f"Pickling {self.__update_queue.qsize()} updates "
                f"to {self.__offline_pickle_path}"
            )
            backup_queue = queue.Queue()
            queue_list = []
            while not self.__update_queue.empty():
                v = self.__update_queue.get()
                queue_list.append(copy.deepcopy(v))
                backup_queue.put(copy.deepcopy(v))

            with open(self.__offline_pickle_path, "wb") as pickle_file:
                pickle.dump(queue_list, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

            self.__update_queue = backup_queue

    def __exit__(self, exit_type, value, traceback):
        self.pickle_updates()

        # Clear the queue since we have now archived it
        self.__update_queue = queue.Queue()

        assert (
            self.__update_queue.empty()
        ), "update_queue is not empty even though we just cleared it"

        # Now that we have cleared the queue, we can go to online mode.
        self.is_online = True

        super().__exit__(exit_type, value, traceback)

    def start_temp_conn(self):
        if not self.is_online:
            raise AssertionError("start_temp_conn is not supported when not online")
        else:
            super(TonyDBC, self).start_temp_conn()

    def close_temp_conn(self):
        if not self.is_online:
            raise AssertionError("close_temp_conn is not supported when not online")
        else:
            super(TonyDBC, self).close_temp_conn()

    def cursor(self):
        if not self.is_online:
            raise AssertionError("cursor is not supported when not online")
        else:
            return super(TonyDBC, self).cursor()

    def begin_transaction(self):
        if not self.is_online:
            raise AssertionError("begin_transaction is not supported when not online")
        else:
            super(TonyDBC, self).begin_transaction()

    def commit(self):
        if not self.is_online:
            raise AssertionError("commit is not supported when not online")
        else:
            super(TonyDBC, self).commit()

    def get_data(self, query: str):
        if not self.is_online:
            raise AssertionError("get_data can only be used while online")
        else:
            return super(TonyDBC, self).get_data(query)

    def drop_database(self, database):
        if not self.is_online:
            raise AssertionError("Cannot drop database if not online")
        else:
            super(TonyDBC, self).drop_database(database)

    def query_table(self, table, query=None):
        if not self.is_online:
            raise AssertionError("Cannot query table if not online")
        else:
            return super(TonyDBC, self).query_table(table, query)

    def refresh_table(self, table):
        if not self.is_online:
            raise AssertionError("Cannot refresh table if not online")
        else:
            super(TonyDBC, self).refresh_table(table)

    def update_table(self, table_name, df):
        """Instead of actually updating the table, just enqueue the updates for doing later."""
        kwargs = {"table_name": table_name, "df": df}
        if self.is_online:
            super(TonyDBC, self).update_table(**kwargs)
        else:
            self.__update_queue.put(("update_table", kwargs))

    def write_dataframe(self, df, table_name, if_exists="replace", index=False):
        kwargs = {
            "df": df,
            "table_name": table_name,
            "if_exists": if_exists,
            "index": index,
        }
        if self.is_online:
            super(TonyDBC, self).write_dataframe(**kwargs)
        else:
            self.__update_queue.put(("write_dataframe", kwargs))

    def update_blob(self, table_name, blob_column, id_value, filepath, max_size_MB=16):
        kwargs = {
            "table_name": table_name,
            "blob_column": blob_column,
            "id_value": id_value,
            "filepath": filepath,
            "max_size_MB": max_size_MB,
        }
        if self.is_online:
            super(TonyDBC, self).update_blob(**kwargs)
        else:
            self.__update_queue.put(("update_blob", kwargs))

    def post_data(self, query: str):
        kwargs = {"query": query}
        if self.is_online:
            return super(TonyDBC, self).post_data(**kwargs)
        else:
            self.__update_queue.put(("post_data", kwargs))

    def post_datalist(self, query: str, insert_data: list):
        kwargs = {
            "query": query,
            "insert_data": insert_data,
        }
        if self.is_online:
            return super(TonyDBC, self).post_datalist(**kwargs)
        else:
            self.__update_queue.put(("post_datalist", kwargs))

    def execute(self, command: str, command_values=None, before_retry_cmd=None):
        kwargs = {
            "command": command,
            "command_values": command_values,
            "before_retry_cmd": before_retry_cmd,
        }
        if self.is_online:
            super(TonyDBC, self).execute(**kwargs)
        else:
            self.__update_queue.put(("execute", kwargs))

    def execute_script(
        self, script_path: str, get_return_values=False, cur_database=None
    ):
        kwargs = {
            "script_path": script_path,
            "get_return_values": get_return_values,
            "cur_database": cur_database,
        }
        if self.is_online:
            return super(TonyDBC, self).execute_script(**kwargs)
        else:
            self.__update_queue.put(("execute_script", kwargs))

    def insert_row_all_string(self, table, row_dict):
        kwargs = {
            "table": table,
            "row_dict": row_dict,
        }

        if self.is_online:
            super(TonyDBC, self).insert_row_all_string(**kwargs)
        else:
            self.__update_queue.put(("insert_row_all_string", kwargs))

    def append_to_table(self, table, df, return_reindexed=False):
        kwargs = {
            "table": table,
            "df": df,
            "return_reindexed": return_reindexed,
        }
        if self.is_online:
            return super(TonyDBC, self).append_to_table(**kwargs)
        else:
            if return_reindexed:
                raise AssertionError(
                    "TonyDBC.append_to_table: cannot return reindexed while in `offline` mode"
                )
            self.__update_queue.put(("append_to_table", kwargs))

    def log_to_db(self, log: dict):
        kwargs = {"log": log}
        if self.is_online:
            super(TonyDBC, self).log_to_db(**kwargs)
        else:
            self.__update_queue.put(("log_to_db", kwargs))
