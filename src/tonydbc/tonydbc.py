"""
A very convenient library, which contains:

    TonyDBC: A context manager class for MariaDB

To protect certain databases against accidental deletion, please set in your .env:

    PRODUCTION_DATABASES = ["important_db", "other_important_db"]

To instrument TonyDBC to diagnose performance:

     simply populate the following in .env:

        AUDIT_PATH = C:\\tony_log.csv

    or to populate ONLY the database table `tony`:

        AUDIT_PATH = database

    or to disable performance logging:

        AUDIT_PATH =

DOCUMENTATION

The common core methods of TonyDBC:

    append_to_table: INSERT; returns DataFrame (optionally)
        table, df, return_reindexed=False
    query_table:     SELECT; returns a DataFrame
        table, query=None
    execute:         does not return any data
        command, command_values=None, before_retry_cmd=None
    get_data:        SELECT; returns a list of dicts.
        query
    update_blob:     UPDATE
        table_name, blob_column, id_value, filepath, max_size_MB=16

"""

import code
import copy
import inspect
import os
import pathlib
import pickle
import queue
import random
import shutil
import sys
import threading
import time
import zoneinfo
from contextlib import contextmanager
from types import TracebackType
from typing import Any, Callable, Literal, overload

import dateutil
import filelock

# Use the official mariadb Python connection library
import mariadb  # type: ignore
import pandas as pd
import tzlocal
import uuid6
from mariadb.constants.CLIENT import MULTI_STATEMENTS  # type: ignore

from .dataframe_fast import DATATYPE_MAP, FIELD_TYPE_DICT, DataFrameFast, read_sql_table
from .env_utils import get_env_list
from .tony_utils import (
    deserialize_table,
    get_next_word_after_from,
    get_payload_info,
    get_tz_offset,
    prepare_scripts,
    serialize_table,
)

# Maximum characters to output to `tony`.`query`
QUERY_AUDIT_MAX_CHARS = 1000

# Default to these environment variables if no credentials are provided
DEFAULT_CREDENTIALS = {
    "host": "MYSQL_HOST",
    "user": "MYSQL_READWRITE_USER",
    "password": "MYSQL_READWRITE_PASSWORD",
    "database": "MYSQL_DATABASE",
}

# Max number of times to re-try a command if connection is lost
MAX_RECONNECTION_ATTEMPTS = 3


def get_currentframe_method() -> str:
    cur_frame = inspect.currentframe()
    if cur_frame and hasattr(cur_frame, "f_code"):
        return cur_frame.f_code.co_name
    else:
        return "unknown"


def check_connection(fn: Callable) -> Callable:
    def conn_wrapper(self: "_TonyDBCOnlineOnly", *args: Any, **kwargs: Any) -> Any:
        try:
            self._mariatonydbcn.ping()
        except mariadb.Error:
            _ping_str = "Ping failed: Restarting mariadb connection"
            self.log(_ping_str)
            self.__enter__()
        result = fn(self, *args, **kwargs)
        return result

    return conn_wrapper


class _TonyDBCOnlineOnly:
    """
    Generic context manager for a database connection and for several
    common queries and commands made against a database.

    Creates a persistent database connection, using the mariadb connector,
        the official Python connection library for mariadb

    You can change the database it points to by running .use(new_database)

    """

    # Type annotations for instance attributes
    host: str
    user: str
    password: str
    database: str
    port: int
    session_uuid: str
    session_timezone: str
    ipath: str
    media_to_deserialize: dict[str, list[str]]
    _audit_db: "_TonyDBCOnlineOnly | None"

    def __init__(
        self,
        host: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
        port: int = 3306,
        media_to_deserialize: dict[str, list[str]] = {},
        autocommit: bool = True,
        logger_instance: Any | None = None,
        prefix: str = "",
        lost_connection_callback: Callable | None = None,
        session_timezone: str | None = None,
        interact_after_error: bool = False,
        force_no_audit: bool = False,
    ) -> None:
        """
        Parameters:
            media_to_deserialize: dict of tables with lists of columns to deserialize  e.g.
                {
                    "location_image": [
                        "bounds_i",
                        "bounds_w",
                        "bounds_clipped_w",
                        "big_box_bounds_clipped_i",
                    ]
                }

            lost_internet_callback: this function will be called if the
                                    connection to the database is lost
                                    while running an SQL command.
        """
        for field, env_key in DEFAULT_CREDENTIALS.items():
            param_value = locals().get(field, None)
            if param_value is None:
                # Try to grab the default
                if env_key in os.environ:
                    setattr(self, field, os.environ[env_key])
                else:
                    raise AssertionError(
                        f"TonyDBC: Not all credentials provided: "
                        f" e.g. {field} not provided and not in os.environ.  You must provide {DEFAULT_CREDENTIALS}"
                    )
            else:
                setattr(self, field, param_value)

            val = getattr(self, field, "")
            assert isinstance(val, str), (
                f"TonyDBC: Credential {field} ({val}) should be a string, not {type(val).__name__}"
            )
            assert len(val) > 0, (
                f"TonyDBC: Credential {field} ({val}) should be a string of length > 0."
            )

        # uuid will be set when the connection is made
        self.session_uuid = str(uuid6.uuid8())
        self.port = port
        self.media_to_deserialize = media_to_deserialize
        self.autocommit = autocommit
        self._l = logger_instance
        self._lost_connection_callback = lost_connection_callback
        self.using_temp_conn = False
        self.interact_after_error = interact_after_error

        # Used to preface all logging statements
        self.prefix = prefix

        self.prelim_session_timezone = session_timezone

        self.do_audit = False
        self._audit_db = None  # Separate TonyDBC instance for audit operations

        # For debugging purposes, we may wish to instrument all queries
        if (not force_no_audit) and "AUDIT_PATH" in os.environ:
            ipath = os.environ["AUDIT_PATH"]
            if ipath == "database":
                # Track only on the database, not locally
                self.do_audit = True
                self.ipath = ipath
            elif ipath != "":
                self.do_audit = True
                self.ipath = str(pathlib.Path(ipath).resolve())
                # Delete the instrumentation csv file if it has zero size
                if (
                    pathlib.Path(self.ipath).exists()
                    and pathlib.Path(self.ipath).stat().st_size == 0
                ):
                    pathlib.Path(self.ipath).unlink()

    def __enter__(self) -> "_TonyDBCOnlineOnly":
        self.make_connection()

        if self.do_audit:
            # Create a separate database connection specifically for audit operations
            # (we do this to avoid interfering with last_insert_id in the main connection)
            self.log(
                "Creating separate audit connection to preserve last_insert_id integrity."
            )
            # Create a separate _TonyDBCOnlineOnly instance for audit operations
            self._audit_db = _TonyDBCOnlineOnly(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                autocommit=True,  # Always use autocommit for audit operations
                lost_connection_callback=self._lost_connection_callback,
                session_timezone=self.session_timezone,
                interact_after_error=self.interact_after_error,
                # THIS IS CRUCIAL TO AVOID RECURSIVELY SETTING UP INFINITE AUDIT CONNECTIONS:
                force_no_audit=True,
            )
            # Initialize the audit connection
            self._audit_db.__enter__()

        return self

    def make_connection(self) -> None:
        self.log(f"Connecting to database {self.database} on {self.host}.")
        num_attempts = MAX_RECONNECTION_ATTEMPTS
        while True:
            if num_attempts < MAX_RECONNECTION_ATTEMPTS:
                self.log(
                    f"TonyDBC.__enter__ {num_attempts} / {MAX_RECONNECTION_ATTEMPTS} remaining"
                )
            try:
                # DOCS:
                # https://github.com/mariadb-corporation/mariadb-connector-python/blob/f26934540d9506b6079ad92f603b697c761622de/mariadb/mariatonydbcnection.c#L301
                self._mariatonydbcn = mariadb.connect(
                    host=self.host,
                    port=self.port,
                    user=self.user,
                    password=self.password,
                    database=self.database,
                    client_flag=MULTI_STATEMENTS,
                    autocommit=self.autocommit,
                    # connect_timeout=30,  # Default is 0, in seconds (have not yet tried this)
                    read_timeout=3600,  # 15,  # in seconds
                    write_timeout=3600,  # 20  # in seconds
                    local_infile=True,
                    compress=True,
                )
            except mariadb.InterfaceError as e:
                if num_attempts > 0:
                    num_attempts -= 1
                    self.log(f"num_attempts {num_attempts} here")
                    continue

                raise ConnectionError(
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

                if num_attempts > 0:
                    num_attempts -= 1
                    self.log(f"num_attempts {num_attempts} here")
                    continue

                if self._lost_connection_callback is not None:
                    self.log(
                        "mariadb.OperationalError during initial connect(): "
                        "Calling the callback provided for this situation."
                    )
                    self._lost_connection_callback()

                raise ConnectionError(
                    f"Database server is down?  Got mariadb.OperationalError {e}"
                )
            break

        # If the connection fails, attempt to reconnect (cool!)
        # but sadly due to issue https://github.com/Fling-Asia/tonydbc/issues/7
        # this will reset local_infile = False so we cannot use it.
        # instead we rely on our lost_connection_callback to reconnect
        self._mariatonydbcn.auto_reconnect = False

        if not self._mariatonydbcn.open:
            if self._lost_connection_callback is not None:
                self.log("not self._mariatonydbcn.open so calling the callback")
                self._lost_connection_callback()

            raise AssertionError("Database could not open via mariadb")

        # Set the timezone for the first time
        self.set_timezone(self.prelim_session_timezone)

        self.session_uuid = str(uuid6.uuid8())

        if self.do_audit:
            # Create the `tony` table if it doesn't exist
            current_dir = os.path.dirname(os.path.abspath(__file__))

            # Construct the full path to the log.txt file
            sql_path = pathlib.Path(current_dir) / "audit_table.sql"
            assert sql_path.is_file()
            cmd = prepare_scripts(self.database, [str(sql_path)])

            self.execute(cmd, no_tracking=True)

    def set_timezone(self, session_timezone: str | None = None) -> "_TonyDBCOnlineOnly":
        """
        Set the IANA session time zone for display and input purposes for TIMEZONE fields
        e.g. session_timezone = "Asia/Bangkok" => session_time_offset = "+07:00"

        if None specified, default to os.environ["DEFAULT_TIMEZONE"]
        """
        if session_timezone is None:
            session_timezone = os.environ["DEFAULT_TIMEZONE"]
            self.log(f"set_timezone defaulting to DEFAULT_TIMEZONE {session_timezone}")
            assert session_timezone is not None, "Error: DEFAULT_TIMEZONE is None"

        if not hasattr(self, "session_timezone"):
            self.session_timezone = session_timezone
            self.log(
                f"session_timezone was not set; now set to {self.session_timezone}"
            )
        elif session_timezone == self.session_timezone:
            self.log(f"session_timezone unchanged at {self.session_timezone}")
            return self
        else:
            self.log(
                f"session_timezone changing from {self.session_timezone} -> {session_timezone}"
            )
            self.session_timezone = session_timezone

        if self.session_timezone not in zoneinfo.available_timezones():
            raise AssertionError(
                f"The session_timezone specified, {self.session_timezone}, "
                "is not a valid IANA time zone (for a complete list,"
                " see zoneinfo.available_timezones())."
            )

        system_timezone = str(tzlocal.get_localzone())

        if self.session_timezone != system_timezone:
            self.log(
                f"WARNING: The session timezone specified, {self.session_timezone}, "
                f"is not the same as the system time "
                f"zone {system_timezone}.  This might be fine if you are rendering data for another time zone, "
                "but it WILL cause problems if you are APPENDING or INSERTING data if you are not careful."
            )

        self.session_time_offset = get_tz_offset(
            iana_tz=self.session_timezone
        )  # e.g. "+07:00"

        # Set the database session to be this time zone
        # any INSERT or UPDATE commands executed on TIMEZONE data types will assume
        # the data was expressed in this time zone.
        self.execute(
            f"SET @@session.time_zone:='{self.session_time_offset}';", no_tracking=True
        )
        self.default_tz = dateutil.tz.gettz(self.session_timezone)
        assert self.default_tz is not None

        self.log(
            f"TonyDBC: @@session.time_zone set to {self.session_time_offset}, "
            f"which is IANA timezone {self.session_timezone}."
        )

        # e.g. '+00:00'
        timezone_data = self.get_data("SELECT @@session.time_zone;")
        actual_time_offset = timezone_data[0]["@@session.time_zone"]
        assert actual_time_offset == self.session_time_offset, (
            f"Actual timezone offset {actual_time_offset} does not match expected timezone offset {self.session_time_offset} of timezone {self.session_timezone}"
        )

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # Commit all pending transactions if necessary
        if not self.autocommit:
            self.log("TonyDBC commit pending transactions.")
            self.commit()
        # Only close if connection is still open
        if (
            hasattr(self, "_mariatonydbcn")
            and self._mariatonydbcn
            and not getattr(self._mariatonydbcn, "_closed", True)
        ):
            self._mariatonydbcn.close()
            self.log("TonyDBC mariadb connection closed.")
        else:
            self.log("TonyDBC mariadb connection already closed.")

        if self.do_audit:
            assert self._audit_db is not None, (
                "Audit connection is not available at __exit__ to shut down"
            )

            # Close the separate audit connection
            self._audit_db.__exit__(None, None, None)
            self.log("TonyDBC audit connection closed.")
            self._audit_db = None

            self.log(f"TonyDBC debug logs at: `{self.host}`.`{self.database}`.`tony`")
            if str(self.ipath) != "database":
                self.log(f"TonyDBC debug logs also at: {self.ipath}")

        if isinstance(exc_type, type(SystemExit)):
            self.log("TonyDBC: user typed exit() in interpreter.")
        elif exc_type is not None:
            self.log(
                f"TonyDBC: exception triggered __exit__: \n"
                f"exc_type: {exc_type}\nvalue: {exc_val}\ntraceback: {exc_tb}"
            )
            return None  # Do not handle the exception; propagate it up
        else:
            self.log("TonyDBC: normal __exit__ successful.")
            # (No need to return a value since `exit_type` is None.)

    def start_temp_conn(self) -> None:
        """Start using a temporary database connection"""
        if self.using_temp_conn:
            raise AssertionError("You are already using your temporary connection.")
        self.using_temp_conn = True
        self._mariatonydbcn_old = self._mariatonydbcn
        # Create a new connection
        self.__enter__()

    def close_temp_conn(self) -> None:
        """Stop using the temporary database connection"""
        if not self.using_temp_conn:
            raise AssertionError("You are not using your temporary connection.")
        # Close the temporary connection
        self.__exit__(None, None, None)
        # Restore the original connection
        self._mariatonydbcn = self._mariatonydbcn_old
        self.using_temp_conn = False

    def now(self) -> pd.Timestamp:
        return pd.Timestamp.now(tz=self.session_timezone)

    def cursor(self) -> mariadb.Cursor:
        return self._mariatonydbcn.cursor()

    def begin_transaction(self) -> None:
        """This must be followed by a commit later.  Turns off autocommit."""
        # We must ensure autocommit is not on to be able to create a transaction
        if self.autocommit:
            self.execute("SET AUTOCOMMIT=0;")

        self._mariatonydbcn.begin()

    def commit(self) -> None:
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

            raise ConnectionError(
                f"Database server is down?  Got mariadb.InterfaceError {e}"
            )

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

    def column_info(self, table_name: str | None = None) -> pd.DataFrame:
        """Returns the column information.
        Parameters:
            table_name: string.  If None, returns column info for ALL tables.
        """
        clauses = [f"TABLE_SCHEMA = '{self.database}'"]
        if table_name is not None:
            clauses.append(f"TABLE_NAME = '{table_name}'")

        r = self.get_data(
            f"""
            SELECT *
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE {"AND ".join(clauses)};
            """,
            no_tracking=True,
        )

        return pd.DataFrame(r)

    @property
    def last_insert_id(self) -> int:
        """Get the last insert ID from the main connection.

        Note: This is now stable even when audit is enabled, because audit operations
        use a separate database connection that doesn't interfere with the main connection's
        last_insert_id value.
        """
        data = self.get_data("SELECT LAST_INSERT_ID() AS id;", no_tracking=True)
        return data[0]["id"]

    def use(self, new_database: str) -> None:
        """Change databases"""
        self.database = new_database
        # Reset current connections
        self.__exit__(None, None, None)
        # Make new connections
        self.__enter__()

    def iso_timestamp_to_session_time_zone(self, iso_timestamp_string: str):
        """e.g. converts an ISO-formatted string to the database session's time zone,
        self.default_tz, which is useful when adding to a mariadb TIMESTAMP
        which assumes the time is formatted as the session time zone.
        e.g. self.default_tz = "Asia/Bangkok",
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
            return iso_ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

    def write_dataframe(self, df, table_name, if_exists="replace", index=False):
        """Write all rows of a dataframe to the database
        Assumes the fields are the same in number and data type
        """
        df0 = DataFrameFast(df)
        df0.to_sql_fast(
            name=table_name,
            con=self._mariatonydbcn,
            session_timezone=self.session_timezone,
            if_exists=if_exists,
            index=index,
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
            self.log(f"UPDATE 0 rows in {self.database}.{table_name}")
            return

        pk = self.get_primary_key(table=table_name, default="id")
        # For vanity's sake, give the dataframe index the correct name
        df.index.name = pk
        # We don't try to cast the primary key (which is always an int)
        col_dtypes = {
            k: v
            for k, v in self.get_column_datatypes(table=table_name).items()
            if k != pk
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
            SET {", ".join([v + " = ?" for v in df_serialized.columns])}
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
        self, table_name: str, query: str, columns_to_deserialize: list[str] = []
    ) -> pd.DataFrame:
        """Read the dataframe but don't set the index"""
        full_name = ".".join([self.database, table_name])
        df = read_sql_table(name=full_name, con=self, query=query)

        primary_key = self.get_primary_key(table=table_name, default="id")

        # If the primary key was selected, then use it, otherwise leave the table with no index
        if primary_key in df.columns:
            df.set_index(primary_key, inplace=True)
            df.sort_index(inplace=True)

        # DESERIALIZE!
        # (I wrote this but it's not needed; only arrays need special deserialization
        # and this cannot be inferred from the SQL datatype which will be varchar)  :(
        # Query INFORMATION_SCHEMA to get column data types
        # col_df = self.column_info(con=self._mariatonydbcn)
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

    @overload
    def get_data(
        self,
        query: str,
        before_retry_cmd: str | None = ...,
        no_tracking: bool = ...,
        return_type_codes: Literal[False] = ...,
    ) -> list[dict[str, Any]]: ...

    @overload
    def get_data(
        self,
        query: str,
        before_retry_cmd: str | None = ...,
        no_tracking: bool = ...,
        return_type_codes: Literal[True] = ...,
    ) -> tuple[list[dict[str, Any]], dict[Any, str]]: ...

    def get_data(
        self,
        query: str,
        before_retry_cmd: str | None = None,
        no_tracking: bool = False,
        return_type_codes: bool = False,
    ) -> Any:
        if self.do_audit and not no_tracking:
            started_at = self.now()

        attempts_remaining = MAX_RECONNECTION_ATTEMPTS
        while attempts_remaining > 0:
            with self.cursor() as cursor:
                try:
                    # TODO: Consider adding
                    # cursor.execute("SET time_zone = '+00:00'")
                    # every time though???
                    # https://stackoverflow.com/questions/1136437/
                    if (
                        before_retry_cmd is not None
                        and attempts_remaining < MAX_RECONNECTION_ATTEMPTS
                    ):
                        cursor.execute(before_retry_cmd)

                    cursor.execute(query)
                    records0 = cursor.fetchall()
                    fields = cursor.description
                    """
                    Cursor.description is an 11-item tuple with the following:
                        name
                        type_code   can be looked up via FIELD_TYPE_DICT
                        display_size
                        internal_size
                        precision
                        scale
                        null_ok
                        field_flags
                        table_name
                        original_column_name
                        original_table_name
                    """
                except mariadb.InterfaceError:
                    self.log(
                        f"Reconnecting to mariadb; attempting query again. "
                        f"Attempts remaining {attempts_remaining} BEFORE this attempt."
                    )
                    self.make_connection()
                    attempts_remaining -= 1
                except Exception as e:
                    if self.interact_after_error:
                        self.log(
                            f"mariadb execute command failed: {query} with error {e}"
                        )
                        code.interact(local=locals(), banner=f"{e}")
                    else:
                        raise Exception(
                            f"mariadb execute command failed: {query} with error {e}"
                        )
                else:
                    # if False and cursor.lastrowid is None:
                    #    raise AssertionError(
                    #        f"An error occurred with one of the commands {command}; lastrowid is None"
                    #    )
                    break
                try:
                    cursor.execute("SHOW WARNINGS;")
                except mariadb.InterfaceError as e:
                    self.log(
                        f"Cannot run cursor.execute('SHOW WARNINGS;') because mariadb.InterfaceError {e}."
                    )
                    warnings = []
                else:
                    # Check for warnings
                    warnings = cursor.fetchall()

            if warnings:
                for warning in warnings:
                    self.log(warning)

        records = [
            {fields[i][0]: field_value for i, field_value in enumerate(v)}
            for v in records0
        ]

        if self.do_audit and not no_tracking:
            self._save_instrumentation(
                method=get_currentframe_method(),
                table=None,
                query=query,
                started_at=started_at,
                **get_payload_info(records),
            )

        if return_type_codes:
            # Get a list of dicts with proper field names
            # (i.e. records in the pandas sense)
            # (e.g. {'media_object_id': 'LONGLONG', 'full_path': 'VAR_STRING'}
            type_codes = {v[0]: FIELD_TYPE_DICT[v[1]] for v in fields}

            return records, type_codes
        else:
            return records

    @property
    def databases(self):
        return [d["Database"] for d in self.get_data("SHOW DATABASES;")]

    @property
    def users(self):
        return [
            (d["Host"], d["User"])
            for d in self.get_data("SELECT * FROM mysql.user", no_tracking=True)
        ]

    def show_grants(self, username: str, host="%"):
        return self.get_data(
            f"SHOW GRANTS FOR '{username}'@'{host}';", no_tracking=True
        )

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

    def post_datalist(self, query: str, insert_data: list[Any]) -> Any:
        with self.cursor() as cursor:
            record = cursor.executemany(query, insert_data)
        return record

    def execute(
        self,
        command: str,
        command_values: tuple | None = None,
        before_retry_cmd: str | None = None,
        no_tracking: bool = False,
        log_progress: bool = False,
    ) -> None:
        """Parameters:
        command_values: a tuple of values [optional]
        """
        if log_progress:
            self.log(f"Executing {command[:50]}")

            stop_wait_message = (
                threading.Event()
            )  # Event to signal the waiting thread to stop

            def wait_message():
                log_increment = 0
                while not stop_wait_message.is_set():
                    if log_increment >= 50:
                        # Every 50 x 0.1s = 5s, we will notify that the query is still going...
                        self.log(f"Waiting for {command[:50]} ...")
                        log_increment = 0
                    # Sleep for only a short time to avoid slowing things down when we .join
                    # if the query ends early
                    time.sleep(0.1)
                    log_increment += 1

            wait_thread = threading.Thread(target=wait_message, daemon=True)
            wait_thread.start()

        if self.do_audit and not no_tracking:
            started_at = self.now()

        attempts_remaining = MAX_RECONNECTION_ATTEMPTS
        while attempts_remaining > 0:
            with self.cursor() as cursor:
                try:
                    # TODO: Consider adding
                    # cursor.execute("SET time_zone = '+00:00'")
                    # every time though???
                    # https://stackoverflow.com/questions/1136437/
                    if (
                        before_retry_cmd is not None
                        and attempts_remaining < MAX_RECONNECTION_ATTEMPTS
                    ):
                        cursor.execute(before_retry_cmd)
                    if command_values is not None:
                        cursor.execute(command, command_values)
                    else:
                        cursor.execute(command)
                except mariadb.InterfaceError:
                    # Try reconnecting by explicitly calling the TonyDBC enter method
                    # (not any child class overridden version) by doing this instead
                    # of saying self.__enter__()
                    self.log(
                        f"Reconnecting to mariadb; attempting command {command} again. "
                        f"Attempts remaining {attempts_remaining} BEFORE this attempt."
                    )
                    self.make_connection()
                    attempts_remaining -= 1
                except Exception as e:
                    if self.interact_after_error:
                        self.log(
                            f"mariadb execute command failed: {command} with error {e}"
                        )
                        code.interact(local=locals(), banner=f"{e}")
                    else:
                        if log_progress:
                            stop_wait_message.set()
                            wait_thread.join()
                        raise Exception(e)
                else:
                    # if False and cursor.lastrowid is None:
                    #    raise AssertionError(
                    #        f"An error occurred with one of the commands {command}; lastrowid is None"
                    #    )
                    break
                cursor.execute("SHOW WARNINGS;")
                # Check for warnings
                warnings = cursor.fetchall()

            if warnings:
                for warning in warnings:
                    self.log(warning)

            # mariadb.InterfaceError: Lost connection to server during query
            # mariadb.OperationalError: Can't connect to server on 'fling.ninja' (10060)
            # mariadb.InterfaceError: Server has gone away

        # Signal the waiting thread to stop
        if log_progress:
            stop_wait_message.set()
            wait_thread.join()
            self.log(f"Executing {command[:50]} - DONE")

        if self.do_audit and not no_tracking:
            self._save_instrumentation(
                method=get_currentframe_method(),
                table=None,
                query=command,
                started_at=started_at,
                **get_payload_info(command_values),
            )

    def execute_script(
        self, script_path: str, get_return_values=False, cur_database=None
    ):
        return_values: list[list[dict[str, Any]] | None] = []
        # Read the SQL schema file
        with open(script_path, "r") as file:
            script_string = file.read()

        if cur_database is None:
            cur_database = self.database

        # KLUDGE
        if "DELIMITER" in script_string:
            self.log(
                "Sorry, the python Mariadb connector doesn't seem to be "
                "able to handle the DELIMITER directive.  Can you please go "
                "to HeidiSQL or another MySQL terminal and run this script "
                "manually.  Once it's completed, come back here and press Ctrl-Z."
                f"\nFirst run: USE {cur_database};"
                f"\nScript to run: {script_path}"
            )
            # Wait for user command
            code.interact(local=locals(), banner="Run script manually please")

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

    def get_primary_key(self, table: str, default: str | None = None):
        try:
            return self.primary_keys[table]
        except KeyError as _:
            # Try refreshing before returning the error
            self.refresh_primary_keys()
            try:
                return self.primary_keys[table]
            except KeyError as _:
                # Maybe it's a temporary table or something; try to DESCRIBE it.
                result = self.get_data(f"DESCRIBE {table};", no_tracking=True)
                if len(result) == 0:
                    raise KeyError(f"Table {table} does not seem to exist.")
                primary_keys = [col for col in result if col["Key"] == "PRI"]
                if len(primary_keys) == 0:
                    if default is not None:
                        return default
                    raise KeyError(
                        f"Table {table} is not in our list of tables with "
                        f"primary keys: {self.primary_keys}.  (And no default was provided)"
                    )
                elif len(primary_keys) > 1:
                    raise KeyError(f"Table {table} has more than one primary key!")
                else:
                    return primary_keys[0]["Field"]

    def refresh_primary_keys(self):
        r = self.get_data(
            query=f"""
            SELECT TABLE_NAME, COLUMN_NAME
            FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
            WHERE
                TABLE_SCHEMA = '{self.database}' AND
                CONSTRAINT_NAME = 'PRIMARY'
            """,
            no_tracking=True,
        )
        self._primary_keys = {v["TABLE_NAME"]: v["COLUMN_NAME"] for v in r}

    @property
    def primary_keys(self):
        """Returns a dict of all tables in the database
        and the field which is the primary key  e.g. {'drone': 'id'}
        NOTE: assumes only one primary key
        """
        try:
            return self._primary_keys
        except AttributeError:
            self.refresh_primary_keys()
            return self._primary_keys

    def get_column_datatypes(self, table: str) -> dict[str, Any]:
        try:
            return self.column_datatypes[table]
        except KeyError:
            # Maybe it's a temporary table or something; try to DESCRIBE it.
            result = self.get_data(f"DESCRIBE {table};")
            if len(result) == 0:
                raise KeyError(f"Table {table} does not seem to exist.")
            # We need to map Types like "bigint(20)" to "bigint" to match our DATATYPE_MAP
            col_datatypes = {
                v["Field"]: DATATYPE_MAP[v["Type"].split("(")[0].strip().lower()]
                for v in result
            }
            return col_datatypes

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
                """,
                no_tracking=True,
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
        pk = self.get_primary_key(table=table)
        cols = self.get_column_datatypes(table=table).keys()
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
                       ({", ".join(stringified_row_dict.keys())})
                VALUES ({", ".join(["%s" for _ in stringified_row_dict])})
            """
        command_values = list(map(str, stringified_row_dict.values()))
        self.execute(command=command, command_values=command_values)

    @contextmanager
    def temp_id_table(self, id_list):
        """Create a temporary table filled with ids for JOINing purposes"""
        # Generate a random temporary table name
        temp_table_name = f"temp_loc_ids_{random.randint(1000, 9999)}"

        try:
            # Create the temporary table
            self.execute(
                f"CREATE TEMPORARY TABLE {temp_table_name} (`id` BIGINT PRIMARY KEY);"
            )

            # Insert the data into the temporary table
            df = pd.DataFrame(id_list, columns=["id"])
            self.append_to_table(temp_table_name, df)

            # Yield the temporary table name to the context
            yield temp_table_name

        finally:
            # Drop the temporary table when the context ends
            self.execute(f"DROP TEMPORARY TABLE IF EXISTS {temp_table_name};")

    def append_to_table(
        self,
        table: str,
        df: pd.DataFrame,
        return_reindexed: bool = False,
        no_tracking: bool = False,
    ) -> pd.DataFrame | None:
        if self.do_audit and not no_tracking:
            started_at = self.now()

        pk = self.get_primary_key(table=table, default="id")
        # For vanity's sake, give the dataframe index the correct name
        # but really we never actually use this index since we let the
        # database autoincrement during the insert
        df.index.name = pk
        # We don't try to cast the primary key (which is always an int)
        col_dtypes = {
            k: v for k, v in self.get_column_datatypes(table=table).items() if k != pk
        }

        columns_to_serialize: list[str]
        if table in self.media_to_deserialize:
            columns_to_serialize = self.media_to_deserialize[table]
        else:
            columns_to_serialize = []

        df_serialized = serialize_table(df, col_dtypes, columns_to_serialize)
        payload_info = get_payload_info(df_serialized)
        payload_size_MB = payload_info["payload_size"] / (1024**2)

        # Append our values to the actual database table
        self.log(
            f"INSERT {payload_info['num_rows']}r x {payload_info['num_cols']}c "
            f"in {self.database}.{table} ({payload_size_MB:.2f} MB)"
        )
        self.write_dataframe(df_serialized, table, if_exists="append", index=False)

        # Return the dataframe with the actual index AUTOINCREMENTED with the correct numbers
        # This is actually GUARANTEED to work, by the way mariadb does consecutive inserts!
        # cool right?
        if return_reindexed:
            # From testing on 2023-11-24, THIS is correct:
            last_insert_id = self.last_insert_id
            # mypy: df.index is Index[Any]; assigning list is valid at runtime but not in types
            df.index = list(range(last_insert_id, last_insert_id + len(df)))  # type: ignore[assignment]
            # NOT this:
            # df.index = list(range(last_insert_id - len(df) + 1, last_insert_id))
            df.index.name = pk

        if self.do_audit and not no_tracking:
            self._save_instrumentation(
                method=get_currentframe_method(),
                table=table,
                query="INSERT",
                started_at=started_at,
                **payload_info,
            )

        return df if return_reindexed else None

    def query_table(self, table: str, query: str) -> pd.DataFrame:
        """Query a single table and deserialize if necessary"""
        columns_to_deserialize: list[str]
        if table in self.media_to_deserialize:
            columns_to_deserialize = self.media_to_deserialize[table]
        else:
            columns_to_deserialize = []

        started_at = self.now()

        cur_df = self.read_dataframe_from_table(
            table_name=table, query=query, columns_to_deserialize=columns_to_deserialize
        )

        if self.do_audit and (query is not None):
            self._save_instrumentation(
                method=get_currentframe_method(),
                table=table,
                query=query,
                started_at=started_at,
                **get_payload_info(cur_df),
            )

        return cur_df

    def refresh_table(self, table) -> None:
        """Load the ENTIRE table which might be highly inefficient
        but for small scales this should be fine.
        """
        cur_df = self.query_table(table, f"""SELECT * FROM {table};""")
        setattr(self, f"{table}_df", cur_df)

    def log_to_db(self, log_dict: dict) -> None:
        """Save log to database
        This module will get the dictionary of log to print put and add
        to the database 'server_log' table. ALL of the keys need past
        to the function with db_credentials. Dictionary format :
            log_message = {
                'log_module' : log_module,
                'log_state' : "STARTED/ERROR/INFO/WARNING/COMPLETED/NOTIFY,
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
        if all(k in log_template for k in log_dict):
            query = f"""
                INSERT INTO server_log (`module`, `state`, `log_event`, `message`, `_host`)
                VALUES ('{str(log_dict["log_module"])}','{str(log_dict["log_state"])}','{str(log_dict["log_event"])}','{str(log_dict["log_message"])}','{str(log_dict["log_hostname"])}');
            """
            query = query.replace("'None'", "null")
            self.execute(query)
            self.log(
                f"{log_dict['log_state']} | {log_dict['log_module']} : {log_dict['log_event']} {log_dict['log_message']} {log_dict['log_hostname']}"
            )
        else:
            if "log_module" in log_dict.keys():
                failed_module = str(log_dict["log_module"])
            else:
                failed_module = "unknown"
            self.log(
                f"ERROR | central_log warning : missing log parameters from - {failed_module}"
            )

    def _save_instrumentation(
        self,
        method: str,
        table: str | None,
        query: str,
        started_at: pd.Timestamp,
        payload_size: int,
        num_rows: int,
        num_cols: int,
    ) -> None:
        """Save debugging information about each query that was run"""
        if not self.do_audit:
            return

        assert str(self.ipath) != ""

        # Save in the session timezone
        started_at = started_at.tz_convert(self.session_timezone)
        completed_at = self.now()
        duration = (completed_at - started_at).total_seconds()

        if (duration is not None) and duration > 0:
            MBps = (payload_size / duration) / (1024 * 1024)
        else:
            MBps = None

        # Extract the table name from the query string if necessary
        if table is None or table == "":
            table = get_next_word_after_from(query)

        payload = {
            "table_name": table,
            "query": query[:QUERY_AUDIT_MAX_CHARS],
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_seconds": duration,
            "payload_size_bytes": payload_size,
            "num_rows": num_rows,
            "num_cols": num_cols,
            "method": method,
            "MBps": MBps,
            "session_uuid": self.session_uuid,
            "host": self.host,
            "database_name": self.database,
            "timezone": self.session_timezone,
        }
        df = pd.DataFrame([payload])
        # Remove newlines in query
        df["query"] = df["query"].str.strip().replace("\n", " ")

        # Use separate audit connection to avoid interfering with last_insert_id
        # Use the audit connection's append_to_table method
        assert self._audit_db is not None
        self._audit_db.append_to_table("tony", df, no_tracking=True)

        # If we aren't just tracking exclusively in the database,
        # then write to a file please
        if str(self.ipath) != "database":
            self.log("SAVING LOCK PATH")
            ipath_path = pathlib.Path(self.ipath)
            lock_path = ipath_path.with_suffix(ipath_path.suffix + ".lock")
            with filelock.FileLock(lock_path):
                try:
                    df.to_csv(
                        str(self.ipath),
                        mode="a",
                        header=(not ipath_path.exists()),
                        index=False,
                    )
                except PermissionError as e:
                    self.log(
                        f"WARNING: Instrumentation file is locked for writing: "
                        f"{self.ipath} {e}\n{payload}"
                    )


# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------


class TonyDBC(_TonyDBCOnlineOnly):
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

    def __init__(self, *args, **kwargs) -> None:
        mb: str = "MEDIA_BASE_PATH_PRODUCTION"
        pickle_base_path: str
        if mb in os.environ:
            pickle_base_path = os.environ[mb]
        else:
            # Default to the script path if no pickle path was provided
            pickle_base_path = sys.path[0]

        self.__offline_status: str = "online"
        self.__offline_pickle_path: str = os.path.join(
            pickle_base_path, "dbcon_pickle.PICKLE"
        )
        self.__update_queue: queue.Queue = queue.Queue()
        super().__init__(*args, **kwargs)

    @property
    @check_connection
    def is_online(self) -> bool:
        return self.__offline_status != "offline"

    @is_online.setter
    def is_online(self, value: bool) -> None:
        """If the user sets is_online to True, and it was not before, then flush updates."""
        assert isinstance(value, bool)
        if self.__offline_status == "offline" and value:
            self.__offline_status = "flushing"
            self.flush_updates()
            self.__offline_status = "online"
        elif self.__offline_status == "online" and not value:
            self.__offline_status = "offline"

    def flush_updates(self) -> None:
        """Make all the updates to the tables that we have been saving up"""
        # Pickle our updates in case of error
        try:
            if not self.__update_queue.empty():
                self.log("Pickling before flushing to be safe")
                self.pickle_updates()
                self.log(f"Flushing {self.__update_queue.qsize()} database updates")
                while not self.__update_queue.empty():
                    method, kwargs = self.__update_queue.get()
                    getattr(self, method)(**kwargs)
                self.log("Flushing database updates - DONE")
                self.log("Backing up temp pickle")
                shutil.move(
                    str(self.__offline_pickle_path),
                    str(self.__offline_pickle_path) + ".BAK",
                )

        except AttributeError as e:
            if self.interact_after_error:
                code.interact(banner=f"Bad TonyDBC {e}", local=locals())
            else:
                raise AttributeError(e)

    def __enter__(self) -> "TonyDBC":
        super().__enter__()

        if not os.path.isfile(self.__offline_pickle_path):
            return self

        self.is_online = False
        # Try loading unsaved stuff if it exists
        try:
            with open(self.__offline_pickle_path, "rb") as pickle_file:
                update_list = pickle.load(pickle_file)
        except EOFError:
            self.log(
                f"Deleting corrupt pickle file {self.__offline_pickle_path} that is 0 bytes"
            )
            os.remove(self.__offline_pickle_path)
        else:
            self.log(
                f"Finished loading DB updates pickle {self.__offline_pickle_path}."
            )
            # Now get rid of the pickle so we don't use it again
            shutil.move(
                str(self.__offline_pickle_path),
                str(self.__offline_pickle_path) + ".BAK",
            )

        for v in update_list:
            self.__update_queue.put(v)

        # Now flush the pickled updates
        self.is_online = True

        return self

    def pickle_updates(self) -> None:
        # Pickle our queue
        if not self.__update_queue.empty():
            self.log(
                f"Pickling {self.__update_queue.qsize()} updates "
                f"to {self.__offline_pickle_path}"
            )
            backup_queue: queue.Queue = queue.Queue()
            queue_list: list[Any] = []
            while not self.__update_queue.empty():
                v = self.__update_queue.get()
                queue_list.append(copy.deepcopy(v))
                backup_queue.put(copy.deepcopy(v))

            with open(self.__offline_pickle_path, "wb") as pickle_file:
                pickle.dump(queue_list, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

            self.__update_queue = backup_queue

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.pickle_updates()

        # Clear the queue since we have now archived it
        self.__update_queue = queue.Queue()

        assert self.__update_queue.empty(), (
            "update_queue is not empty even though we just cleared it"
        )

        # Now that we have cleared the queue, we can go to online mode.
        self.is_online = True

        super().__exit__(exc_type, exc_val, exc_tb)
        return None

    def start_temp_conn(self) -> None:
        if not self.is_online:
            raise AssertionError("start_temp_conn is not supported when not online")
        else:
            super(TonyDBC, self).start_temp_conn()

    def close_temp_conn(self) -> None:
        if not self.is_online:
            raise AssertionError("close_temp_conn is not supported when not online")
        else:
            super(TonyDBC, self).close_temp_conn()

    def cursor(self) -> mariadb.Cursor:
        if not self.is_online:
            raise AssertionError("cursor is not supported when not online")
        else:
            return super(TonyDBC, self).cursor()

    def begin_transaction(self) -> None:
        if not self.is_online:
            raise AssertionError("begin_transaction is not supported when not online")
        else:
            super(TonyDBC, self).begin_transaction()

    def commit(self) -> None:
        if not self.is_online:
            raise AssertionError("commit is not supported when not online")
        else:
            super(TonyDBC, self).commit()

    def get_data(
        self,
        query: str,
        before_retry_cmd=None,
        no_tracking=False,
        return_type_codes=False,
    ) -> Any:
        if not self.is_online:
            raise AssertionError("get_data can only be used while online")
        else:
            return super(TonyDBC, self).get_data(
                query=query,
                before_retry_cmd=before_retry_cmd,
                no_tracking=no_tracking,
                return_type_codes=return_type_codes,
            )

    def drop_database(self, database) -> None:
        if not self.is_online:
            raise AssertionError("Cannot drop database if not online")
        else:
            super(TonyDBC, self).drop_database(database)

    def query_table(self, table: str, query: str) -> pd.DataFrame:
        if not self.is_online:
            raise AssertionError("Cannot query table if not online")
        else:
            return super(TonyDBC, self).query_table(table, query)

    def refresh_table(self, table) -> None:
        if not self.is_online:
            raise AssertionError("Cannot refresh table if not online")
        else:
            super(TonyDBC, self).refresh_table(table)

    def update_table(self, table_name, df) -> None:
        """Instead of actually updating the table, just enqueue the updates for doing later."""
        kwargs = {"table_name": table_name, "df": df}
        if self.is_online:
            super(TonyDBC, self).update_table(**kwargs)
        else:
            self.__update_queue.put(("update_table", kwargs))

    def write_dataframe(self, df, table_name, if_exists="replace", index=False) -> None:
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

    def update_blob(
        self, table_name, blob_column, id_value, filepath, max_size_MB=16
    ) -> None:
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

    def post_data(self, query: str) -> Any:
        kwargs = {"query": query}
        if self.is_online:
            return super(TonyDBC, self).post_data(**kwargs)
        else:
            self.__update_queue.put(("post_data", kwargs))

    def post_datalist(self, query: str, insert_data: list) -> Any:
        kwargs = {
            "query": query,
            "insert_data": insert_data,
        }
        if self.is_online:
            return super(TonyDBC, self).post_datalist(
                query=query, insert_data=insert_data
            )
        else:
            self.__update_queue.put(("post_datalist", kwargs))

    def execute(
        self,
        command: str,
        command_values=None,
        before_retry_cmd=None,
        no_tracking=False,
        log_progress=False,
    ) -> None:
        kwargs = {
            "command": command,
            "command_values": command_values,
            "before_retry_cmd": before_retry_cmd,
            "no_tracking": no_tracking,
            "log_progress": log_progress,
        }
        if self.is_online:
            super(TonyDBC, self).execute(**kwargs)
        else:
            self.__update_queue.put(("execute", kwargs))

    def execute_script(
        self, script_path: str, get_return_values=False, cur_database=None
    ) -> Any:
        kwargs = {
            "script_path": script_path,
            "get_return_values": get_return_values,
            "cur_database": cur_database,
        }
        if self.is_online:
            return super(TonyDBC, self).execute_script(**kwargs)
        else:
            self.__update_queue.put(("execute_script", kwargs))

    def insert_row_all_string(self, table, row_dict) -> None:
        kwargs = {
            "table": table,
            "row_dict": row_dict,
        }

        if self.is_online:
            super(TonyDBC, self).insert_row_all_string(**kwargs)
        else:
            self.__update_queue.put(("insert_row_all_string", kwargs))

    def append_to_table(
        self, table, df, return_reindexed=False, no_tracking=False
    ) -> pd.DataFrame | None:
        kwargs = {
            "table": table,
            "df": df,
            "return_reindexed": return_reindexed,
            "no_tracking": no_tracking,
        }
        if self.is_online:
            return super(TonyDBC, self).append_to_table(**kwargs)
        else:
            if return_reindexed:
                raise AssertionError(
                    "TonyDBC.append_to_table: cannot return reindexed while in `offline` mode"
                )
            self.__update_queue.put(("append_to_table", kwargs))
            return None

    def log_to_db(self, log_dict: dict) -> None:
        kwargs = {"log_dict": log_dict}
        if self.is_online:
            super(TonyDBC, self).log_to_db(**kwargs)
        else:
            self.__update_queue.put(("log_to_db", kwargs))
