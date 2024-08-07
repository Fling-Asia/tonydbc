"""
Drop-in replacement for pandas.DataFrame methods for reading and writing to database:

    pandas.DataFrame.to_sql
    pandas.read_sql_table

For some reason the current (Jan 2023) implementation in Pandas
using sqlalchemy is really slow.  These methods are ~300x faster.

NOTE: Works only with mariadb's connector for now (pip3 install mariadb)
    https://mariadb.com/resources/blog/how-to-connect-python-programs-to-mariadb/

"""

import os
import code
import tempfile
import csv
import numpy as np
import pandas as pd
import warnings
import mariadb
from .env_utils import get_env_bool

# Map SQL types to Python datatypes
DATATYPE_MAP = {
    "double": np.float64,
    "float": float,
    "int": int,
    "longlong": int,
    "smallint": int,
    "datetime": str,
    "var_string": str,
    "varchar": str,
    "char": str,
    "blob": object,  # Not sure about this one..
    "mediumblob": object,  # Not sure about this one..
    "longblob": object,  # Not sure about this one..
    "bigint": np.int64,
    "shorttext": str,
    "mediumtext": str,
    "longtext": str,
    "enum": str,
    "tinyint": int,
    "text": str,
    "bool": bool,
    "decimal": float,
    "timestamp": str,
    "bit": bool,
    "mediumint": int,
    "smallint": int,
    "integer": int,
    "int8": np.int64,
    "int4": int,
    "int3": int,
    "int2": int,
    "int1": int,
    "middleint": int,
    "serial": int,
}

# TODO: HANDLE ALL THESE:
# 'BIT', 'BLOB', 'DATE', 'DATETIME', 'DATETIME2', 'DECIMAL', 'DOUBLE', 'ENUM',
# 'FLOAT', 'GEOMETRY', 'INT24', 'JSON', 'LONG', 'LONGLONG', 'LONG_BLOB',
# 'MEDIUM_BLOB', 'NEWDATE', 'NEWDECIMAL', 'NULL', 'SET', 'SHORT', 'STRING',
# 'TIME', 'TIME2', 'TIMESTAMP', 'TIMESTAMP2', 'TINY', 'TINY_BLOB', 'VARCHAR',
# 'VAR_STRING', 'YEAR'

# Get a lookup of what all the data_types are
FIELD_TYPE = mariadb.constants.FIELD_TYPE
FIELD_TYPE_DICT = {
    getattr(FIELD_TYPE, k): k for k in dir(FIELD_TYPE) if not k.startswith("_")
}

# To handle as many strings as possible without any issues, let's use some very obscure characters to delimit the CSV
FIELD_DELIMITER = "\x1F"  # Unit Separator (␟)
ENCLOSURE_CHAR = "\x1E"  # Record Separator (␞)
LINE_TERMINATOR = "\n"  # Keep newline as is since the record separator is enough.


class DataFrameFast(pd.DataFrame):
    def to_sql(self, name, con, if_exists="append", index=False, *args, **kwargs):
        if not if_exists in ["replace", "append"]:
            raise AssertionError(
                "not if_exists in ['replace', 'append'] is not yet impemented"
            )

        # Truncate database table
        # NOTE: Users may have to perform to_sql in the correct
        # sequence to avoid causing foreign key errors with this step
        if if_exists == "replace":
            raise AssertionError(
                "'replace' is not implemented correctly... it will erase the whole table!"
            )
            with con.cursor() as cursor:
                r = cursor.execute(f"TRUNCATE TABLE {name}")

        # Nothing to do if the dataframe is empty
        if len(self) == 0:
            return

        assert self.columns.is_unique, f"Dataframe columns are not unique:\n{df0}"

        # Drop columns which contain only NULL values since mariadb's stupid executemany might chokes on these
        # if they are in the final column
        # (note that this will screw things up if the column has a DEFAULT of something other than NULL)
        cols_with_all_NULLs = self.columns[pd.isna(self).all()]

        df0 = self.drop(columns=cols_with_all_NULLs).copy()

        # Find any datetime columns.  e.g. dt_cols = {'file_created_at': datetime64[ns, Asia/Bangkok]}
        dt_cols = {
            col: dtype
            for col, dtype in df0.dtypes.to_dict().items()
            if isinstance(dtype, pd.DatetimeTZDtype)
        }

        # Convert all the pandas timestamp columns to string since suddenly we are getting
        # mariadb.NotSupportedError when trying to append
        if len(dt_cols) > 0:
            with con.cursor() as cursor:
                cursor.execute("SELECT @@session.time_zone;")
                time_zones = cursor.fetchone()
                assert len(time_zones) == 1
                time_zone = time_zones[0]
                # e.g. "+07:00"

            for dt_col in dt_cols.keys():
                # I think this is okay if we were careful to have the column's data
                # in the current session's datetime format
                new_col = (
                    df0.loc[:, dt_col]
                    # Convert to the correct time zone if necessary
                    .dt.tz_convert(time_zone)
                    # Convert to a string for mariadb to digest
                    .dt.strftime("%Y-%m-%d %H:%M:%S.%f")
                )

                # Force the dtype conversion, which otherwise doesn't happen for some reason
                df0.loc[:, dt_col] = ""
                df0.loc[:, dt_col] = "override"
                df0.loc[:, dt_col] = df0.loc[:, dt_col].astype(str)
                # Now assign the string date information
                df0.loc[:, dt_col] = new_col

        # Convert np.int64 columns to np.int32 since MySQL cannot accept np.int64 as a param
        small_int64cols = {
            col: dtype
            for col, dtype in df0.dtypes.to_dict().items()
            if isinstance(dtype, np.dtypes.Int64DType)
            # They must fit within the int32 boundaries
            and df0[col].max() < np.iinfo(np.int32).max
            and df0[col].min() > np.iinfo(np.int32).min
        }
        df0 = df0.astype({col: np.int32 for col in small_int64cols})

        # Prepare an INSERT which will populate the real mariadb table with df's data
        # INSERT INTO table(c1,c2,...) VALUES (v11,v12,...), ... (vnn,vn2,...);
        # If index, then we also want the index inserted
        cols = [df0.index.name] * index + list(df0.columns)
        # Sanitize the column names
        cols = [f"`{c}`" for c in cols]
        cmd = (
            f"INSERT INTO `{name}` ({', '.join(cols)})"
            f" VALUES ({', '.join(['?']*len(cols))})"
        )

        # Double-check that our dataframe does not contain our special csv control characters
        def has_ctrl_chars(s):
            if isinstance(s, str):
                return ENCLOSURE_CHAR in s or FIELD_DELIMITER in s
            return False

        df_has_ctrl_chars = df0.map(has_ctrl_chars).values.any()

        if len(df0) == 0:
            # Nothing to INSERT
            return
        elif len(df0) > 3 and not df_has_ctrl_chars:
            temp_dir = tempfile.TemporaryDirectory()
            tmp_filepath = os.path.join(temp_dir.name, name + ".csv")

            # Identify boolean columns and convert them to int (1 for True, 0 for False)
            # to avoid the warning: ('Warning', 1366, "Incorrect integer value:
            # 'False' for column michael_db.cats.is_success at row 1")
            bool_cols = df0.select_dtypes(include=["bool"]).columns
            df0[bool_cols] = df0[bool_cols].astype(int)

            # Convert to csv for uploading to the server as a file, which is faster
            # for some reason
            # the fillna is to handle NULLs
            df0.fillna("\\N").to_csv(
                tmp_filepath,
                index=False,
                header=True,
                quotechar=ENCLOSURE_CHAR,
                quoting=csv.QUOTE_ALL,
                sep=FIELD_DELIMITER,
                lineterminator=LINE_TERMINATOR,
            )

            # Use the faster INFILE method for larger sets of data
            # Set the path to save the CSV file
            tmp_filepath = tmp_filepath.replace("\\", "\\\\")

            # Write DataFrame to a CSV file with specified delimiters and text qualifiers
            cmd = f"""
                LOAD DATA LOCAL INFILE %s 
                    INTO TABLE `{name}`
                    FIELDS TERMINATED BY %s 
                    ENCLOSED BY %s
                    LINES TERMINATED BY %s
                    IGNORE 1 ROWS
                    ({', '.join(cols)});
                """

            with con.cursor() as cursor:
                cursor.execute(
                    cmd,
                    (tmp_filepath, FIELD_DELIMITER, ENCLOSURE_CHAR, LINE_TERMINATOR),
                )
                cursor.execute("SHOW WARNINGS;")
                # Check for warnings
                warnings = cursor.fetchall()

            if warnings:
                print(f"TonyDBC ran the command:\n{cmd}\nWARNINGS:\n")
                for warning in warnings:
                    print(warning)  # Each warning is a tuple (Level, Code, Message)

            temp_dir.cleanup()

        else:
            # For a small amount of data, I guess we can use the old way
            table_data = list(df0.itertuples(index=index))

            # https://mariadb-corporation.github.io/mariadb-connector-python/usage.html
            # "When using executemany(), there are a few restrictions:"

            # "1. Special values like None or column default value needs to
            #     be indicated by an indicator."
            MARIADB_NULL = mariadb.constants.INDICATOR.NULL
            try:
                table_data = [
                    tuple(
                        [MARIADB_NULL if pd.isna(value) else value for value in sublist]
                    )
                    for sublist in table_data
                ]
            except ValueError as e:
                if get_env_bool("INTERACT_AFTER_ERROR"):
                    code.interact(local=locals(), banner="tupling no workie")
                else:
                    raise ValueError(e)

            # Optimization: just use `execute` if it's a single line of data
            if len(table_data) == 1:
                with con.cursor() as cursor:
                    cursor.execute(cmd, table_data[0])
                return

            # 2. A workaround to https://jira.mariadb.org/browse/CONPY-254
            table_data_bad = [
                v for i, v in enumerate(table_data) if v[-1] == MARIADB_NULL
            ]
            table_data_good = [
                v for i, v in enumerate(table_data) if v[-1] != MARIADB_NULL
            ]

            with con.cursor() as cursor:
                # Try to use the bulk (faster) option for the "good" rows
                if len(table_data_good) > 0:
                    try:
                        cursor.executemany(cmd, table_data_good)
                    except mariadb.ProgrammingError as e:
                        print(
                            f"Warning: problem with executemany; probably because you are using a "
                            f"np.dtypes.Int64DType data type.  We'll just insert these one at a time. {e}"
                        )
                        for v in table_data_good:
                            cursor.execute(cmd, v)
                    except SystemError as e:
                        # 3. "All tuples must have the same types as in first tuple.
                        #    E.g. the parameter [(1),(1.0)] or [(1),(None)] are invalid."
                        print(
                            "Warning: due to a known bug in a mariadb connector, e.g. None in final field or something, "
                            f"`executemany` raises {e}.  Failing over to row-by-row `execute`"
                        )
                        for v in table_data_good:
                            cursor.execute(cmd, v)

                if len(table_data_bad) > 0:
                    for v in table_data_bad:
                        cursor.execute(cmd, v)

    def column_info(self):
        """Returns the column information.
        Parameters:
            table_name: string.  If None, returns column info for ALL tables.
        """
        clauses = [f"TABLE_SCHEMA = '{self.database}'"]
        if not table_name is None:
            clauses.append(f"TABLE_NAME = '{table_name}'")

        with con.cursor() as cursor:
            cursor.execute(
                f"SELECT * FROM INFORMATION_SCHEMA.COLUMNS "
                f"WHERE {'AND '.join(clauses)};"
            )
            records0 = cursor.fetchall()

        return pd.DataFrame(r)


def get_field_names(con, query):
    """Use a trick to get the column names,
    even if the result set is 0 rows.  ChatGPT told me!  Wow.
    """
    with con.cursor() as cursor:
        cursor.execute(query)
        records0 = cursor.fetchall()
        # Get the field names
        cursor.execute(query + " LIMIT 0")
        columns = [v[0] for v in cursor.description]

    return columns


def get_data(con, query, return_type_codes=False):
    with con.cursor() as cursor:
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

    records = [
        {fields[i][0]: field_value for i, field_value in enumerate(v)} for v in records0
    ]

    if return_type_codes:
        # Get a list of dicts with proper field names
        # (i.e. records in the pandas sense)
        # (e.g. {'media_object_id': 'LONGLONG', 'full_path': 'VAR_STRING'}
        type_codes = {v[0]: FIELD_TYPE_DICT[v[1]] for v in fields}

        return records, type_codes
    else:
        return records


def read_sql_table(name, con, query=None, *args, **kwargs):
    """A drop-in replacement for pd.read_sql_table
    Note: this does not set an index.
    """
    # Just use a basic query if none was specified
    if query is None:
        query = f"SELECT * FROM {name};"
    records, type_codes = get_data(con, query, return_type_codes=True)

    # TODO: convert all cols
    pass

    # KLUDGE: for now, just convert any BIT fields to bool
    bit_cols = {k: bool for k, v in type_codes.items() if v.upper() == "BIT"}

    if len(records) > 0:
        df = DataFrameFast.from_records(records)

        for k in bit_cols.keys():
            # Convert b'\x01' to True and b'\x00' to False
            df[k] = df.apply(
                lambda v: bool(int.from_bytes(v[k], byteorder="big")), axis=1
            )
    else:
        # We also have to return the columns names in case records is []
        columns = get_field_names(con, query)
        df = DataFrameFast(columns=columns)
        df = df.astype(bit_cols)

    return df
