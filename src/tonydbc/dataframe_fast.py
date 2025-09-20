"""
Drop-in replacement for pandas.DataFrame methods for reading and writing to database:

    pandas.DataFrame.to_sql_fast
    pandas.read_sql_table

For some reason the current (Jan 2023) implementation in Pandas
using sqlalchemy is really slow.  These methods are ~300x faster.

NOTE: Works only with mariadb's connector for now (pip3 install mariadb)
    https://mariadb.com/resources/blog/how-to-connect-python-programs-to-mariadb/

"""

import code
import csv
import os
import tempfile
from typing import Any, cast

import mariadb  # type: ignore
import numpy as np
import pandas as pd

from .env_utils import get_env_bool

# Map SQL types to pandas nullable datatypes
DATATYPE_MAP = {
    "double": "Float64",
    "float": "Float64",
    "int": "Int64",
    "longlong": "Int64",
    "long": "Int64",
    "smallint": "Int64",
    "datetime": "string",
    "var_string": "string",
    "string": "string",
    "varchar": "string",
    "char": "string",
    "blob": "object",  # Keep as object for binary data
    "mediumblob": "object",  # Keep as object for binary data
    "longblob": "object",  # Keep as object for binary data
    "bigint": "Int64",
    "shorttext": "string",
    "mediumtext": "string",
    "longtext": "string",
    "enum": "string",
    "tinyint": "Int64",
    "tinyint(1)": "boolean",  # TODO: actually, this could be anything from -127 to 128 but we'll just assume boolean for now
    "tiny": "Int64",
    "text": "string",
    "bool": "boolean",
    "decimal": "Float64",
    "newdecimal": "Float64",
    "timestamp": "string",
    "bit": "boolean",
    "mediumint": "Int64",
    "integer": "Int64",
    "int8": "Int64",
    "int4": "Int64",
    "int3": "Int64",
    "int2": "Int64",
    "int1": "Int64",
    "middleint": "Int64",
    "serial": "Int64",
}

# TODO: HANDLE ALL THESE:
# 'BIT', 'BLOB', 'DATE', 'DATETIME', 'DATETIME2', 'DECIMAL', 'DOUBLE', 'ENUM',
# 'FLOAT', 'GEOMETRY', 'INT24', 'JSON', 'LONG', 'LONGLONG', 'LONG_BLOB',
# 'MEDIUM_BLOB', 'NEWDATE', 'NEWDECIMAL', 'NULL', 'SET', 'SHORT', 'STRING',
# 'TIME', 'TIME2', 'TIMESTAMP', 'TIMESTAMP2', 'TINY', 'TINY_BLOB', 'VARCHAR',
# 'VAR_STRING', 'YEAR'


def refine_dtype(data_type: str, column_type: str):
    """
    You can get this information from e.g.

        SELECT DATA_TYPE, COLUMN_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = 'your_database'
        AND TABLE_NAME = 'your_table';

    """
    data_type = data_type.lower()
    column_type = column_type.lower()

    # 1) boolean synonyms / tinyint(1)
    if data_type in ("bool", "boolean"):
        return "boolean"
    if data_type == "tinyint":
        # Treat tinyint(1) as boolean ONLY if truly boolean
        if column_type == "tinyint(1)":
            # optional: whitelist by name convention or check constraints
            return "boolean"
        # otherwise keep as small int
        return "Int8"

    # 2) BIT
    if data_type == "bit":
        if column_type == "bit(1)":
            return "boolean"
        # For bit(n>1): either keep bytes or convert yourself later
        return "object"

    # 3) unsigned integers
    if data_type in ("tinyint", "smallint", "mediumint", "int", "integer", "bigint"):
        unsigned = "unsigned" in column_type
        if unsigned:
            return {
                "tinyint": "UInt8",
                "smallint": "UInt16",
                "mediumint": "UInt32",
                "int": "UInt32",
                "integer": "UInt32",
                "bigint": "UInt64",
            }[data_type]
        else:
            return {
                "tinyint": "Int8",
                "smallint": "Int16",
                "mediumint": "Int32",
                "int": "Int32",
                "integer": "Int32",
                "bigint": "Int64",
            }[data_type]

    # 4) decimals: consider precision
    if data_type in ("decimal", "newdecimal", "numeric"):
        # If you care about exactness, pick 'object' (Decimal) instead of Float64
        return "Float64"

    # otherwise use base
    return "object"


# Get a lookup of what all the data_types are
FIELD_TYPE = mariadb.constants.FIELD_TYPE
FIELD_TYPE_DICT = {
    getattr(FIELD_TYPE, k): k for k in dir(FIELD_TYPE) if not k.startswith("_")
}

# To handle as many strings as possible without any issues, let's use some very obscure characters to delimit the CSV
FIELD_DELIMITER = "\x1f"  # Unit Separator (␟)
ENCLOSURE_CHAR = "\x1e"  # Record Separator (␞)
LINE_TERMINATOR = "\n"  # Keep newline as is since the record separator is enough.


class DataFrameFast(pd.DataFrame):
    # Use a distinct name to avoid overriding pandas' final to_sql
    def to_sql_fast(
        self,
        name: str,
        con: mariadb.Connection,
        session_timezone: str,
        if_exists: str = "append",
        index: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if if_exists not in ["replace", "append"]:
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

        assert self.columns.is_unique, (
            f"Dataframe columns are not unique:\n{self.columns}"
        )

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
            for dt_col in dt_cols.keys():
                # I think this is okay if we were careful to have the column's data
                # in the current session's datetime format
                # Convert timezone-aware timestamps to MariaDB-compatible strings
                col_data = df0.loc[:, dt_col]

                # Convert to the target timezone if necessary
                if col_data.dt.tz is not None:
                    # Convert to target timezone, then remove timezone info for MariaDB
                    col_data = col_data.dt.tz_convert(session_timezone).dt.tz_localize(
                        None
                    )

                # Convert to string format that MariaDB accepts
                new_col = col_data.dt.strftime("%Y-%m-%d %H:%M:%S.%f").astype("string")

                # Directly assign the converted datetime strings without intermediate assignments
                # This avoids the pandas FutureWarning about incompatible dtype assignment
                # And by creating a fresh Series here, this will work even if df0 is empty
                # Note: we need to create SeriesCtor to satisfy mypy's "Series[Any]" not collable [operator] error
                SeriesCtor = cast(type[pd.Series], pd.Series)
                df0[dt_col] = SeriesCtor(new_col, dtype="string", index=df0.index)

        # Convert boolean columns to integers for MariaDB TINYINT(1) compatibility
        bool_cols: dict[str, Any] = {
            col: dtype
            for col, dtype in df0.dtypes.to_dict().items()
            if str(dtype) == "boolean" or isinstance(dtype, bool)
        }
        for col in bool_cols:
            # Convert boolean to int (True->1, False->0, pd.NA->None)
            df0[col] = df0[col].astype("Int64")

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
        cols: list[str] = ([] if not index else [str(df0.index.name)]) + [
            str(c) for c in list(df0.columns)
        ]
        # Sanitize the column names
        cols = [f"`{c}`" for c in cols]
        assert all(isinstance(c, str) for c in cols)
        cmd = (
            f"INSERT INTO `{name}` ({', '.join([str(c) for c in cols])})"
            f" VALUES ({', '.join(['?'] * len(cols))})"
        )

        # Double-check that our dataframe does not contain our special csv control characters
        def has_ctrl_chars(s: Any) -> bool:
            if isinstance(s, str):
                return ENCLOSURE_CHAR in s or FIELD_DELIMITER in s
            return False

        df_has_ctrl_chars = df0.map(has_ctrl_chars).to_numpy().any()

        if len(df0) == 0:
            # Nothing to INSERT
            return
        elif len(df0) > 3 and not df_has_ctrl_chars:
            temp_dir = tempfile.TemporaryDirectory()
            tmp_filepath = os.path.join(temp_dir.name, name + ".csv")

            # Identify boolean columns and convert them to int (1 for True, 0 for False)
            # to avoid the warning: ('Warning', 1366, "Incorrect integer value:
            # 'False' for column michael_db.cats.is_success at row 1")
            bool_cols_to_exclude: list[str] = list(
                df0.select_dtypes(include=["bool", "boolean"]).columns
            )
            df0[bool_cols_to_exclude] = df0[bool_cols_to_exclude].astype(int)

            # Convert to csv for uploading to the server as a file, which is faster
            # for some reason
            # the fillna is to handle NULLs
            df0.to_csv(
                tmp_filepath,
                index=False,
                header=True,
                na_rep="\\N",  # ← this prints pd.NA/NaN as \N
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
                    ({", ".join([str(c) for c in cols])});
                """

            with con.cursor() as cursor:
                # Perform the insert
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

            def int_converter(v: Any) -> Any:
                if isinstance(v, np.int64):
                    return int(v)
                else:
                    return v

            try:
                table_data_typed: list[tuple[Any, ...]] = [
                    tuple(
                        [
                            MARIADB_NULL if pd.isna(value) else int_converter(value)
                            for value in sublist
                        ]
                    )
                    for sublist in table_data
                ]
                table_data = table_data_typed  # type: ignore[assignment]
            except ValueError as e:
                if get_env_bool("INTERACT_AFTER_ERROR"):
                    code.interact(
                        local=locals(), banner=f"DataFrameFast tupling error: {e}"
                    )
                else:
                    raise ValueError(e)

            # Optimization: just use `execute` if it's a single line of data
            if len(table_data) == 1:
                with con.cursor() as cursor:
                    try:
                        cursor.execute(cmd, table_data[0])
                    except Exception as e:
                        if get_env_bool("INTERACT_AFTER_ERROR"):
                            code.interact(
                                banner=f"DataFrameFast error: {e}", local=locals()
                            )
                        else:
                            raise Exception(e)
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

    def column_info(
        self, con: mariadb.Connection, table_name: str | None = None
    ) -> pd.DataFrame:
        """Returns the column information.
        Parameters:
            table_name: string.  If None, returns column info for ALL tables.
        """
        clauses: list[str] = [f"TABLE_SCHEMA = '{self.database}'"]
        if table_name is not None:
            clauses.append(f"TABLE_NAME = '{table_name}'")

        with con.cursor() as this_cursor:
            this_cursor.execute(
                f"SELECT * FROM INFORMATION_SCHEMA.COLUMNS "
                f"WHERE {'AND '.join(clauses)};"
            )
            records = this_cursor.fetchall()
            fields = this_cursor.description

        # Convert records to list of dicts
        records_dict = [
            {fields[i][0]: field_value for i, field_value in enumerate(v)}
            for v in records
        ]

        return pd.DataFrame(records_dict)


def read_sql_table(
    name: str,
    con: mariadb.Connection,
    query: str,
    *args: Any,
    **kwargs: Any,
) -> pd.DataFrame:
    """A drop-in replacement for pd.read_sql_table
    Note: this does not set an index.
    """
    assert query is not None

    # Use the main tonydbc to retrieve this with
    records = con.get_data(query=query, no_tracking=True)
    type_codes = con.get_type_codes(query=query)

    # Convert SQL types to pandas nullable datatypes
    dtype_conversions = {}
    for col, sql_type in type_codes.items():
        sql_type_lower: str = sql_type.lower()

        # Special handling for TINY type - need to check if it's boolean
        # In MariaDB, BOOL is an alias for TINYINT(1), so TINY fields might be boolean
        if sql_type_lower == "tiny":
            # For now, assume TINY is boolean if it's a single-byte field
            # This is a heuristic - ideally we'd check the actual column definition
            dtype_conversions[col] = "boolean"
        elif sql_type_lower in DATATYPE_MAP:
            dtype_conversions[col] = DATATYPE_MAP[sql_type_lower]
        else:
            # Raise error for unknown types
            raise ValueError(
                f"Unknown SQL type '{sql_type_lower}' for column '{col}'. Please add it to DATATYPE_MAP."
            )

    # Special handling for BIT fields (convert binary to boolean)
    bit_cols = {k: v for k, v in type_codes.items() if v.upper() == "BIT"}

    if len(records) > 0:
        df = DataFrameFast.from_records(records)

        # Convert BIT fields from binary to boolean
        for k in bit_cols.keys():
            # Convert b'\x01' to True and b'\x00' to False
            df[k] = df[k].apply(
                lambda cell: bool(int.from_bytes(cell, byteorder="big"))
            )

        # Apply nullable datatype conversions
        for col, target_dtype in dtype_conversions.items():
            if col in df.columns:
                df[col] = df[col].astype(target_dtype)
    elif "CALL" not in query.upper():
        # We also have to return the columns names in case records is []
        """Use a trick to get the column names,
        even if the result set is 0 rows.  ChatGPT told me!  Wow.
        """
        with con.cursor() as cursor:
            cursor.execute(query)
            # Get the field names - we don't need the records for this case
            q0 = query.strip().rstrip(";")
            if " LIMIT " in q0.upper():
                cursor.execute(q0)
            else:
                cursor.execute(q0 + " LIMIT 0;")
            columns = [v[0] for v in cursor.description]

        df = DataFrameFast(columns=columns)
        # Apply both bit column conversions and nullable datatype conversions for empty DataFrames
        all_conversions = {**bit_cols, **dtype_conversions}
        if all_conversions:
            df = df.astype(all_conversions)
    else:
        # We cannot get the columns if it was a stored procedure
        return DataFrameFast(columns=[])

    return df
