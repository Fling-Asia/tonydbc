"""
Support utilities for TonyDBC:
    json_dumps_numpy
    serialize_table
    deserialize_table
    get_current_time
    set_MYSQL_DATABASE
    iso_timestamp_to_utc
    get_tz_offset
    validate_tz_offset

"""

import code
import datetime
import json
import os
import re
import sys
import zoneinfo
from typing import Any, Union

import dateutil
import numpy as np
import pandas as pd
import pytz
import tzlocal

from .env_utils import get_env_bool


# A custom encoder to handle np.int64 and other non-serializable types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)  # Convert any numpy integer to Python int
        if isinstance(obj, np.floating):
            return float(obj)  # Convert any numpy float to Python float
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy arrays to Python lists
        if isinstance(obj, list):
            # Recursively convert numpy integers within lists
            return [
                int(item) if np.issubdtype(type(item), np.integer) else item
                for item in obj
            ]
        # Let the base class handle other types
        return super(NumpyEncoder, self).default(obj)


def serialize_table(
    cur_df: pd.DataFrame, col_dtypes: dict[str, Any], columns_to_serialize: list[str]
) -> pd.DataFrame:
    # If we don't make a copy, we will create side effects in the original cur_df
    cur_df = cur_df.copy()

    # Nothing to convert if it's empty
    if len(cur_df) == 0:
        return cur_df

    # Exclude columns which are not present in the table
    columns_to_serialize0 = list(
        set(cur_df.columns).intersection(set(columns_to_serialize))
    )

    # Serialize the relevant columns from strings into nested arrays of dicts and lists
    for c in columns_to_serialize0:
        # No need to serialize into STRING 'None' anymore; mariadb connector can handle None
        try:
            # Use Series.apply on the single column to satisfy mypy's overloads
            col_series = cur_df.loc[:, c]
            cur_df.loc[:, c] = col_series.apply(
                lambda x: None
                if np.any(pd.isna(x))
                else json.dumps(x, cls=NumpyEncoder)
            )
        except TypeError as e:
            raise TypeError(f"Column {c} could not be serialized; {e}")

    # Serialization approach #1 (older): cast int and also float.
    # We don't do ndarray anymore.  (we just use lists of lists and we do that with approach 2 below)
    # Now cast the non-ndarray columns
    int_cols = [
        col
        for col, dt in dict(col_dtypes).items()
        if (dt is int or dt is np.int64) and col in cur_df.columns
    ]
    # Columns with NaN or None should use pandas nullable integer types
    # Convert to proper nullable int types that can handle pd.NA
    for col in int_cols:
        if col in cur_df.columns:
            # Convert to nullable integer type (Int64 for int64, Int32 for int32, etc.)
            # these types properly handle None/NaN as pd.NA
            target_dtype = dict(col_dtypes)[col]
            if target_dtype is int or target_dtype is np.int64:
                cur_df[col] = cur_df[col].astype("Int64")
            elif target_dtype is np.int32:
                cur_df[col] = cur_df[col].astype("Int32")
            else:
                cur_df[col] = cur_df[col].astype("Int64")

    # If we explicitly cast string columns to str instead of leaving them as object
    # then None values will be replaced with 'None' and won't insert properly into the database
    # so we don't bother to cast to str (object will work fine anyway)
    try:
        cur_df = cur_df.astype(
            {
                col: dt
                for col, dt in dict(col_dtypes).items()
                if dt != np.ndarray and dt is not str and col in cur_df.columns
            }
        )
    except ValueError:
        # We will end up here if one of the values in an int column is a np.nan since np.nan cnanot be cast to int
        # so let's ignore this error (??)
        # TODO: fix KLUDGE (maybe iterate through each separately so at least we don't break early in the
        # conversion process)
        pass

    return cur_df


def deserialize_table(
    cur_df: pd.DataFrame, columns_to_deserialize: list[str], session_timezone: str
) -> pd.DataFrame:
    if len(cur_df) == 0:
        return cur_df

    # Exclude columns which are not present in the table
    columns_to_deserialize0 = list(
        set(cur_df.columns).intersection(set(columns_to_deserialize))
    )
    # Deserialize the relevant columns from strings into nested arrays of dicts and lists
    for c in columns_to_deserialize0:
        try:
            cur_df.loc[:, [c]] = cur_df.loc[:, [c]].fillna("nan")
        except KeyError as e:
            if get_env_bool("INTERACT_AFTER_ERROR"):
                print(f"KEY ERROR {e}")
                code.interact(local=locals(), banner=f"{e}")
            else:
                raise KeyError(e)

        def json_loads(v):
            try:
                return json.loads(v[c]) if v[c] not in ["nan", "None"] else np.nan
            except json.decoder.JSONDecodeError as e:
                print(
                    f"tonydbc.deserialize_table JSON DECODE ERROR: column {c} row {v.name}: {v[c]}: {e}"
                )
                return np.nan
            except TypeError as e:
                print(
                    f"tonydbc.deserialize_table ERROR: column {c} row {v.name}: {v[c]}: {e}"
                )
                return np.nan

        cur_df.loc[:, [c]] = cur_df.loc[:, [c]].apply(json_loads, axis=1)

    # CONVERT all pd.Timestamp objects (which have been provided by mariadb
    # in the session timezone anyway)
    # into a fully localized timestamp
    for c in cur_df.columns:
        if cur_df[c].dtype.type == np.datetime64:
            cur_df[c] = cur_df.apply(
                lambda v: v[c].tz_localize(session_timezone), axis=1
            )
        elif isinstance(cur_df[c].dtype.type(), str) and (
            c.endswith("_at") or c == "timestamp"
        ):
            # Try our best to do it to any other field such as DATETIME entries which
            # were not already set to pd.Timestamp objects
            # (note that this assumes the DATETIME entries are in our session timezone)
            cur_df[c] = cur_df.apply(
                lambda v: pd.Timestamp(v[c]).tz_localize(session_timezone), axis=1
            )

    return cur_df


def get_current_time(
    use_utc: bool = False, default_timezone: str | None = None
) -> datetime.datetime:
    if default_timezone is None:
        if "DEFAULT_TIMEZONE" in os.environ:
            default_timezone = os.environ["DEFAULT_TIMEZONE"]
        else:
            default_timezone = "Asia/Bangkok"
    if use_utc:
        return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
    else:
        return datetime.datetime.now(pytz.timezone(default_timezone))


def get_current_time_string(
    use_utc: bool = False, default_timezone: str | None = None
) -> str:
    return get_current_time(
        use_utc=use_utc, default_timezone=default_timezone
    ).strftime("%Y-%m-%d %H:%M:%S")


def set_MYSQL_DATABASE(
    use_production_database: bool = get_env_bool("USE_PRODUCTION_DATABASE"),
    mysql_production_database: str = os.environ["MYSQL_PRODUCTION_DATABASE"],
    mysql_test_database: str = os.environ["MYSQL_TEST_DATABASE"],
) -> None:
    """Set the MYSQL_DATABASE environment variable"""
    # If it's already been set, don't take any action
    # (this applies to our docker containers, which have it pre-set)
    # if "MYSQL_DATABASE" in os.environ:
    #    return

    # Otherwise, set it according to whether we should be in production or not
    if use_production_database:
        os.environ["MYSQL_DATABASE"] = mysql_production_database
    else:
        os.environ["MYSQL_DATABASE"] = mysql_test_database


def iso_timestamp_to_utc(
    iso_timestamp_string: str,
) -> str:
    """e.g. converts an ISO-formatted string to UTC which is useful
    when adding to a mariadb TIMESTAMP which has no timezone awareness
    e.g. "2023-03-03T09:00:00+07:00" ->
         "2023-03-03 02:00:00+00:00"
    """
    try:
        ts = dateutil.parser.parse(iso_timestamp_string)
    except dateutil.parser._parser.ParserError as e:
        raise ValueError(f"Timestamp '{iso_timestamp_string}' is invalid. {e}")
    else:
        iso_ts = ts.astimezone(dateutil.tz.UTC)
        return iso_ts.strftime("%Y-%m-%d %H:%M:%S")


def get_tz_offset(iana_tz: str | None = None) -> str:
    """Converts IANA Time Zone to offset string
    e.g.  'Asia/Bangkok' -> '+07:00'

    Use the system time zone if no time zone provided
    """
    if iana_tz is None:
        # Use the system time zone if no time zone provided
        iana_tz = str(tzlocal.get_localzone())
    assert iana_tz in zoneinfo.available_timezones()
    pytz_tz = pytz.timezone(iana_tz)
    offset_str = datetime.datetime.now().astimezone(pytz_tz).strftime("%z")
    offset_str = offset_str[:3] + ":" + offset_str[3:]

    return offset_str


def validate_tz_offset(iana_tz: str, tz_offset: str) -> None:
    """e.g. Validates that '07:00' is the correct offset from UTC for 'Asia/Bangkok'"""
    assert get_tz_offset(iana_tz) == tz_offset


def get_next_word_after_from(input_string: str) -> str | None:
    """Help to find the table name after FROM"""

    # Convert the input string to uppercase for case-insensitive search
    input_upper = input_string.upper()

    # Use regex to search for the word after "FROM", stripping backticks (`))
    match = re.search(r"\bFROM\b\s+`?([\w.]+)`?", input_upper)

    if match:
        # Use the match's span to get the range and
        # extract the word from the original input string, preserving the original case.
        start, end = match.span(1)
        return input_string[start:end]
    else:
        return None


def get_payload_info(payload: Any) -> dict[str, int]:
    """Get the size of a payload, which can be a dataframe or nested iterables of various kinds"""
    payload_size = sys.getsizeof(payload)
    if payload is None:
        num_rows = 0
        num_cols = 0
    elif isinstance(payload, pd.DataFrame):
        payload_size = payload.memory_usage(deep=True).sum()
        num_rows, num_cols = payload.shape
    elif hasattr(payload, "__len__"):
        num_rows = len(payload)
        if num_rows == 0:
            num_cols = 0
        elif hasattr(payload[0], "__len__"):
            num_cols = len(payload[0])
        else:
            num_cols = 0
    else:
        num_rows = 0
        num_cols = 0
    return {
        "payload_size": payload_size,
        "num_rows": num_rows,
        "num_cols": num_cols,
    }


def prepare_scripts(test_db: str, schema_filepaths: list[str]) -> str:
    """Return a string consisting of all the contents of the scripts provided

    Parameters:
        test_db: a string
        schema_filepaths: a list of strings which are paths to scripts

    """
    assert not isinstance(schema_filepaths, str), (
        f"schema_filepaths ({schema_filepaths}) must be a list"
    )
    program_to_run0 = ""

    # Run all the scripts requested
    for schema_filepath in schema_filepaths:
        program_to_run0 += f"\n\n\nUSE {test_db};\n"
        with open(schema_filepath, "r") as f:
            program_to_run0 += f.read()
        program_to_run0 += "\n\n\n"

    return program_to_run0


# This might be quite a large string but hopefully SQL doesn't choke
# Convert a list [1,2,3] into string "(1,2,3)"
def list_to_SQL(v: list[Union[str, int, float]]) -> str:
    if all(isinstance(k, str) for k in v):
        return "(" + ",".join([f"'{k}'" for k in v]) + ")"
    elif all(not isinstance(k, str) for k in v):
        return "(" + ",".join([f"{k}" for k in v]) + ")"
    else:
        raise AssertionError(f"Elements in the list are of mixed type: {v}")


def list_to_SQL2(col_ids: list[int], column_name: str) -> str:
    """A more advanced version that also uses ranges"""
    col_ids = np.unique(np.sort(col_ids)).tolist()  # Sort and remove duplicates
    diffs = np.diff(col_ids)  # Get the difference between consecutive elements

    # Find indices where the difference is greater than 1 (i.e., a break in a range)
    breaks = np.where(diffs > 1)[0]

    # Start each range at the beginning or after a break
    starts = np.concatenate(([0], breaks + 1))
    # End each range at the break or at the end of the array
    ends = np.concatenate((breaks, [len(col_ids) - 1]))

    singletons = []
    queries = []
    for start, end in zip(starts, ends):
        if col_ids[start] == col_ids[end]:
            singletons.append(col_ids[start])
        else:
            queries.append(
                f"({column_name} BETWEEN {col_ids[start]} AND {col_ids[end]})"
            )

    if len(singletons) > 0:
        queries.append(f"{column_name} in {list_to_SQL(singletons)}")

    return " OR ".join(queries)
