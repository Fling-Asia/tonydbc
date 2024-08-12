"""
Support utilities for TonyDBC:
    json_dumps_numpy
    serialize_table
    deserialize_table
    get_current_time
    get_current_time_string
    set_MYSQL_DATABASE
    iso_timestamp_to_utc
    get_tz_offset
    validate_tz_offset

"""

import os
import sys
import re
import code
import json
import pytz
import zoneinfo
import tzlocal
import typing
import numpy as np
import pandas as pd
import datetime
import dateutil
from .env_utils import get_env_bool


def json_dumps_numpy(x):
    # Handle numpy arrays as well as regular nested lists
    if type(x) is np.ndarray:
        x = x.tolist()

    # When appending to the database, json.dumps will complain if this isn't a proper int
    # so get rid of the np.int32 format here:
    if type(x) == list:
        x = [int(xx) if np.issubdtype(type(xx), np.integer) else xx for xx in x]

    return json.dumps(x)


def serialize_table(cur_df, col_dtypes, columns_to_serialize: typing.List[str]):
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
        cur_df.loc[:, c] = cur_df.loc[:, [c]].apply(
            lambda v: None if np.any(pd.isna(v[c])) else json_dumps_numpy(v[c]),
            axis=1,
        )

    # Serialization approach #1 (older): cast int and also float.
    # We don't do ndarray anymore.  (we just use lists of lists and we do that with approach 2 below)
    # Now cast the non-ndarray columns
    int_cols = [
        col
        for col, dt in dict(col_dtypes).items()
        if dt == int and col in cur_df.columns
    ]
    # Columns with NaN can't be cast to int so let's first change NaN to -1
    cur_df[int_cols] = cur_df[int_cols].fillna(-1)

    # If we explicitly cast string columns to str instead of leaving them as object
    # then None values will be replaced with 'None' and won't insert properly into the database
    # so we don't bother to cast to str (object will work fine anyway)
    try:
        cur_df = cur_df.astype(
            {
                col: dt
                for col, dt in dict(col_dtypes).items()
                if dt != np.ndarray and dt != str and col in cur_df.columns
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
    cur_df, columns_to_deserialize: typing.List[str], session_timezone
):
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

        try:
            cur_df.loc[:, [c]] = cur_df.loc[:, [c]].apply(
                lambda v: json.loads(v[c]) if v[c] not in ["nan", "None"] else np.nan,
                axis=1,
            )
        except TypeError as e:
            if get_env_bool("INTERACT_AFTER_ERROR"):
                print(f"tonydbc.deserialize_table ERROR {e}")
                code.interact(local=locals(), banner=f"{e}")
            else:
                raise TypeError(e)
        except json.decoder.JSONDecodeError as e:
            if get_env_bool("INTERACT_AFTER_ERROR"):
                print(f"tonydbc.deserialize_table ERROR {e}")
                code.interact(local=locals(), banner=f"{e}")
            else:
                raise json.decoder.JSONDecodeError(e)

    # CONVERT all pd.Timestamp objects (which have been provided by mariadb
    # in the session timezone anyway)
    # into a fully localized timestamp
    for c in cur_df.columns:
        if cur_df[c].dtype.type == np.datetime64:
            cur_df[c] = cur_df.apply(
                lambda v: v[c].tz_localize(session_timezone), axis=1
            )
        elif cur_df[c].dtype.type == str and c.endswith("_at") or c == "timestamp":
            # Try our best to do it to any other field such as DATETIME entries which
            # were not already set to pd.Timestamp objects
            # (note that this assumes the DATETIME entries are in our session timezone)
            cur_df[c] = cur_df.apply(
                lambda v: pd.Timestamp(v[c]).tz_localize(session_timezone), axis=1
            )

    return cur_df


def get_current_time(use_utc=False):
    if use_utc:
        return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
    else:
        return datetime.datetime.now(pytz.timezone(os.environ["DEFAULT_TIMEZONE"]))


def get_current_time_string(use_utc=False):
    return get_current_time(use_utc).strftime("%Y-%m-%d %H:%M:%S")


def set_MYSQL_DATABASE():
    """Set the MYSQL_DATABASE environment variable"""
    # If it's already been set, don't take any action
    # (this applies to our docker containers, which have it pre-set)
    # if "MYSQL_DATABASE" in os.environ:
    #    return

    # Otherwise, set it according to whether we should be in production or not
    if get_env_bool("USE_PRODUCTION_DATABASE"):
        os.environ["MYSQL_DATABASE"] = os.environ["MYSQL_PRODUCTION_DATABASE"]
    else:
        os.environ["MYSQL_DATABASE"] = os.environ["MYSQL_TEST_DATABASE"]


def iso_timestamp_to_utc(
    iso_timestamp_string,
):
    """e.g. converts an ISO-formatted string to UTC which is useful
    when adding to a mariadb TIMESTAMP which has no timezone awareness
    e.g. "2023-03-03T09:00:00+07:00" ->
         "2023-03-03 02:00:00+00:00"
    """
    try:
        ts = dateutil.parser.parse(iso_timestamp_string)
    except dateutil.parser._parser.ParserError as e:
        raise ValueError(f"Timestamp '{iso_timestamp}' is invalid. {e}")
    else:
        iso_ts = ts.astimezone(dateutil.tz.UTC)
        return iso_ts.strftime("%Y-%m-%d %H:%M:%S")


def get_tz_offset(iana_tz=None):
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


def validate_tz_offset(iana_tz, tz_offset):
    """e.g. Validates that '07:00' is the correct offset from UTC for 'Asia/Bangkok'"""
    assert get_tz_offset(iana_tz) == tz_offset


def get_next_word_after_from(input_string):
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


def get_payload_info(payload):
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


def prepare_scripts(test_db: str, schema_filepaths: typing.List[str]):
    """Return a string consisting of all the contents of the scripts provided

    Parameters:
        test_db: a string
        schema_filepaths: a list of strings which are paths to scripts

    """
    assert not isinstance(
        schema_filepaths, str
    ), f"schema_filepaths ({schema_filepaths}) must be a list"
    program_to_run0 = ""

    # Run all the scripts requested
    for schema_filepath in schema_filepaths:
        program_to_run0 += f"\n\n\nUSE {test_db};\n"
        with open(schema_filepath, "r") as f:
            program_to_run0 += f.read()
        program_to_run0 += "\n\n\n"

    return program_to_run0
