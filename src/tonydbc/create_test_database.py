# -*- coding: utf-8 -*-
########################################
## TEST HARNESS

import os
import pathlib

import pyperclip

from .tony_utils import prepare_scripts
from .tonydbc import TonyDBC


def create_test_database(
    test_db: str,
    source_db: str,
    schema_filepaths: list[str],
    tables_to_copy: dict[str, int],
    custom_commands: list[str],
    host: str,
    user: str,
    password: str,
    users_to_grant: list[str],
) -> None:
    """
    Drops the test_db, recreates all tables, copies the data from a source_db.

    Parameters:
        test_db:          The name of the database to DROP / CREATE
        source_db:        The name of the database to copy table rows from
        schema_filepaths: The SQL Data Definition commands to run.  Will be run in sequence.
        tables_to_copy:   The tables to copy from source_db, and the max # of rows to copy.
        custom_commands:  A list of commands to execute after tables are created
        host:             The IP or domain name to connect to.
        user:             The power user that can create databases, etc.
        password:
        users_to_grant:   The users that will be granted permission to access this test database.
    """
    for schema_filepath in schema_filepaths:
        if not os.path.isfile(schema_filepath):
            raise AssertionError(
                f"Schema file {schema_filepath} does not point to a valid file."
            )
        if not pathlib.Path(schema_filepath).suffix.lower() == ".sql":
            raise AssertionError(f"Schema file {schema_filepath} does not end in .sql.")

    program_to_run = ""

    with TonyDBC(host=host, user=user, password=password, database=source_db) as db:
        if test_db in db.production_databases:
            raise AssertionError(
                f"DANGER DANGER!  You set the test_db to be {test_db}, "
                f"one of our production databases ({db.production_databases})! "
                f"You were about to drop the production database! "
                f"Please talk to your manager for advice on what to do. "
                f"Do NOT bypass this error."
            )

        if test_db in db.databases:
            program_to_run += f"DROP DATABASE {test_db};\n"

    program_to_run += f"CREATE DATABASE {test_db};\n"
    for user_to_grant in users_to_grant:
        # No need to assign yourself permissions :D
        if not user_to_grant == user:
            program_to_run += (
                f"GRANT ALL ON {test_db}.* TO "
                f"'{user_to_grant}'@'%' WITH GRANT OPTION;\n"
            )
            # FYI: No need to "FLUSH PRIVILEGES;" if using the "GRANT" command
    program_to_run += f"USE {test_db};\n"

    # We must run the trigger script AFTER tables are copied since there might be
    # DB triggers which would fail during the INSERT commands above if the triggers
    # were already in place
    non_trigger_scripts = [f for f in schema_filepaths if str(f).find("trigger") == -1]
    trigger_scripts = [f for f in schema_filepaths if str(f).find("trigger") != -1]

    program_to_run += prepare_scripts(test_db, non_trigger_scripts)

    program_to_run += f"USE {test_db};\nSET FOREIGN_KEY_CHECKS = 0;\n"

    # Copy data from the source database
    # Prepend the source database to each entry, e.g. fling_db.wsc_event_log
    tables_to_copy = {
        ".".join([source_db, table]): max_rows
        for table, max_rows in tables_to_copy.items()
    }
    # Let's make it alphabetical for convenience during debugging
    # (note that e.g. "B" comes before "a", in lexicographical order, due to upper case being first)
    tables_to_copy = dict(sorted(tables_to_copy.items()))
    for table_string, max_rows in tables_to_copy.items():
        # Convert from 'src_db.table'
        assert "." in table_string
        cur_source_db, table = table_string.split(".")
        # print(f"COPY {table_string} -> {test_db}.{table}")
        cmd = (
            f"ALTER TABLE `{test_db}`.`{table}` DISABLE KEYS; "
            f"INSERT INTO `{test_db}`.`{table}` "
            f"SELECT * FROM `{cur_source_db}`.`{table}` LIMIT {max_rows}; "
            f"ALTER TABLE `{test_db}`.`{table}` ENABLE KEYS;\n"
        )
        program_to_run += cmd

        # Disabled due to pathetic bugginess and slowness
        # db.execute(
        #    cmd,
        #    before_retry_cmd=f"SET FOREIGN_KEY_CHECKS = 0; TRUNCATE {test_db}.{table};",
        # )
        # TODO: This command is causing 'mariadb.InterfaceError:
        # Lost connection to server during query'
        # The docs say this happens https://dev.mysql.com/doc/refman/8.0/en/gone-away.html
        # because of an INSERT statement that inserts a great many rows.
        # therefore we should find a way to insert in batches, or
        # maybe use a stored procedure or something - not sure if that would help
        # but just re-running the refresh seems to work since we only get the error
        # about 20% of the time.  It happens on wsc_status_log, the biggest table,
        # with 580k+ rows.
        pass

    program_to_run += "\n".join(custom_commands)

    program_to_run += prepare_scripts(test_db, trigger_scripts)

    program_to_run += "\nSET FOREIGN_KEY_CHECKS = 1;"

    pyperclip.copy(program_to_run)

    print(
        """
        SQL CODE has been COPIED TO YOUR CLIPBOARD:

        📂 ----> 📚 ---> 🗃️

        This code runs really slowly and sometimes aborts if we try to run it on
        the weak official mariadb Python connector.

        So instead, please:

        ** RUN it in HeidiSQL or your favourite Database Client.  Thank you! **
    """
    )
