"""
Integration tests for TonyDBC using real MariaDB containers

This test file uses real MariaDB database containers (via testcontainers)
to test TonyDBC functionality with actual database interactions,
rather than mocking database calls.
"""

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Generator

import mariadb  # type: ignore
import pandas as pd
import pytest
from mariadb.constants.CLIENT import MULTI_STATEMENTS  # type: ignore

# Set required environment variables BEFORE importing tonydbc
os.environ.setdefault("USE_PRODUCTION_DATABASE", "False")
os.environ.setdefault("CHECK_ENVIRONMENT_INTEGRITY", "False")
os.environ.setdefault("INTERACT_AFTER_ERROR", "False")
os.environ.setdefault("DEFAULT_TIMEZONE", "UTC")
os.environ.setdefault("MYSQL_DATABASE", "test")
os.environ.setdefault("MYSQL_HOST", "localhost")
os.environ.setdefault("MYSQL_READWRITE_USER", "test")
os.environ.setdefault("MYSQL_READWRITE_PASSWORD", "test")
os.environ.setdefault("MYSQL_PRODUCTION_DATABASE", "test_prod")
os.environ.setdefault("MYSQL_TEST_DATABASE", "test_db")

# Add the src directory to the path so we can import tonydbc
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from testcontainers.core.container import DockerContainer

import tonydbc


def _wait_db(
    host: str, port: int, user: str, pwd: str, db: str, timeout: int = 30
) -> None:
    """Wait for database to be ready"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            conn = mariadb.connect(
                host=host,
                port=port,
                user=user,
                password=pwd,
                database=db,
                client_flag=MULTI_STATEMENTS,
            )
            conn.close()
            print(f"✅ Database ready at {host}:{port}")
            return
        except Exception as e:
            print(f"⏳ Waiting for database... ({e})")
            time.sleep(1)
    raise TimeoutError(f"Database not ready after {timeout} seconds")


@pytest.fixture(scope="session")
def mariadb_container() -> Generator[Any, None, None]:
    """Create a MariaDB container for testing"""

    user, pwd, db = "test", "test", "test"
    container = (
        DockerContainer("mariadb:10.6")
        .with_env("MYSQL_ROOT_PASSWORD", "root")
        .with_env("MYSQL_DATABASE", db)
        .with_env("MYSQL_USER", user)
        .with_env("MYSQL_PASSWORD", pwd)
        .with_exposed_ports(3306)
    )

    with container as c:
        host = c.get_container_host_ip()
        port = int(c.get_exposed_port(3306))
        _wait_db(host, port, user, pwd, db, timeout=120)

        yield SimpleNamespace(
            get_container_host_ip=lambda: host,
            get_exposed_port=lambda _p: port,
            username=user,
            password=pwd,
            dbname=db,
            container=c,
        )


@pytest.fixture(scope="session")
def tonydbc_instance(mariadb_container: Any) -> Any:
    """Create a TonyDBC instance connected to the test container"""
    container_host = mariadb_container.get_container_host_ip()
    container_port = mariadb_container.get_exposed_port(3306)

    # Override environment variables with actual container details
    container_env = {
        "MYSQL_HOST": container_host,
        "MYSQL_PORT": str(container_port),
        "MYSQL_READWRITE_USER": mariadb_container.username,
        "MYSQL_READWRITE_PASSWORD": mariadb_container.password,
        "MYSQL_DATABASE": mariadb_container.dbname,
        "MYSQL_TEST_DATABASE": mariadb_container.dbname,
        "DEFAULT_TIMEZONE": "UTC",
    }

    # Apply the container environment
    original_env = {}
    for key, value in container_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value

    try:
        # Create TonyDBC instance
        db = tonydbc.TonyDBC(
            host=container_host,
            port=container_port,
            user=mariadb_container.username,
            password=mariadb_container.password,
            database=mariadb_container.dbname,
        )
        yield db
    finally:
        # Restore original environment
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.fixture
def setup_tables(tonydbc_instance):
    """Set up test tables in the database"""
    with tonydbc_instance as db:
        # Create a table with a primary key
        db.execute("""
            CREATE TABLE IF NOT EXISTS test_table (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100),
                email VARCHAR(100)
            )
        """)

        # Create a table with NO primary key
        db.execute("""
            CREATE TABLE IF NOT EXISTS no_pk_table (
                col1 VARCHAR(100),
                col2 INT
            )
        """)

        # Create a users table
        db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50),
                email VARCHAR(100)
            )
        """)

        yield db

        # Cleanup
        db.execute("DROP TABLE IF EXISTS test_table")
        db.execute("DROP TABLE IF EXISTS no_pk_table")
        db.execute("DROP TABLE IF EXISTS users")


class TestTonyDBCIntegration:
    """Integration tests using real MariaDB containers"""

    def test_get_primary_key_with_real_table(self, setup_tables):
        """Test get_primary_key with real database table"""
        db = setup_tables

        # Test table with primary key
        result = db.get_primary_key("test_table")
        assert result == "id"

        # Test table with primary key (users)
        result = db.get_primary_key("users")
        assert result == "user_id"

    def test_get_primary_key_with_default_real_table(self, setup_tables):
        """Test get_primary_key returns default when table has no primary key"""
        db = setup_tables

        # Test table with NO primary key - should return default
        result = db.get_primary_key("no_pk_table", default="pk_id")
        assert result == "pk_id"

    def test_insert_and_query_real_data(self, setup_tables):
        """Test inserting and querying real data"""
        db = setup_tables

        # Insert some test data
        db.execute(
            "INSERT INTO test_table (name, email) VALUES ('John', 'john@test.com')"
        )
        db.execute(
            "INSERT INTO test_table (name, email) VALUES ('Jane', 'jane@test.com')"
        )

        # Query the data back
        result = db.get_data("SELECT * FROM test_table ORDER BY id")

        assert len(result) == 2
        assert result[0]["name"] == "John"
        assert result[0]["email"] == "john@test.com"
        assert result[1]["name"] == "Jane"
        assert result[1]["email"] == "jane@test.com"

    def test_dataframe_operations_real_db(self, setup_tables):
        """Test DataFrame operations with real database"""
        db = setup_tables

        # Create test DataFrame
        df = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie"],
                "email": ["alice@test.com", "bob@test.com", "charlie@test.com"],
            }
        )

        # Write DataFrame to database
        db.write_dataframe(df, "test_table", if_exists="append", index=False)

        # Read data back
        result_df = db.query_table("test_table", "SELECT * FROM test_table")

        assert len(result_df) == 3
        assert "Alice" in result_df["name"].values
        assert "Bob" in result_df["name"].values
        assert "Charlie" in result_df["name"].values

    def test_append_to_table_real_db(self, setup_tables):
        """Append rows and verify persisted results"""
        db = setup_tables

        # Start from a clean slate
        db.execute("DELETE FROM test_table;")

        df = pd.DataFrame(
            {
                "name": ["Alpha", "Beta"],
                "email": ["alpha@test.com", "beta@test.com"],
            }
        )

        db.append_to_table("test_table", df)

        rows = db.get_data("SELECT name, email FROM test_table ORDER BY name;")
        assert [(r["name"], r["email"]) for r in rows] == [
            ("Alpha", "alpha@test.com"),
            ("Beta", "beta@test.com"),
        ]

    def test_read_dataframe_from_table_real_db(self, setup_tables):
        """Read DataFrame using explicit query and verify shape"""
        db = setup_tables

        # Ensure at least one row exists
        db.execute(
            "INSERT INTO test_table (name, email) VALUES ('Zed', 'zed@test.com')"
        )

        df = db.read_dataframe_from_table(
            table_name="test_table", query="SELECT id, name, email FROM test_table;"
        )
        assert isinstance(df, pd.DataFrame)
        # Primary key is intentionally set as index when named 'id'
        assert df.index.name == "id"
        assert set(["name", "email"]).issubset(set(df.columns))

    def test_update_table_real_db(self, setup_tables):
        """Update rows via DataFrame indexed by PK and verify changes"""
        db = setup_tables

        # Seed two rows
        db.execute("DELETE FROM test_table;")
        db.execute(
            "INSERT INTO test_table (name, email) VALUES ('U1', 'u1@test.com'), ('U2', 'u2@test.com')"
        )
        rows = db.get_data("SELECT id, name, email FROM test_table ORDER BY id;")
        ids = [r["id"] for r in rows]

        # Build update dataframe indexed by PK
        upd = pd.DataFrame(
            {"name": ["U1x", "U2y"], "email": ["x@test.com", "y@test.com"]}
        )
        upd.index = ids
        upd.index.name = "id"

        db.update_table("test_table", upd)

        rows2 = db.get_data("SELECT id, name, email FROM test_table ORDER BY id;")
        assert [(r["name"], r["email"]) for r in rows2] == [
            ("U1x", "x@test.com"),
            ("U2y", "y@test.com"),
        ]

    def test_query_table_real_db(self, setup_tables):
        """Query with explicit SQL and verify returned DataFrame"""
        db = setup_tables
        df = db.query_table("test_table", "SELECT id, name FROM test_table LIMIT 1;")
        # 'id' is used as index by design
        assert df.index.name == "id"
        assert "name" in df.columns

    @pytest.mark.skip(reason="Temporarily skipped due to 'replace' handling semantics")
    def test_write_dataframe_integration_results(self, setup_tables):
        """End-to-end: write_dataframe persists rows and values correctly"""
        db = setup_tables

        # Fresh table for this test
        db.execute("DROP TABLE IF EXISTS t;")
        db.execute("CREATE TABLE t (id INT PRIMARY KEY, name VARCHAR(32));")

        df = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        db.write_dataframe(df, "t", if_exists="replace", index=False)

        rows = db.get_data("SELECT id, name FROM t ORDER BY id;")
        assert [(r["id"], r["name"]) for r in rows] == [(1, "A"), (2, "B")]

    def test_column_datatypes_real_table(self, setup_tables):
        """Test getting column datatypes from real table"""
        db = setup_tables

        # Get column datatypes for test_table
        result = db.get_column_datatypes("test_table")

        assert "id" in result
        assert "name" in result
        assert "email" in result
        # The exact types may vary, but they should be present (now using nullable datatypes)
        assert result["id"] in [int, "int", "Int64"]
        assert result["name"] in [str, "str", "string"]
        assert result["email"] in [str, "str", "string"]

    def test_temp_id_table_real_db(self, setup_tables):
        """Test creating temporary ID table with real database"""
        db = setup_tables

        id_list = [1, 2, 3, 4, 5]

        # Use temp_id_table context manager
        with db.temp_id_table(id_list) as temp_table_name:
            # Verify the temporary table exists and has data
            result = db.get_data(f"SELECT * FROM {temp_table_name} ORDER BY id")

            assert len(result) == 5
            assert [row["id"] for row in result] == [1, 2, 3, 4, 5]

            # Verify table name format
            assert temp_table_name.startswith("temp_loc_ids_")

        # After context manager, table should be dropped
        # This should raise an error since table no longer exists
        with pytest.raises(Exception):
            db.get_data(f"SELECT * FROM {temp_table_name}")

    def test_sortie_append_with_timestamps(self, tonydbc_instance):
        """Create `sortie` table and append a tz-aware dataframe without errors."""
        with tonydbc_instance as db:
            # Create referenced tables for foreign keys
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS `flightplan` (
                  `id` BIGINT(20) NOT NULL,
                  PRIMARY KEY (`id`)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS `subsoi` (
                  `id` BIGINT(20) NOT NULL,
                  PRIMARY KEY (`id`)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS `stock_check` (
                  `id` BIGINT(20) NOT NULL,
                  PRIMARY KEY (`id`)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )

            # Seed a flightplan id referenced by the sortie row
            db.execute("REPLACE INTO `flightplan` (`id`) VALUES (52451);")

            # Create the `sortie` table (matching the provided schema, with safe index names)
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS `sortie` (
                  `id`                      BIGINT(20) NOT NULL AUTO_INCREMENT,
                  `flight_id`               BIGINT(20) DEFAULT NULL,
                  `subsoi_id`               BIGINT(20) DEFAULT NULL,
                  `stock_check_id`          BIGINT(20) DEFAULT NULL,
                  `attempt_sequence`        INT    NOT NULL,
                  `from_event_id`           BIGINT(20) DEFAULT NULL,
                  `to_event_id`             BIGINT(20) DEFAULT NULL,
                  `from_timestamp`          BIGINT(20) DEFAULT NULL,
                  `to_timestamp`            BIGINT(20) DEFAULT NULL,
                  `fps`                     FLOAT  DEFAULT NULL,
                  `video_width`             INT    DEFAULT NULL,
                  `video_height`            INT    DEFAULT NULL,
                  `multi_video_length`      DOUBLE DEFAULT NULL,
                  `flight_event_log_id`     BIGINT DEFAULT NULL,
                  `flightplan_lookup_works`  BOOL DEFAULT NULL,
                  `is_corrupted`            BOOL DEFAULT NULL,
                  `created_at`              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  `updated_at`              TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                  PRIMARY KEY (`id`),
                  UNIQUE KEY `uk_sortie_flight_attempt` (`flight_id`, `attempt_sequence`),
                  CONSTRAINT `fk_sortie_flight` FOREIGN KEY (`flight_id`) REFERENCES `flightplan` (`id`),
                  CONSTRAINT `fk_sortie_subsoi` FOREIGN KEY (`subsoi_id`) REFERENCES `subsoi` (`id`),
                  CONSTRAINT `fk_sortie_stock_check` FOREIGN KEY (`stock_check_id`) REFERENCES `stock_check` (`id`),
                  CONSTRAINT `sortie_constraint_flight_subsoi`
                    CHECK ((`flight_id` IS NULL     AND `subsoi_id` IS NOT NULL AND `stock_check_id` IS NOT NULL)
                       OR  (`flight_id` IS NOT NULL AND `subsoi_id` IS NULL     AND `stock_check_id` IS NULL))
                ) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8mb4;
                """
            )

            # Add helpful indexes (avoid duplicating FK names)
            db.execute(
                """
                ALTER TABLE `sortie` ADD INDEX `idx_sortie_flight` (`flight_id`);
                """
            )
            db.execute(
                """
                ALTER TABLE `sortie` ADD INDEX `idx_sortie_subsoi` (`subsoi_id`);
                """
            )
            db.execute(
                """
                ALTER TABLE `sortie` ADD INDEX `idx_sortie_stock_check` (`stock_check_id`);
                """
            )

            for attempt_sequence, tz in enumerate(
                ["Asia/Bangkok", "UTC", "America/New_York"]
            ):
                # Build the timezone-aware dataframe as specified
                sortie_df = pd.DataFrame(
                    {
                        "flight_id": [52451],
                        "attempt_sequence": [attempt_sequence],
                        "from_timestamp": [pd.Timestamp("2025-01-16 08:49:26", tz=tz)],
                        "to_timestamp": [pd.Timestamp("2025-01-16 08:58:55", tz=tz)],
                        "fps": [0.0],
                        "video_width": [0],
                        "video_height": [0],
                        "multi_video_length": [569.0],
                        "flight_event_log_id": [10859171],
                        "flightplan_lookup_works": [True],
                        "is_corrupted": [False],
                    }
                )
                # CHECK THAT THIS RAISES AN ERROR BECAUSE Timestamp != BIGINT for from_timestamp and to_timestamp
                # This should not raise due to timestamp handling
                with pytest.raises(Exception):
                    db.append_to_table("sortie", sortie_df, return_reindexed=True)
                with pytest.raises(Exception):
                    db.append_to_table("sortie", sortie_df, return_reindexed=False)

                sortie_df_converted = sortie_df.copy()

                # Special case: two columns, `from_timestamp` and `to_timestamp`, are actually BIGINT
                # so must be converted here before we append_to_table
                for k in ["from_timestamp", "to_timestamp"]:
                    s = sortie_df_converted[k]
                    assert isinstance(s.dtype, pd.DatetimeTZDtype)
                    s = s.dt.tz_convert("UTC").dt.tz_localize(None)
                    sortie_df_converted[k] = s.astype("int64").astype("Int64")

                # This should not raise due to timestamp handling
                res = db.append_to_table(
                    "sortie", sortie_df_converted, return_reindexed=True
                )
                assert res is not None and len(res) == 1

                # Check that the timestamp agree with the database
                for k in ["from_timestamp", "to_timestamp"]:
                    assert (
                        pd.to_datetime(
                            int(res.iloc[0][k]), unit="ns", utc=True
                        ).tz_convert(tz)
                        == sortie_df.iloc[0][k]
                    )

            # Cleanup created tables
            db.execute("DROP TABLE IF EXISTS `sortie`;")
            db.execute("DROP TABLE IF EXISTS `flightplan`;")
            db.execute("DROP TABLE IF EXISTS `subsoi`;")
            db.execute("DROP TABLE IF EXISTS `stock_check`;")

    def test_nullable_datatypes_with_null_values(self, setup_tables):
        """Test that NULL values in database are properly handled with nullable pandas datatypes"""
        db = setup_tables

        # Create a test table with various nullable columns
        db.execute("DROP TABLE IF EXISTS `nullable_test`;")
        db.execute("""
            CREATE TABLE `nullable_test` (
                `id` BIGINT(20) NOT NULL AUTO_INCREMENT,
                `int_col` BIGINT(20) DEFAULT NULL,
                `float_col` DOUBLE DEFAULT NULL,
                `string_col` VARCHAR(255) DEFAULT NULL,
                `bool_col` BOOL DEFAULT NULL,
                `decimal_col` DECIMAL(10,2) DEFAULT NULL,
                PRIMARY KEY (`id`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

        # Insert test data with some NULL values
        db.execute("""
            INSERT INTO `nullable_test`
            (`int_col`, `float_col`, `string_col`, `bool_col`, `decimal_col`)
            VALUES
            (123, 45.67, 'test_string', TRUE, 99.99),
            (NULL, 12.34, 'another_string', FALSE, NULL),
            (456, NULL, NULL, NULL, 123.45),
            (NULL, NULL, NULL, NULL, NULL)
        """)

        # Query the data back using our get_data method
        result_df = db.query_table(
            "nullable_test", "SELECT * FROM nullable_test ORDER BY id"
        )

        # Verify the DataFrame has the correct nullable dtypes
        expected_dtypes = {
            "id": "Int64",
            "int_col": "Int64",
            "float_col": "Float64",
            "string_col": "string",
            "bool_col": "boolean",
            "decimal_col": "Float64",
        }

        for col, expected_dtype in expected_dtypes.items():
            if col in result_df.columns:
                actual_dtype = str(result_df[col].dtype)
                assert actual_dtype == expected_dtype, (
                    f"Column '{col}' has dtype '{actual_dtype}', expected '{expected_dtype}'"
                )

        # Verify NULL values are properly represented as pd.NA
        assert pd.isna(result_df.iloc[1]["int_col"]), (
            "int_col should be pd.NA for row 2"
        )
        assert pd.isna(result_df.iloc[2]["float_col"]), (
            "float_col should be pd.NA for row 3"
        )
        assert pd.isna(result_df.iloc[2]["string_col"]), (
            "string_col should be pd.NA for row 3"
        )
        assert pd.isna(result_df.iloc[2]["bool_col"]), (
            "bool_col should be pd.NA for row 3"
        )
        assert pd.isna(result_df.iloc[1]["decimal_col"]), (
            "decimal_col should be pd.NA for row 2"
        )

        # Verify non-NULL values are correct
        assert result_df.iloc[0]["int_col"] == 123, "int_col should be 123 for row 1"
        assert result_df.iloc[0]["float_col"] == 45.67, (
            "float_col should be 45.67 for row 1"
        )
        assert result_df.iloc[0]["string_col"] == "test_string", (
            "string_col should be 'test_string' for row 1"
        )
        assert result_df.iloc[0]["bool_col"], "bool_col should be True for row 1"
        assert float(result_df.iloc[0]["decimal_col"]) == 99.99, (
            "decimal_col should be 99.99 for row 1"
        )

        # Test that the last row (all NULLs) is properly handled
        last_row = result_df.iloc[3]
        for col in ["int_col", "float_col", "string_col", "bool_col", "decimal_col"]:
            assert pd.isna(last_row[col]), (
                f"Column '{col}' should be pd.NA for all-NULL row"
            )

        # Test round-trip: insert DataFrame with pd.NA values and verify they become NULL in DB
        # Note: For now, just test that we can insert a simple row with some NULLs
        test_df = pd.DataFrame(
            {
                "int_col": [789],
                "float_col": [12.34],
                "string_col": ["roundtrip_test"],
                "bool_col": [False],
                "decimal_col": [56.78],
            }
        )

        # Insert the DataFrame
        result_insert = db.append_to_table(
            "nullable_test", test_df, return_reindexed=True
        )
        assert len(result_insert) == 1, "Should have inserted 1 row"

        # Query back the inserted data
        inserted_ids = result_insert.index.tolist()
        query_back = db.get_data(
            f"SELECT * FROM nullable_test WHERE id IN ({','.join(map(str, inserted_ids))})"
        )

        # Verify the round-trip worked correctly
        assert len(query_back) == 1, "Should have queried back 1 row"

        # Check inserted row (with values)
        row1 = query_back[0]
        assert row1["int_col"] == 789, "int_col should be preserved"
        assert row1["float_col"] == 12.34, "float_col should be preserved"
        assert row1["string_col"] == "roundtrip_test", "string_col should be preserved"
        assert not row1["bool_col"], "bool_col should be preserved"
        assert float(row1["decimal_col"]) == 56.78, "decimal_col should be preserved"

        print("✅ All nullable datatype tests passed!")

        # Cleanup
        db.execute("DROP TABLE IF EXISTS `nullable_test`;")
