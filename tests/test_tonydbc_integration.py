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


def check_docker_available() -> bool:
    """Check if Docker is available and running"""
    try:
        import docker

        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


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
    if not check_docker_available():
        pytest.skip("Docker not available")

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
        result_df = db.query_table("test_table")

        assert len(result_df) == 3
        assert "Alice" in result_df["name"].values
        assert "Bob" in result_df["name"].values
        assert "Charlie" in result_df["name"].values

    def test_column_datatypes_real_table(self, setup_tables):
        """Test getting column datatypes from real table"""
        db = setup_tables

        # Get column datatypes for test_table
        result = db.get_column_datatypes("test_table")

        assert "id" in result
        assert "name" in result
        assert "email" in result
        # The exact types may vary, but they should be present
        assert result["id"] in [int, "int"]
        assert result["name"] in [str, "str"]
        assert result["email"] in [str, "str"]

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
