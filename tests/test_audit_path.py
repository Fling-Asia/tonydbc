"""
Test module for AUDIT_PATH functionality in TonyDBC

This module tests the audit logging functionality that tracks query performance
when AUDIT_PATH environment variable is set using real MariaDB containers.

AUDIT_PATH can be set to:
- "database" - logs to the `tony` table in the database
- "/path/to/file.csv" - logs to a CSV file
- "" or unset - disables audit logging
"""

import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Generator
from unittest.mock import patch

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
os.environ.setdefault("MYSQL_TEST_DATABASE", "test")

# Add the src directory to the path so we can import tonydbc
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from testcontainers.core.container import DockerContainer

import tonydbc


def _wait_db(host: str, port: int, user: str, password: str, database: str, timeout: int = 60) -> None:
    """Wait for database to be ready"""
    start_time = time.time()
    last_err = None
    
    while time.time() - start_time < timeout:
        try:
            conn = mariadb.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
            )
            conn.close()
            return
        except Exception as e:
            last_err = e
            time.sleep(1)
    raise TimeoutError(f"Database not ready after {timeout} seconds; last error: {last_err}")


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


class TestAuditPath:
    """Test AUDIT_PATH functionality with real MariaDB containers"""

    def test_audit_path_database_logging(self, tonydbc_instance):
        """Test AUDIT_PATH = 'database' logs to the tony table"""
        # Set AUDIT_PATH to database
        with patch.dict(os.environ, {"AUDIT_PATH": "database"}):
            # Create a new TonyDBC instance with audit enabled
            db = tonydbc.TonyDBC(
                host=tonydbc_instance.host,
                port=tonydbc_instance.port,
                user=tonydbc_instance.user,
                password=tonydbc_instance.password,
                database=tonydbc_instance.database,
            )
            
            with db:
                # Verify audit is enabled
                assert db.do_audit is True
                assert db.ipath == "database"
                
                # Clear any existing audit records for this session
                db.execute("DELETE FROM tony WHERE session_uuid = %s", (db.session_uuid,))
                
                # Execute a test query that should be audited
                test_query = "SELECT 1 as test_column"
                result = db.get_data(test_query)
                
                # Verify the query executed correctly
                assert len(result) == 1
                assert result[0]["test_column"] == 1
                
                # Give a small delay to ensure audit record is written
                time.sleep(0.1)
                
                # Check that audit record was created
                audit_records = db.get_data(
                    "SELECT * FROM tony WHERE session_uuid = %s ORDER BY id DESC LIMIT 1",
                    (db.session_uuid,)
                )
                
                assert len(audit_records) == 1
                audit_record = audit_records[0]
                
                # Verify audit record contains expected data
                assert audit_record["query"] == test_query
                assert audit_record["method"] == "get_data"
                assert audit_record["session_uuid"] == db.session_uuid
                assert audit_record["host"] == db.host
                assert audit_record["database_name"] == db.database
                assert audit_record["num_rows"] == 1
                assert audit_record["num_cols"] == 1
                assert audit_record["duration_seconds"] is not None
                assert audit_record["duration_seconds"] > 0

    def test_audit_path_csv_file_logging(self, tonydbc_instance):
        """Test AUDIT_PATH = '/path/to/file.csv' logs to CSV file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "test_audit.csv"
            
            # Set AUDIT_PATH to CSV file
            with patch.dict(os.environ, {"AUDIT_PATH": str(csv_path)}):
                # Create a new TonyDBC instance with audit enabled
                db = tonydbc.TonyDBC(
                    host=tonydbc_instance.host,
                    port=tonydbc_instance.port,
                    user=tonydbc_instance.user,
                    password=tonydbc_instance.password,
                    database=tonydbc_instance.database,
                )
                
                with db:
                    # Verify audit is enabled
                    assert db.do_audit is True
                    assert db.ipath == str(csv_path.resolve())
                    
                    # Execute a test query that should be audited
                    test_query = "SELECT 2 as test_column"
                    result = db.get_data(test_query)
                    
                    # Verify the query executed correctly
                    assert len(result) == 1
                    assert result[0]["test_column"] == 2
                
                # After context manager exits, CSV file should exist
                assert csv_path.exists()
                
                # Read and verify CSV content
                audit_df = pd.read_csv(csv_path)
                assert len(audit_df) == 1
                
                audit_record = audit_df.iloc[0]
                assert audit_record["query"] == test_query
                assert audit_record["method"] == "get_data"
                assert audit_record["num_rows"] == 1
                assert audit_record["num_cols"] == 1
                assert audit_record["duration_seconds"] > 0

    def test_audit_path_disabled(self, tonydbc_instance):
        """Test AUDIT_PATH = '' disables audit logging"""
        # Set AUDIT_PATH to empty string
        with patch.dict(os.environ, {"AUDIT_PATH": ""}):
            # Create a new TonyDBC instance with audit disabled
            db = tonydbc.TonyDBC(
                host=tonydbc_instance.host,
                port=tonydbc_instance.port,
                user=tonydbc_instance.user,
                password=tonydbc_instance.password,
                database=tonydbc_instance.database,
            )
            
            with db:
                # Verify audit is disabled
                assert db.do_audit is False
                assert not hasattr(db, 'ipath')
                
                # Execute a test query
                result = db.get_data("SELECT 3 as test_column")
                
                # Verify the query executed correctly
                assert len(result) == 1
                assert result[0]["test_column"] == 3

    def test_audit_path_force_no_audit(self, tonydbc_instance):
        """Test force_no_audit=True overrides AUDIT_PATH"""
        # Set AUDIT_PATH to database
        with patch.dict(os.environ, {"AUDIT_PATH": "database"}):
            # Create a new TonyDBC instance with force_no_audit=True
            db = tonydbc.TonyDBC(
                host=tonydbc_instance.host,
                port=tonydbc_instance.port,
                user=tonydbc_instance.user,
                password=tonydbc_instance.password,
                database=tonydbc_instance.database,
                force_no_audit=True
            )
            
            with db:
                # Verify audit is disabled despite AUDIT_PATH being set
                assert db.do_audit is False
                assert not hasattr(db, 'ipath')

    def test_audit_path_multiple_queries(self, tonydbc_instance):
        """Test multiple queries are all audited"""
        with patch.dict(os.environ, {"AUDIT_PATH": "database"}):
            # Create a new TonyDBC instance with audit enabled
            db = tonydbc.TonyDBC(
                host=tonydbc_instance.host,
                port=tonydbc_instance.port,
                user=tonydbc_instance.user,
                password=tonydbc_instance.password,
                database=tonydbc_instance.database,
            )
            
            with db:
                # Clear any existing audit records for this session
                db.execute("DELETE FROM tony WHERE session_uuid = %s", (db.session_uuid,))
                
                # Execute multiple test queries
                queries = [
                    "SELECT 1 as first_query",
                    "SELECT 2 as second_query", 
                    "SELECT 3 as third_query"
                ]
                
                for query in queries:
                    db.get_data(query)
                
                # Give a small delay to ensure all audit records are written
                time.sleep(0.2)
                
                # Check that all audit records were created
                audit_records = db.get_data(
                    "SELECT query FROM tony WHERE session_uuid = %s ORDER BY id",
                    (db.session_uuid,)
                )
                
                assert len(audit_records) == len(queries)
                for i, record in enumerate(audit_records):
                    assert record["query"] == queries[i]

    def test_audit_path_different_methods(self, tonydbc_instance):
        """Test that different TonyDBC methods are audited with correct method names"""
        with patch.dict(os.environ, {"AUDIT_PATH": "database"}):
            # Create a new TonyDBC instance with audit enabled
            db = tonydbc.TonyDBC(
                host=tonydbc_instance.host,
                port=tonydbc_instance.port,
                user=tonydbc_instance.user,
                password=tonydbc_instance.password,
                database=tonydbc_instance.database,
            )
            
            with db:
                # Clear any existing audit records for this session
                db.execute("DELETE FROM tony WHERE session_uuid = %s", (db.session_uuid,))
                
                # Test get_data method
                db.get_data("SELECT 1 as test_get_data")
                
                # Test execute method (this one has no_tracking=True by default for some operations,
                # so we'll use a simple SELECT that gets tracked)
                db.execute("SELECT 2 as test_execute", no_tracking=False)
                
                # Give a small delay to ensure audit records are written
                time.sleep(0.2)
                
                # Check audit records
                audit_records = db.get_data(
                    "SELECT method, query FROM tony WHERE session_uuid = %s ORDER BY id",
                    (db.session_uuid,)
                )
                
                # Should have at least the get_data call (execute might be filtered out)
                assert len(audit_records) >= 1
                
                # Check that get_data was recorded
                get_data_records = [r for r in audit_records if r["method"] == "get_data"]
                assert len(get_data_records) >= 1


if __name__ == "__main__":
    # Allow running this test module directly
    pytest.main([__file__, "-v"])
