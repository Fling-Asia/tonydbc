"""
Test module for AUDIT_PATH functionality in TonyDBC

This module tests the audit logging functionality that tracks query performance
when AUDIT_PATH environment variable is set using fresh MariaDB containers.

AUDIT_PATH can be set to:
- "database" - logs to the `tony` table in the database
- "/path/to/file.csv" - logs to a CSV file
- "" or unset - disables audit logging

Note: These tests now use the shared fresh database fixture from conftest.py
to ensure they never connect to production databases.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# Add the src directory to the path so we can import tonydbc
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import tonydbc


@pytest.fixture
def setup_test_user_table(safe_test_env):
    """Set up a simple user table for testing queries"""
    # Create a TonyDBC instance without audit to set up test data
    db = tonydbc.TonyDBC(
        host=safe_test_env["host"],
        port=safe_test_env["port"],
        user=safe_test_env["user"],
        password=safe_test_env["password"],
        database=safe_test_env["database"],
        force_no_audit=True
    )
    
    with db:
        # Create a simple user table for testing
        db.execute("""
            CREATE TABLE IF NOT EXISTS user (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                email VARCHAR(100) NOT NULL
            )
        """)
        
        # Insert some test data
        test_users = pd.DataFrame([
            {"name": "John Doe", "email": "john@example.com"},
            {"name": "Jane Smith", "email": "jane@example.com"},
            {"name": "Bob Johnson", "email": "bob@example.com"}
        ])
        db.append_to_table("user", test_users)
        
        yield safe_test_env


class TestAuditPath:
    """Test AUDIT_PATH functionality with real MariaDB containers"""

    def test_audit_path_database_logging(self, setup_test_user_table):
        """Test AUDIT_PATH = 'database' logs to the tony table"""
        # Set AUDIT_PATH to database
        with patch.dict(os.environ, {"AUDIT_PATH": "database"}):
            # Create a new TonyDBC instance with database audit enabled
            audit_db = tonydbc.TonyDBC(
                host=setup_test_user_table["host"],
                port=setup_test_user_table["port"],
                user=setup_test_user_table["user"],
                password=setup_test_user_table["password"],
                database=setup_test_user_table["database"],
            )

            with audit_db:
                # Verify audit is enabled and set to database
                assert audit_db.do_audit is True
                assert audit_db.ipath == "database"

                # Execute a test query
                result = audit_db.get_data("SELECT * FROM user LIMIT 1")

                # Verify the query executed correctly
                assert len(result) == 1
                assert "name" in result[0]
                assert "email" in result[0]

                # Check that the tony table contains audit records
                # Use a separate connection to avoid interfering with audit connection
                check_db = tonydbc.TonyDBC(
                    host=setup_test_user_table["host"],
                    port=setup_test_user_table["port"],
                    user=setup_test_user_table["user"],
                    password=setup_test_user_table["password"],
                    database=setup_test_user_table["database"],
                    force_no_audit=True
                )
                
                with check_db:
                    # Check that audit records exist in the tony table
                    audit_records = check_db.get_data("SELECT * FROM tony WHERE query LIKE '%SELECT * FROM user LIMIT 1%'")
                    assert len(audit_records) > 0, "No audit records found in tony table"
                    
                    # Verify audit record contains expected fields
                    record = audit_records[0]
                    assert record["table_name"] == "user"
                    assert "SELECT * FROM user LIMIT 1" in record["query"]
                    assert record["method"] == "get_data"
                    assert record["num_rows"] == 1
                    assert record["duration_seconds"] is not None

    def test_audit_path_csv_file_logging(self, setup_test_user_table):
        """Test AUDIT_PATH = '/path/to/file.csv' logs to CSV file"""
        # Create a temporary CSV file for audit logging
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            csv_path = temp_file.name
        
        try:
            # Set AUDIT_PATH to the CSV file path
            with patch.dict(os.environ, {"AUDIT_PATH": csv_path}):
                # Create a new TonyDBC instance with CSV audit enabled
                audit_db = tonydbc.TonyDBC(
                    host=setup_test_user_table["host"],
                    port=setup_test_user_table["port"],
                    user=setup_test_user_table["user"],
                    password=setup_test_user_table["password"],
                    database=setup_test_user_table["database"],
                )

                with audit_db:
                    # Verify audit is enabled and set to CSV file
                    assert audit_db.do_audit is True
                    assert Path(audit_db.ipath).resolve() == Path(csv_path).resolve()

                    # Execute a test query
                    result = audit_db.get_data("SELECT * FROM user LIMIT 1")

                    # Verify the query executed correctly
                    assert len(result) == 1
                    assert "name" in result[0]
                    assert "email" in result[0]

            # Check that the CSV file contains audit records
            assert Path(csv_path).exists(), "CSV audit file was not created"
            
            # Read the CSV file and verify it contains audit data
            audit_df = pd.read_csv(csv_path)
            assert len(audit_df) > 0, "No audit records found in CSV file"
            
            # Verify audit record contains expected fields
            record = audit_df.iloc[0]
            assert record["table_name"] == "user"
            assert "SELECT * FROM user LIMIT 1" in record["query"]
            assert record["method"] == "get_data"
            assert record["num_rows"] == 1
            assert pd.notna(record["duration_seconds"])
            
        finally:
            # Clean up the temporary CSV file
            if Path(csv_path).exists():
                Path(csv_path).unlink()

    def test_audit_path_disabled(self, setup_test_user_table):
        """Test AUDIT_PATH = '' disables audit logging"""
        # Set AUDIT_PATH to empty string
        with patch.dict(os.environ, {"AUDIT_PATH": ""}):
            # Create a new TonyDBC instance with audit disabled
            audit_db = tonydbc.TonyDBC(
                host=setup_test_user_table["host"],
                port=setup_test_user_table["port"],
                user=setup_test_user_table["user"],
                password=setup_test_user_table["password"],
                database=setup_test_user_table["database"],
            )

            with audit_db:
                # Verify audit is disabled
                assert audit_db.do_audit is False
                assert not hasattr(audit_db, 'ipath')

                # Execute a test query
                result = audit_db.get_data("SELECT * FROM user LIMIT 1")

                # Verify the query executed correctly
                assert len(result) == 1
                assert "name" in result[0]
                assert "email" in result[0]
                
                # Verify no audit connection was created
                assert audit_db._audit_db is None

    def test_audit_path_force_no_audit(self, safe_test_env):
        """Test force_no_audit=True overrides AUDIT_PATH"""
        # Set AUDIT_PATH to database
        with patch.dict(os.environ, {"AUDIT_PATH": "database"}):
            # Create a new TonyDBC instance with force_no_audit=True
            db = tonydbc.TonyDBC(
                host=safe_test_env["host"],
                port=safe_test_env["port"],
                user=safe_test_env["user"],
                password=safe_test_env["password"],
                database=safe_test_env["database"],
                force_no_audit=True
            )

            with db:
                # Verify audit is disabled despite AUDIT_PATH being set
                assert db.do_audit is False
                assert not hasattr(db, 'ipath')

    def test_audit_path_multiple_queries(self, setup_test_user_table):
        """Test multiple queries are all audited"""
        # Set AUDIT_PATH to database
        with patch.dict(os.environ, {"AUDIT_PATH": "database"}):
            # Create a new TonyDBC instance with database audit enabled
            audit_db = tonydbc.TonyDBC(
                host=setup_test_user_table["host"],
                port=setup_test_user_table["port"],
                user=setup_test_user_table["user"],
                password=setup_test_user_table["password"],
                database=setup_test_user_table["database"],
            )

            with audit_db:
                # Execute multiple test queries
                result1 = audit_db.get_data("SELECT * FROM user LIMIT 1")
                result2 = audit_db.get_data("SELECT * FROM user LIMIT 2")
                result3 = audit_db.get_data("SELECT COUNT(*) as total FROM user")

                # Verify all queries executed correctly
                assert len(result1) == 1
                assert len(result2) == 2
                assert result3[0]["total"] >= 3  # At least 3 users (may be more from other tests)

                # Check that all queries were audited
                check_db = tonydbc.TonyDBC(
                    host=setup_test_user_table["host"],
                    port=setup_test_user_table["port"],
                    user=setup_test_user_table["user"],
                    password=setup_test_user_table["password"],
                    database=setup_test_user_table["database"],
                    force_no_audit=True
                )
                
                with check_db:
                    # Check that audit records exist for all queries
                    audit_records = check_db.get_data("SELECT * FROM tony WHERE table_name = 'user' ORDER BY id")
                    assert len(audit_records) >= 3, f"Expected at least 3 audit records, found {len(audit_records)}"

    def test_audit_path_different_methods(self, setup_test_user_table):
        """Test that different TonyDBC methods are audited with correct method names"""
        # Set AUDIT_PATH to database
        with patch.dict(os.environ, {"AUDIT_PATH": "database"}):
            # Create a new TonyDBC instance with database audit enabled
            audit_db = tonydbc.TonyDBC(
                host=setup_test_user_table["host"],
                port=setup_test_user_table["port"],
                user=setup_test_user_table["user"],
                password=setup_test_user_table["password"],
                database=setup_test_user_table["database"],
            )

            with audit_db:
                # Test get_data method
                result1 = audit_db.get_data("SELECT * FROM user LIMIT 1")
                assert len(result1) == 1

                # Test query_table method
                result2 = audit_db.query_table("user", "SELECT * FROM user LIMIT 1")
                assert len(result2) == 1

                # Check that different methods were recorded correctly
                check_db = tonydbc.TonyDBC(
                    host=setup_test_user_table["host"],
                    port=setup_test_user_table["port"],
                    user=setup_test_user_table["user"],
                    password=setup_test_user_table["password"],
                    database=setup_test_user_table["database"],
                    force_no_audit=True
                )
                
                with check_db:
                    # Check that audit records exist with correct method names
                    audit_records = check_db.get_data("SELECT * FROM tony WHERE table_name = 'user' ORDER BY id")
                    assert len(audit_records) >= 2, f"Expected at least 2 audit records, found {len(audit_records)}"
                    
                    # Verify method names are recorded correctly
                    methods = [record["method"] for record in audit_records]
                    assert "get_data" in methods, f"get_data method not found in audit records: {methods}"
                    assert "query_table" in methods, f"query_table method not found in audit records: {methods}"


if __name__ == "__main__":
    # Allow running this test module directly
    pytest.main([__file__, "-v"])
