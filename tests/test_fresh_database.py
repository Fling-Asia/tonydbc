"""
Test module for fresh database setup using shared MariaDB container

This test uses the shared MariaDB container from conftest.py and validates
that TonyDBC can connect and perform basic operations against a fresh database.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

# Add the src directory to the path so we can import tonydbc
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))



class TestFreshDatabase:
    """Test TonyDBC with fresh MariaDB database"""

    def test_fresh_database_connection(self, fresh_tonydbc_instance):
        """Test basic connection and operations with fresh database"""
        with fresh_tonydbc_instance as connection:
            # Test basic query
            result = connection.get_data("SELECT 1 as test_value")
            assert len(result) == 1
            assert result[0]["test_value"] == 1
            print("Basic query test passed")

            # Test table creation and data insertion
            connection.execute("""
                CREATE TABLE IF NOT EXISTS test_table (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            connection.execute("""
                INSERT INTO test_table (name) VALUES ('test_record')
            """)

            # Test data retrieval
            result = connection.get_data("SELECT * FROM test_table WHERE name = 'test_record'")
            assert len(result) == 1
            assert result[0]["name"] == 'test_record'
            print("Table creation and data insertion test passed")

            # Verify we're connected to the correct database
            db_name = connection.get_data("SELECT DATABASE() as db_name")
            assert db_name[0]["db_name"] == "test_fresh_db"
            print(f"Connected to correct database: {db_name[0]['db_name']}")

            print("All fresh database tests passed!")

    def test_fresh_database_audit_logging_disabled(self, fresh_tonydbc_instance):
        """Test that audit logging is properly disabled"""
        with fresh_tonydbc_instance as connection:
            # This should work without audit errors since AUDIT_PATH is set to ""
            result = connection.get_data("SELECT 'audit_disabled' as test")
            assert result[0]["test"] == 'audit_disabled'
            print("Audit disabled test passed")

        # Test with FORCE_NO_AUDIT = True
        with patch.dict(os.environ, {"FORCE_NO_AUDIT": "True"}):
            with fresh_tonydbc_instance as connection:
                # This should also work without audit errors
                result = connection.get_data("SELECT 'force_no_audit' as test")
                assert result[0]["test"] == 'force_no_audit'
                print("Force no audit test passed")

        print("All audit control tests passed!")

    def test_fresh_database_multiple_operations(self, fresh_tonydbc_instance):
        """Test multiple database operations in sequence"""
        with fresh_tonydbc_instance as connection:
            # Create multiple tables
            connection.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(50) UNIQUE,
                    email VARCHAR(100)
                )
            """)

            connection.execute("""
                CREATE TABLE IF NOT EXISTS posts (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT,
                    title VARCHAR(200),
                    content TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)

            # Insert test data
            connection.execute("""
                INSERT INTO users (username, email) VALUES
                ('testuser1', 'test1@example.com'),
                ('testuser2', 'test2@example.com')
            """)

            # Get user IDs
            users = connection.get_data("SELECT id, username FROM users ORDER BY id")
            assert len(users) == 2

            user1_id, user2_id = users[0]["id"], users[1]["id"]

            # Insert posts
            connection.execute(f"""
                INSERT INTO posts (user_id, title, content) VALUES
                ({user1_id}, 'First Post', 'This is the first post'),
                ({user2_id}, 'Second Post', 'This is the second post')
            """)

            # Test JOIN query
            posts_with_users = connection.get_data("""
                SELECT u.username, p.title, p.content
                FROM users u
                JOIN posts p ON u.id = p.user_id
                ORDER BY u.id
            """)

            assert len(posts_with_users) == 2
            assert posts_with_users[0]["username"] == 'testuser1'
            assert posts_with_users[1]["username"] == 'testuser2'

            print("Multiple database operations test passed")
