"""
Test module for fresh database setup using persistent MariaDB container

This test creates a persistent MariaDB container on localhost:5000 and validates
that TonyDBC can connect and perform basic operations against a fresh database.
"""

import os
import sys
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import mariadb  # type: ignore
import pandas as pd
import pytest

# Set required environment variables BEFORE importing tonydbc
os.environ.setdefault("USE_PRODUCTION_DATABASE", "False")
os.environ.setdefault("CHECK_ENVIRONMENT_INTEGRITY", "False")
os.environ.setdefault("INTERACT_AFTER_ERROR", "False")
os.environ.setdefault("DEFAULT_TIMEZONE", "UTC")

# Add the src directory to the path so we can import tonydbc
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import tonydbc


def _wait_for_database(host: str, port: int, user: str, password: str, database: str, timeout: int = 60) -> None:
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
            print(f"Database ready at {host}:{port}")
            return
        except Exception as e:
            last_err = e
            print(f"Waiting for database... ({e})")
            time.sleep(2)
    
    raise TimeoutError(f"Database not ready after {timeout} seconds; last error: {last_err}")


def _start_mariadb_container() -> dict[str, Any]:
    """Start the MariaDB container and return connection details"""
    container_name = "tonydbc-test-db"
    
    # Stop and remove existing container if it exists
    try:
        subprocess.run(["docker", "stop", container_name], 
                      capture_output=True, check=False)
        subprocess.run(["docker", "rm", container_name], 
                      capture_output=True, check=False)
    except Exception:
        pass  # Container might not exist
    
    # Start new container
    docker_cmd = [
        "docker", "run", "-d", "--name", container_name,
        "-p", "5000:3306",
        "-e", "MYSQL_ROOT_PASSWORD=root",
        "-e", "MYSQL_DATABASE=test_fresh_db",
        "-e", "MYSQL_USER=test",
        "-e", "MYSQL_PASSWORD=test",
        "mariadb:10.6"
    ]
    
    print(f"Starting MariaDB container: {' '.join(docker_cmd)}")
    result = subprocess.run(docker_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to start container: {result.stderr}")
    
    container_id = result.stdout.strip()
    print(f"Container started with ID: {container_id}")
    
    return {
        "container_name": container_name,
        "container_id": container_id,
        "host": "localhost",
        "port": 5000,
        "database": "test_fresh_db",
        "user": "test",
        "password": "test"
    }


def _stop_mariadb_container(container_name: str) -> None:
    """Stop and remove the MariaDB container"""
    try:
        print(f"Stopping container: {container_name}")
        subprocess.run(["docker", "stop", container_name], 
                      capture_output=True, check=True)
        subprocess.run(["docker", "rm", container_name], 
                      capture_output=True, check=True)
        print(f"Container {container_name} stopped and removed")
    except subprocess.CalledProcessError as e:
        print(f"Error stopping container: {e}")


@pytest.fixture(scope="module")
def fresh_mariadb():
    """Start a fresh MariaDB container for testing"""
    container_info = _start_mariadb_container()
    
    # Wait for database to be ready
    _wait_for_database(
        host=container_info["host"],
        port=container_info["port"],
        user=container_info["user"],
        password=container_info["password"],
        database=container_info["database"],
        timeout=120
    )
    
    yield container_info
    
    # Cleanup
    _stop_mariadb_container(container_info["container_name"])


class TestFreshDatabase:
    """Test TonyDBC with fresh MariaDB database"""

    def test_fresh_database_connection(self, fresh_mariadb):
        """Test basic connection and operations with fresh database"""
        db_info = fresh_mariadb
        
        # Override environment variables for this test
        test_env = {
            "MYSQL_HOST": db_info["host"],
            "MYSQL_PORT": str(db_info["port"]),
            "MYSQL_READWRITE_USER": db_info["user"],
            "MYSQL_READWRITE_PASSWORD": db_info["password"],
            "MYSQL_DATABASE": db_info["database"],
            "MYSQL_TEST_DATABASE": db_info["database"],
        }
        
        # Apply test environment
        original_env = {}
        for key, value in test_env.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        try:
            # Create TonyDBC instance
            db = tonydbc.TonyDBC(
                host=db_info["host"],
                port=db_info["port"],
                user=db_info["user"],
                password=db_info["password"],
                database=db_info["database"],
            )
            
            with db:
                # Test 1: Basic query
                result = db.get_data("SELECT 1 as test_value, 'hello' as test_string")
                assert len(result) == 1
                assert result[0]["test_value"] == 1
                assert result[0]["test_string"] == "hello"
                print("Basic query test passed")
                
                # Test 2: Create table and insert data
                db.execute("""
                    CREATE TABLE IF NOT EXISTS test_table (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        name VARCHAR(100),
                        value INT
                    )
                """)
                
                db.execute(
                    "INSERT INTO test_table (name, value) VALUES (%s, %s)",
                    ("test_record", 42)
                )
                
                # Test 3: Query the inserted data
                records = db.get_data("SELECT * FROM test_table WHERE name = 'test_record'")
                assert len(records) == 1
                assert records[0]["name"] == "test_record"
                assert records[0]["value"] == 42
                print("Table creation and data insertion test passed")
                
                # Test 4: Verify we're using the correct database
                db_name_result = db.get_data("SELECT DATABASE() as current_db")
                assert db_name_result[0]["current_db"] == db_info["database"]
                print(f"Connected to correct database: {db_info['database']}")
                
                print("All fresh database tests passed!")
                
        finally:
            # Restore original environment
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_fresh_database_audit_logging_disabled(self, fresh_mariadb):
        """Test that audit logging can be disabled with fresh database"""
        
        db_info = fresh_mariadb
        
        # Override environment variables for this test
        test_env = {
            "MYSQL_HOST": db_info["host"],
            "MYSQL_PORT": str(db_info["port"]),
            "MYSQL_READWRITE_USER": db_info["user"],
            "MYSQL_READWRITE_PASSWORD": db_info["password"],
            "MYSQL_DATABASE": db_info["database"],
            "MYSQL_TEST_DATABASE": db_info["database"],
        }
        
        # Apply test environment
        original_env = {}
        for key, value in test_env.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        try:
            # Test audit disabled (AUDIT_PATH empty)
            with patch.dict(os.environ, {"AUDIT_PATH": ""}):
                db = tonydbc.TonyDBC(
                    host=db_info["host"],
                    port=db_info["port"],
                    user=db_info["user"],
                    password=db_info["password"],
                    database=db_info["database"],
                )
                
                with db:
                    # Verify audit is disabled
                    assert db.do_audit is False
                    assert not hasattr(db, 'ipath')
                    
                    # Execute a test query
                    result = db.get_data("SELECT 'no_audit_test' as test_column")
                    
                    # Verify the query executed correctly
                    assert len(result) == 1
                    assert result[0]["test_column"] == "no_audit_test"
                    
                    print("Audit disabled test passed")
                    
            # Test force_no_audit parameter
            db = tonydbc.TonyDBC(
                host=db_info["host"],
                port=db_info["port"],
                user=db_info["user"],
                password=db_info["password"],
                database=db_info["database"],
                force_no_audit=True
            )
            
            with db:
                # Verify audit is disabled despite any environment settings
                assert db.do_audit is False
                assert not hasattr(db, 'ipath')
                
                # Execute a test query
                result = db.get_data("SELECT 'force_no_audit_test' as test_column")
                
                # Verify the query executed correctly
                assert len(result) == 1
                assert result[0]["test_column"] == "force_no_audit_test"
                
                print("Force no audit test passed")
                
            print("All audit control tests passed!")
                
        finally:
            # Restore original environment
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_fresh_database_multiple_operations(self, fresh_mariadb):
        """Test multiple database operations with fresh database"""
        db_info = fresh_mariadb
        
        # Override environment variables for this test
        test_env = {
            "MYSQL_HOST": db_info["host"],
            "MYSQL_PORT": str(db_info["port"]),
            "MYSQL_READWRITE_USER": db_info["user"],
            "MYSQL_READWRITE_PASSWORD": db_info["password"],
            "MYSQL_DATABASE": db_info["database"],
            "MYSQL_TEST_DATABASE": db_info["database"],
        }
        
        # Apply test environment
        original_env = {}
        for key, value in test_env.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        try:
            db = tonydbc.TonyDBC(
                host=db_info["host"],
                port=db_info["port"],
                user=db_info["user"],
                password=db_info["password"],
                database=db_info["database"],
            )
            
            with db:
                # Test 1: Create multiple tables
                db.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        username VARCHAR(50) UNIQUE,
                        email VARCHAR(100),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                db.execute("""
                    CREATE TABLE IF NOT EXISTS posts (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        user_id INT,
                        title VARCHAR(200),
                        content TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users(id)
                    )
                """)
                
                # Test 2: Insert test data
                db.execute(
                    "INSERT INTO users (username, email) VALUES (%s, %s)",
                    ("testuser", "test@example.com")
                )
                
                # Get the user ID
                user_result = db.get_data("SELECT id FROM users WHERE username = 'testuser'")
                user_id = user_result[0]["id"]
                
                db.execute(
                    "INSERT INTO posts (user_id, title, content) VALUES (%s, %s, %s)",
                    (user_id, "Test Post", "This is a test post content")
                )
                
                # Test 3: Complex query with JOIN
                posts_with_users = db.get_data("""
                    SELECT u.username, p.title, p.content, p.created_at
                    FROM posts p
                    JOIN users u ON p.user_id = u.id
                    WHERE u.username = 'testuser'
                """)
                
                assert len(posts_with_users) == 1
                assert posts_with_users[0]["username"] == "testuser"
                assert posts_with_users[0]["title"] == "Test Post"
                assert posts_with_users[0]["content"] == "This is a test post content"
                
                # Test 4: Update operation
                db.execute(
                    "UPDATE posts SET title = %s WHERE user_id = %s",
                    ("Updated Test Post", user_id)
                )
                
                # Verify update
                updated_post = db.get_data(f"SELECT title FROM posts WHERE user_id = {user_id}")
                assert updated_post[0]["title"] == "Updated Test Post"
                
                # Test 5: Delete operation
                db.execute("DELETE FROM posts WHERE user_id = %s", (user_id,))
                db.execute("DELETE FROM users WHERE id = %s", (user_id,))
                
                # Verify deletion
                remaining_users = db.get_data("SELECT COUNT(*) as count FROM users")
                assert remaining_users[0]["count"] == 0
                
                remaining_posts = db.get_data("SELECT COUNT(*) as count FROM posts")
                assert remaining_posts[0]["count"] == 0
                
                print("Multiple database operations test passed")
                
        finally:
            # Restore original environment
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    # Allow running this test module directly
    pytest.main([__file__, "-v", "-s"])
