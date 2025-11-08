"""
Shared test configuration and fixtures for TonyDBC tests

This module provides shared fixtures that create fresh MariaDB containers
for testing, ensuring tests never connect to production databases.
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Any, Generator
from types import SimpleNamespace

import mariadb  # type: ignore
import pytest


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


@pytest.fixture(scope="session")
def fresh_mariadb_container():
    """
    Create a fresh MariaDB container for all tests in the session.
    
    This fixture ensures that all tests use a clean, isolated database
    instead of potentially connecting to production databases via MYSQL_HOST.
    """
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


@pytest.fixture(scope="session")
def safe_test_env(fresh_mariadb_container):
    """
    Set up safe test environment variables that point to the fresh database.
    
    This fixture overrides potentially dangerous environment variables like
    MYSQL_HOST to ensure tests never accidentally connect to production.
    """
    container_info = fresh_mariadb_container
    
    # Safe test environment that points to our fresh container
    safe_env = {
        "USE_PRODUCTION_DATABASE": "False",
        "CHECK_ENVIRONMENT_INTEGRITY": "False", 
        "INTERACT_AFTER_ERROR": "False",
        "DEFAULT_TIMEZONE": "UTC",
        "MYSQL_HOST": container_info["host"],
        "MYSQL_PORT": str(container_info["port"]),
        "MYSQL_DATABASE": container_info["database"],
        "MYSQL_READWRITE_USER": container_info["user"],
        "MYSQL_READWRITE_PASSWORD": container_info["password"],
        "MYSQL_PRODUCTION_DATABASE": container_info["database"],  # Safe fallback
        "MYSQL_TEST_DATABASE": container_info["database"],
    }
    
    # Store original environment
    original_env = {}
    for key, value in safe_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    print(f"Set safe test environment pointing to fresh database at {container_info['host']}:{container_info['port']}")
    
    yield container_info
    
    # Restore original environment
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture
def fresh_tonydbc_instance(safe_test_env):
    """
    Create a TonyDBC instance connected to the fresh test database.
    
    This fixture provides a ready-to-use TonyDBC instance that is guaranteed
    to be connected to the fresh test database, not production.
    """
    # Import here to avoid circular imports and ensure env vars are set
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    import tonydbc
    
    container_info = safe_test_env
    
    db = tonydbc.TonyDBC(
        host=container_info["host"],
        port=container_info["port"],
        user=container_info["user"],
        password=container_info["password"],
        database=container_info["database"],
        force_no_audit=True  # Disable audit by default to avoid the initialization bug
    )
    
    return db


# Legacy fixture names for backward compatibility
@pytest.fixture(scope="session")
def mariadb_container(fresh_mariadb_container):
    """Legacy alias for fresh_mariadb_container"""
    # Convert to the format expected by existing tests
    container_info = fresh_mariadb_container
    return SimpleNamespace(
        get_container_host_ip=lambda: container_info["host"],
        get_exposed_port=lambda _p: container_info["port"],
        username=container_info["user"],
        password=container_info["password"],
        dbname=container_info["database"],
        container=None,  # Not needed for our implementation
    )


@pytest.fixture(scope="session") 
def tonydbc_instance(fresh_tonydbc_instance):
    """Legacy alias for fresh_tonydbc_instance"""
    return fresh_tonydbc_instance
