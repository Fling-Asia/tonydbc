#!/usr/bin/env python3
"""
Simple test runner for TonyDBC timestamp tests

Usage:
    python tests/run_tests.py

This uses testcontainers to spin up a MariaDB instance automatically.
No manual database setup required!

Requirements:
    pip install pytest testcontainers[mariadb] mariadb pandas
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import docker
import pytest


# Check if Docker is available and running
def verify_docker_available() -> None:
    """Check if Docker is available and running"""
    # Run Docker check at module level
    try:
        client = docker.from_env()
        client.ping()
    except Exception as e:
        error_msg = f"""
    Docker is not running or not available! {e}

    To fix this:
    1. Install Docker Desktop from https://www.docker.com/products/docker-desktop/
    2. Start Docker Desktop and wait for it to fully load
    3. Verify with: docker --version && docker ps

    These tests require Docker to spin up a MariaDB container for testing.
    """
        raise RuntimeError(error_msg)


def main() -> int:
    """Run the timestamp tests"""
    print("🚀 Starting TonyDBC Timestamp Issue Tests with TestContainers")
    print("=" * 70)
    print("📦 This will automatically:")
    print("   - Spin up a MariaDB container")
    print("   - Create test database and tables")
    print("   - Run timestamp insertion tests")
    print("   - Clean up automatically")
    print("=" * 70)

    # Run pytest with verbose output
    test_file = Path(__file__).parent / "test_timestamp_issue.py"

    exit_code = pytest.main(
        [
            str(test_file),
            "-v",  # Verbose output
            "-s",  # Don't capture output (so we see print statements)
            "--tb=short",  # Shorter traceback format
            "--color=yes",  # Colored output
        ]
    )

    if exit_code == 0:
        print("\n🎉 All tests completed!")
    else:
        print(f"\n❌ Tests failed with exit code: {exit_code}")

    return exit_code


if __name__ == "__main__":
    verify_docker_available()
    sys.exit(main())
