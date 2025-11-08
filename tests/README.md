# TonyDBC Tests

This directory contains comprehensive tests for TonyDBC using **fresh, isolated MariaDB containers** to ensure tests never interfere with production databases.

## Test Architecture

All tests use **disposable Docker containers** managed by `conftest.py`:
- Each test session gets a fresh MariaDB container on `localhost:5000`
- Database is completely isolated and destroyed after tests
- No risk of connecting to production databases
- No manual cleanup required

## Prerequisites

- **Docker** (for MariaDB containers)
- **Python 3.10+**

## Install Test Dependencies

```bash
pip install -r tests/requirements.txt
```

## Running Tests

**Zero setup required!** Just run pytest:

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v -s

# Run specific test files
pytest tests/test_fresh_database.py -v
pytest tests/test_timestamp_issue.py -v
pytest tests/test_tonydbc_integration.py -v

# Test Docker availability first
pytest tests/test_docker_availability.py -v
```

## Test Categories

### 🐳 **Docker Availability Tests** (`test_docker_availability.py`)
- Verifies Docker is running and accessible
- Checks MariaDB image availability
- Tests Docker API functionality
- **Run this first if you have Docker issues**

### 🗄️ **Fresh Database Tests** (`test_fresh_database.py`)
- Tests basic TonyDBC operations with fresh containers
- Validates database connections and queries
- Tests audit logging controls (disabled mode)
- **Demonstrates the fresh database approach**

### 🔗 **Integration Tests** (`test_tonydbc_integration.py`)
- Comprehensive TonyDBC functionality testing
- Tests table operations, primary keys, data types
- Real database operations with fresh containers
- **Full integration testing suite**

### ⏰ **Timestamp Issue Tests** (`test_timestamp_issue.py`)
- Reproduces pandas Timestamp issues with MariaDB
- Tests various timestamp handling strategies
- Validates DataFrame operations with timestamp columns
- **Specific to timestamp/datetime handling bugs**

### 🔍 **Audit Path Tests** (`test_audit_path.py`)
- Tests TonyDBC audit logging functionality
- Currently skipped due to TonyDBC initialization bug
- Tests audit disabled and force_no_audit modes
- **Audit logging feature testing**

### 🧪 **Unit Tests** (`test_tonydbc_unit.py`)
- Mock-based unit tests for TonyDBC methods
- No real database connections (uses mocks)
- Comprehensive method coverage
- **Fast unit testing without containers**

## Key Features

✅ **Production-Safe**: Impossible to connect to production databases  
✅ **Isolated**: Each test session gets a fresh MariaDB container  
✅ **Automatic**: No manual setup or cleanup required  
✅ **Fast**: Shared container per test session  
✅ **Comprehensive**: Tests all major TonyDBC functionality  

## No Manual Cleanup Needed

Unlike traditional database tests, these use **disposable Docker containers**:
- Container is automatically destroyed after tests
- No leftover databases or test data
- No manual `DROP DATABASE` commands needed
- Fresh state for every test run
