# TonyDBC Tests

This directory contains comprehensive tests for TonyDBC, specifically focused on reproducing and fixing timestamp-related issues with MariaDB Connector/Python.

## The Issue

When using `append_to_table()` with DataFrames containing pandas Timestamp columns, you may encounter:

```
mariadb.NotSupportedError: Data type 'Timestamp' in column X not supported in MariaDB Connector/Python
```

## Test Setup

### 1. Prerequisites

- Docker (for testcontainers)
- Python 3.9+

### 2. Install Test Dependencies

```bash
pip install -r tests/requirements.txt
```

### 3. Running Tests

**No manual database setup required!** The tests use testcontainers to automatically:
- Spin up a MariaDB container
- Create test database and tables  
- Run all tests
- Clean up automatically

```bash
# From the project root
python tests/run_tests.py

# Or use pytest directly
pytest tests/test_timestamp_issue.py -v -s
```

## What the Tests Do

### 🏗️ `test_database_setup()`
- Verifies that MariaDB container is running
- Confirms all tables are created correctly
- Validates foreign key data exists
- **Ensures we have a solid foundation for testing**

### 📊 `test_dataframe_creation()`
- Creates realistic video DataFrames with timestamp columns
- Validates DataFrame structure and types
- Confirms foreign key relationships are valid
- **Ensures our test data is correct**

### 🔍 `test_timestamp_issue_reproduction()`
- Creates DataFrames with pandas Timestamp columns
- Attempts `db.append_to_table("video", videos_df)`
- **Should reproduce the original error**
- Provides detailed error analysis

### 🔧 `test_timestamp_workarounds()`
- Tests 4 different strategies for handling timestamps:
  1. **String conversion**: `dt.strftime('%Y-%m-%d %H:%M:%S')`
  2. **Unix timestamps**: Convert to integer seconds
  3. **Python datetime**: Convert to native datetime objects  
  4. **NULL values**: Test with None/NULL
- **Verifies which strategies work and which fail**
- Shows actual inserted data for successful strategies

### 🔍 `test_column_type_inspection()`
- Detailed analysis of DataFrame column types vs. MariaDB expectations
- Shows pandas dtypes vs. TonyDBC expected types
- **Special focus on timestamp column analysis**
- Helps debug type mismatches

## Expected Results

1. **Database setup should pass** ✅
2. **DataFrame creation should pass** ✅  
3. **Timestamp reproduction should fail** ❌ (this demonstrates the bug)
4. **Some workarounds should succeed** ✅ (shows solutions)
5. **Type inspection shows the mismatch** 🔍 (explains why it fails)

## Database Schema

The tests use this video table schema (matching your production schema):

```sql
CREATE TABLE `video` (
  `id` BIGINT(20) NOT NULL AUTO_INCREMENT,
  `sortie_id` BIGINT(20) NOT NULL,
  `file_created_at` TIMESTAMP DEFAULT NULL,  -- This is the problematic column
  `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  -- ... other columns
  PRIMARY KEY (`id`)
);
```

## Cleanup

The tests automatically create and clean up a test database (`test_tonydbc_timestamps`).
If cleanup fails, you can manually drop it:

```sql
DROP DATABASE IF EXISTS test_tonydbc_timestamps;
```
