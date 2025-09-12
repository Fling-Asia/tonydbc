"""
Test script to reproduce and fix the timestamp issue with MariaDB Connector/Python

This test creates a local MariaDB database using testcontainers, creates the video table,
and tests inserting data with TIMESTAMP columns to reproduce the error:
"Data type 'Timestamp' in column X not supported in MariaDB Connector/Python"
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
# This prevents KeyError during module import
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

# from testcontainers.mysql import MySqlContainer as MariaDbContainer
from testcontainers.core.container import DockerContainer

import tonydbc


def _wait_db(
    host: str, port: int, user: str, pwd: str, db: str, timeout: int = 120
) -> None:
    start = time.time()
    last_err = None
    while time.time() - start < timeout:
        try:
            conn = mariadb.connect(
                host=host, port=int(port), user=user, password=pwd, database=db
            )
            conn.close()
            return
        except Exception as e:
            last_err = e
            time.sleep(1)
    raise TimeoutError(f"DB not ready in {timeout}s; last error: {last_err}")


@pytest.fixture(scope="session")
def mariadb_container() -> Generator[Any, None, None]:
    """Create a MariaDB container for testing (no deprecated waits)."""
    user = pwd = "test"
    db = "test"
    container = (
        DockerContainer("mariadb:11.4")
        .with_env("MYSQL_ROOT_PASSWORD", "test")
        .with_env("MYSQL_DATABASE", "test")
        .with_env("MYSQL_USER", "test")
        .with_env("MYSQL_PASSWORD", "test")
        .with_exposed_ports(3306)
    )
    with container as c:
        # WE CANNOT USE THIS due to a deprecation warning in testcontainers (as of version 4.13.0)
        # Wait for the DB to actually be ready
        # wait_for_logs(container, r"(mariadbd|mysqld): ready for connections\.", timeout=120)
        # Instead, we "roll our own" wait for the DB to be ready

        host = c.get_container_host_ip()
        port = int(c.get_exposed_port(3306))

        _wait_db(host, port, user, pwd, db, timeout=120)

        # Yield a lightweight handle with the bits your tests use
        yield SimpleNamespace(
            get_container_host_ip=lambda: host,
            get_exposed_port=lambda _p: port,
            username=user,
            password=pwd,
            dbname=db,
            container=c,  # keep original if you need it
        )


@pytest.fixture(scope="session")
def tonydbc_instance(mariadb_container: Any) -> Any:
    """Create a TonyDBC instance connected to the test container"""
    # Get the actual container connection details
    container_host = mariadb_container.get_container_host_ip()
    container_port = mariadb_container.get_exposed_port(3306)

    # DEBUG: Print the actual container details
    print(f"DEBUG: Container host = {container_host}")
    print(f"DEBUG: Container port = {container_port}")
    print(f"DEBUG: Container username = {mariadb_container.username}")
    print(f"DEBUG: Container password = {mariadb_container.password}")
    print(f"DEBUG: Container database = {mariadb_container.dbname}")

    # Override environment variables for TonyDBC with actual container details
    container_env = {
        "MYSQL_HOST": container_host,
        "MYSQL_PORT": str(container_port),
        "MYSQL_READWRITE_USER": mariadb_container.username,
        "MYSQL_READWRITE_PASSWORD": mariadb_container.password,
        "MYSQL_DATABASE": mariadb_container.dbname,
        "MYSQL_TEST_DATABASE": mariadb_container.dbname,
        "DEFAULT_TIMEZONE": "UTC",
        "CHECK_ENVIRONMENT_INTEGRITY": "False",
        "USE_PRODUCTION_DATABASE": "False",
    }

    # Store original values to restore later
    original_env = {}
    for key, value in container_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value

    try:
        # FIRST: Test raw MariaDB connection to make sure container is accessible
        print("Testing raw MariaDB connection first...")
        test_conn = mariadb.connect(
            host=container_host,
            port=int(container_port),
            user=mariadb_container.username,
            password=mariadb_container.password,
            database=mariadb_container.dbname,
        )
        print("✅ Raw MariaDB connection successful!")
        test_conn.close()

        # SECOND: Create TonyDBC instance with container connection details
        print("Creating TonyDBC instance...")
        db_instance = tonydbc.TonyDBC(
            host=container_host,
            port=int(container_port),
            user=mariadb_container.username,
            password=mariadb_container.password,
            database=mariadb_container.dbname,
            autocommit=True,
        )
        print("✅ TonyDBC instance created successfully!")
        yield db_instance
    finally:
        # Restore original environment variables
        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


@pytest.fixture(scope="session")
def setup_tables(
    mariadb_container: Any, tonydbc_instance: Any
) -> Generator[Any, None, None]:
    """Set up the required tables for testing"""

    # Get the actual container connection details
    actual_host = mariadb_container.get_container_host_ip()
    actual_port = mariadb_container.get_exposed_port(3306)

    print("Creating temp connection to database")
    print(f"Container host: {actual_host}")
    print(f"Container port: {actual_port}")
    print(f"TonyDBC instance host: {tonydbc_instance.host}")
    print(f"TonyDBC instance port: {getattr(tonydbc_instance, 'port', 'NO_PORT_ATTR')}")

    temp_conn = mariadb.connect(
        host=actual_host,
        port=int(actual_port),
        user=mariadb_container.username,
        password=mariadb_container.password,
        database=mariadb_container.dbname,
        client_flag=MULTI_STATEMENTS,
        autocommit=False,
        read_timeout=3600,
        write_timeout=3600,
        local_infile=True,
        compress=True,
    )
    print("Connection worked!")
    with temp_conn.cursor() as cursor:
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        print(f"Result: {result}")
        cursor.execute("SHOW TABLES;")
        tables = cursor.fetchall()
        print(f"Tables: {tables}")
    temp_conn.close()
    print("Connection closed!")

    with tonydbc_instance as db:
        print("tonydbc instance entered successfully")
        # Create sortie table
        db.execute("""
            CREATE TABLE IF NOT EXISTS `sortie` (
                `id` BIGINT(20) NOT NULL AUTO_INCREMENT,
                `name` VARCHAR(255) DEFAULT NULL,
                `description` TEXT DEFAULT NULL,
                `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                PRIMARY KEY (`id`)
            ) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8mb4;
        """)

        # Create machine table
        db.execute("""
            CREATE TABLE IF NOT EXISTS `machine` (
                `id` BIGINT(20) NOT NULL AUTO_INCREMENT,
                `name` VARCHAR(255) DEFAULT NULL,
                `hostname` VARCHAR(255) DEFAULT NULL,
                `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                PRIMARY KEY (`id`)
            ) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8mb4;
        """)

        # Create the video table with exact schema from production
        db.execute("""
            CREATE TABLE IF NOT EXISTS `video` (
              `id`              BIGINT(20) NOT NULL AUTO_INCREMENT,
              `sortie_id`       BIGINT(20) NOT NULL,
              `event_log_id`    BIGINT(20) DEFAULT NULL,
              `video_sequence`  INT    NOT NULL,
              `full_path`       NVARCHAR(1000) DEFAULT NULL,
              `name`            NVARCHAR(1000) DEFAULT NULL,
              `machine_id`      BIGINT(20) DEFAULT NULL,
              `size`            BIGINT DEFAULT NULL,
              `create_time_DEPRECATED`     BIGINT(20) DEFAULT NULL,
              `file_created_at` TIMESTAMP DEFAULT NULL,
              `width`           INT NOT NULL,
              `height`          INT NOT NULL,
              `num_frames`      BIGINT(20) NOT NULL,
              `fps`             DOUBLE NOT NULL,
              `duration`        DOUBLE NOT NULL,
              `is_valid`        BOOL   DEFAULT NULL,
              `created_at`      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              `updated_at`      TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
              PRIMARY KEY (`id`),
              UNIQUE KEY `uk_video_sc_flight_seq` (`sortie_id`, `video_sequence`),
              CONSTRAINT `fk_video_sortie`    FOREIGN KEY (`sortie_id`)    REFERENCES `sortie` (`id`),
              CONSTRAINT `fk_video_machine`   FOREIGN KEY (`machine_id`)   REFERENCES `machine` (`id`)
            ) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8mb4;
        """)

        # Add indexes
        db.execute(
            "ALTER TABLE `video` ADD INDEX IF NOT EXISTS `fk_video_sortie` (`sortie_id`)"
        )
        db.execute(
            "ALTER TABLE `video` ADD INDEX IF NOT EXISTS `fk_video_machine` (`machine_id`)"
        )

        # Insert test data for foreign keys
        db.execute("""
            INSERT INTO sortie (name, description) VALUES
            ('Test Sortie 1', 'First test sortie for video testing'),
            ('Test Sortie 2', 'Second test sortie for video testing'),
            ('Test Sortie 3', 'Third test sortie for timestamp testing')
        """)

        db.execute("""
            INSERT INTO machine (name, hostname) VALUES
            ('Test Machine 1', 'testmachine1.local'),
            ('Test Machine 2', 'testmachine2.local'),
            ('Recording Station Alpha', 'recorder-alpha.local')
        """)

        # Verify the setup worked
        sortie_count = db.get_data("SELECT COUNT(*) as count FROM sortie")[0]["count"]
        machine_count = db.get_data("SELECT COUNT(*) as count FROM machine")[0]["count"]

        print(f"✅ Setup complete: {sortie_count} sorties, {machine_count} machines")

        yield db


def create_test_video_dataframe(
    num_rows: int = 5, base_sequence: int = 1
) -> pd.DataFrame:
    """Create a test DataFrame with video data including timestamp columns"""

    # Create various timestamp formats to test
    now = pd.Timestamp.now(tz="UTC")
    timestamps = [now - pd.Timedelta(hours=i, minutes=i * 15) for i in range(num_rows)]

    data = {
        "sortie_id": [1, 2, 1, 2, 3][:num_rows],  # Mix different sorties
        "event_log_id": [None] * num_rows,
        "video_sequence": list(range(base_sequence, base_sequence + num_rows)),
        "full_path": [
            f"/recordings/sortie_{i // 2 + 1}/video_{i + base_sequence:03d}.mp4"
            for i in range(num_rows)
        ],
        "name": [f"video_{i + base_sequence:03d}.mp4" for i in range(num_rows)],
        "machine_id": [1, 2, 1, 2, 3][:num_rows],  # Mix different machines
        "size": [
            1024 * 1024 * (i + 1) * 50 for i in range(num_rows)
        ],  # 50MB, 100MB, etc.
        "create_time_DEPRECATED": [
            int(now.timestamp()) + i * 3600 for i in range(num_rows)
        ],
        "file_created_at": timestamps,  # This is the problematic TIMESTAMP column
        "width": [1920, 1920, 3840, 1920, 3840][:num_rows],  # Mix HD and 4K
        "height": [1080, 1080, 2160, 1080, 2160][:num_rows],
        "num_frames": [30 * 60 * (i + 1) for i in range(num_rows)],  # 1min, 2min, etc.
        "fps": [30.0, 29.97, 30.0, 25.0, 30.0][:num_rows],  # Different frame rates
        "duration": [60.0 * (i + 1) for i in range(num_rows)],  # 1min, 2min, etc.
        "is_valid": [True, True, False, True, True][:num_rows],  # Mix valid/invalid
    }

    df = pd.DataFrame(data)

    # Ensure the timestamp column is properly typed
    df["file_created_at"] = pd.to_datetime(df["file_created_at"])

    return df


def test_database_setup(setup_tables: Any) -> None:
    """Test that the database is properly set up with data"""

    with setup_tables as db:
        # Verify tables exist
        tables = db.get_data("SHOW TABLES")
        table_names = [list(t.values())[0] for t in tables]

        expected_tables = ["sortie", "machine", "video"]
        for table in expected_tables:
            assert table in table_names, f"Table {table} not found in database"

        # Verify foreign key data exists
        sorties = db.get_data("SELECT * FROM sortie ORDER BY id")
        machines = db.get_data("SELECT * FROM machine ORDER BY id")

        assert len(sorties) >= 3, f"Expected at least 3 sorties, got {len(sorties)}"
        assert len(machines) >= 3, f"Expected at least 3 machines, got {len(machines)}"

        # Verify sortie data
        assert sorties[0]["name"] == "Test Sortie 1"
        assert "created_at" in sorties[0]
        assert sorties[0]["created_at"] is not None

        # Verify machine data
        assert machines[0]["name"] == "Test Machine 1"
        assert machines[0]["hostname"] == "testmachine1.local"

        print("✅ Database verification passed:")
        print(f"   - Tables: {table_names}")
        print(f"   - Sorties: {len(sorties)}")
        print(f"   - Machines: {len(machines)}")


def test_dataframe_creation() -> None:
    """Test that our test DataFrame is created correctly"""

    df = create_test_video_dataframe(num_rows=3)

    # Verify structure
    assert len(df) == 3, f"Expected 3 rows, got {len(df)}"
    assert "file_created_at" in df.columns, "Missing file_created_at column"
    assert "created_at" not in df.columns, "Should not have created_at (auto-generated)"

    # Verify timestamp column
    assert df["file_created_at"].dtype == "datetime64[ns, UTC]", (
        f"Wrong timestamp dtype: {df['file_created_at'].dtype}"
    )

    # Verify foreign key references
    assert all(df["sortie_id"].isin([1, 2, 3])), "Invalid sortie_id values"
    assert all(df["machine_id"].isin([1, 2, 3])), "Invalid machine_id values"

    # Verify unique constraints won't be violated
    assert df["video_sequence"].nunique() == len(df), "Duplicate video_sequence values"

    print("✅ DataFrame creation test passed:")
    print(f"   - Shape: {df.shape}")
    print(f"   - Timestamp dtype: {df['file_created_at'].dtype}")
    print(f"   - Sample timestamp: {df['file_created_at'].iloc[0]}")


def test_timestamp_data_insertion(setup_tables: Any) -> None:
    """Test pandas timestamp data insertion and data quality"""

    with setup_tables as db:
        # Create test data with pandas timestamps
        videos_df = create_test_video_dataframe(num_rows=2, base_sequence=1)

        print("\n" + "=" * 60)
        print("🔍 TESTING: Timestamp data insertion and quality")
        print("=" * 60)
        print("DataFrame dtypes:")
        for col in ["file_created_at", "created_at", "updated_at"]:
            if col in videos_df.columns:
                print(f"  {col}: {videos_df[col].dtype}")

        # Test data insertion with pandas timestamps
        result_df = db.append_to_table("video", videos_df, return_reindexed=True)

        print("✅ SUCCESS: Data inserted successfully!")
        print(f"   Inserted {len(result_df)} rows with IDs: {result_df.index.tolist()}")

        # Verify the data was inserted correctly and check data quality
        videos = db.get_data(
            f"SELECT * FROM video WHERE id IN ({','.join(map(str, result_df.index))})"
        )

        print("\n📊 DATA QUALITY CHECKS:")
        for video in videos:
            print(f"   Video {video['id']}:")
            print(
                f"     file_created_at: {video['file_created_at']} (type: {type(video['file_created_at'])})"
            )
            print(
                f"     event_log_id: {video['event_log_id']} (should be NULL for test data)"
            )

        # Assertions for data quality
        assert len(videos) == len(videos_df), (
            f"Expected {len(videos_df)} rows, got {len(videos)}"
        )

        # Check that timestamps were preserved correctly
        for video in videos:
            assert video["file_created_at"] is not None, "Timestamp should not be NULL"
            assert video["event_log_id"] is None, (
                "event_log_id should be NULL for test data"
            )

        print("✅ All data quality checks passed!")


def test_timestamp_workarounds(setup_tables: Any) -> None:
    """Test different strategies for handling timestamp data"""

    with setup_tables as db:
        base_df = create_test_video_dataframe(num_rows=2, base_sequence=100)

        print("\n" + "=" * 60)
        print("🔧 TESTING: Timestamp workaround strategies")
        print("=" * 60)

        strategies: list[tuple[str, bool, str | None]] = []

        # Strategy 1: Convert to string
        print("\n1️⃣ Converting timestamps to strings...")
        df_str = base_df.copy()
        df_str["file_created_at"] = df_str["file_created_at"].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        df_str["video_sequence"] = df_str["video_sequence"] + 0  # Keep base sequence

        try:
            result1 = db.append_to_table("video", df_str, return_reindexed=True)
            print(f"✅ String conversion: SUCCESS - {len(result1)} rows inserted")

            # Verify data
            videos = db.get_data(
                f"SELECT * FROM video WHERE id IN ({','.join(map(str, result1.index))})"
            )
            for video in videos:
                print(
                    f"   Video {video['id']}: file_created_at = {video['file_created_at']} (type: {type(video['file_created_at'])})"
                )

            strategies.append(("string", True, None))

        except Exception as e:
            print(f"❌ String conversion: FAILED - {e}")
            strategies.append(("string", False, str(e)))

        # Strategy 2: Convert to Unix timestamp (int)
        print("\n2️⃣ Converting timestamps to Unix timestamps...")
        df_unix = base_df.copy()
        df_unix["file_created_at"] = df_unix["file_created_at"].astype("int64") // 10**9
        df_unix["video_sequence"] = df_unix["video_sequence"] + 200

        try:
            result2 = db.append_to_table("video", df_unix, return_reindexed=True)
            print(f"✅ Unix timestamp: SUCCESS - {len(result2)} rows inserted")

            # Verify data
            videos = db.get_data(
                f"SELECT * FROM video WHERE id IN ({','.join(map(str, result2.index))})"
            )
            for video in videos:
                print(
                    f"   Video {video['id']}: file_created_at = {video['file_created_at']} (type: {type(video['file_created_at'])})"
                )

            strategies.append(("unix", True, None))

        except Exception as e:
            print(f"❌ Unix timestamp: FAILED - {e}")
            strategies.append(("unix", False, str(e)))

        # Strategy 3: Use Python datetime objects
        print("\n3️⃣ Converting to Python datetime objects...")
        df_dt = base_df.copy()
        df_dt["file_created_at"] = df_dt["file_created_at"].apply(
            lambda x: x.to_pydatetime()
        )
        df_dt["video_sequence"] = df_dt["video_sequence"] + 300

        try:
            result3 = db.append_to_table("video", df_dt, return_reindexed=True)
            print(f"✅ Datetime objects: SUCCESS - {len(result3)} rows inserted")

            # Verify data
            videos = db.get_data(
                f"SELECT * FROM video WHERE id IN ({','.join(map(str, result3.index))})"
            )
            for video in videos:
                print(
                    f"   Video {video['id']}: file_created_at = {video['file_created_at']} (type: {type(video['file_created_at'])})"
                )

            strategies.append(("datetime", True, None))

        except Exception as e:
            print(f"❌ Datetime objects: FAILED - {e}")
            strategies.append(("datetime", False, str(e)))

        # Strategy 4: Use None (NULL)
        print("\n4️⃣ Using NULL values...")
        df_null = base_df.copy()
        df_null["file_created_at"] = None
        df_null["video_sequence"] = df_null["video_sequence"] + 400

        try:
            result4 = db.append_to_table("video", df_null, return_reindexed=True)
            print(f"✅ NULL values: SUCCESS - {len(result4)} rows inserted")

            # Verify data
            videos = db.get_data(
                f"SELECT * FROM video WHERE id IN ({','.join(map(str, result4.index))})"
            )
            for video in videos:
                print(
                    f"   Video {video['id']}: file_created_at = {video['file_created_at']} (type: {type(video['file_created_at'])})"
                )

            strategies.append(("null", True, None))

        except Exception as e:
            print(f"❌ NULL values: FAILED - {e}")
            strategies.append(("null", False, str(e)))

        # Summary
        print("\n📊 STRATEGY SUMMARY:")
        successful_strategies = [s for s in strategies if s[1]]
        failed_strategies = [s for s in strategies if not s[1]]

        print(f"✅ Successful strategies: {len(successful_strategies)}")
        for name, _, _ in successful_strategies:
            print(f"   - {name}")

        print(f"❌ Failed strategies: {len(failed_strategies)}")
        for name, _, error in failed_strategies:
            print(f"   - {name}: {error}")

        # Verify total data in table
        total_videos = db.get_data("SELECT COUNT(*) as count FROM video")[0]["count"]
        print(f"\n📈 Total videos in database: {total_videos}")

        # Don't return anything to avoid pytest warning
        assert len(strategies) > 0, "No strategies were tested"


def test_column_type_inspection(setup_tables: Any) -> None:
    """Inspect what types are being sent to MariaDB"""

    with setup_tables as db:
        videos_df = create_test_video_dataframe(num_rows=1)

        print("\n" + "=" * 60)
        print("🔍 INSPECTING: DataFrame column types and values")
        print("=" * 60)

        print(
            f"{'Column':<25} | {'Pandas dtype':<20} | {'Sample Value':<30} | {'Python Type'}"
        )
        print("-" * 100)

        for col in videos_df.columns:
            dtype = videos_df[col].dtype
            sample_value = videos_df[col].iloc[0]
            sample_type = type(sample_value).__name__

            # Truncate long values
            if isinstance(sample_value, str) and len(str(sample_value)) > 25:
                display_value = str(sample_value)[:22] + "..."
            else:
                display_value = str(sample_value)

            print(f"{col:<25} | {str(dtype):<20} | {display_value:<30} | {sample_type}")

        # Check what TonyDBC's serialization expects
        print("\n" + "=" * 60)
        print("🔍 CHECKING: TonyDBC column datatype mapping")
        print("=" * 60)

        col_dtypes = db.get_column_datatypes("video")

        print(
            f"{'Column':<25} | {'Expected Type':<15} | {'Pandas dtype':<20} | {'Match'}"
        )
        print("-" * 80)

        for col, expected_type in col_dtypes.items():
            if col in videos_df.columns:
                actual_dtype = videos_df[col].dtype
                # Simple compatibility check
                is_compatible = (
                    (expected_type is str and "object" in str(actual_dtype))
                    or (expected_type is int and "int" in str(actual_dtype))
                    or (expected_type is float and "float" in str(actual_dtype))
                    or (expected_type is bool and "bool" in str(actual_dtype))
                )
                match_symbol = "✅" if is_compatible else "❌"

                print(
                    f"{col:<25} | {expected_type.__name__:<15} | {str(actual_dtype):<20} | {match_symbol}"
                )

        # Special focus on timestamp columns
        print("\n🕐 TIMESTAMP COLUMN ANALYSIS:")
        timestamp_cols = ["file_created_at", "created_at", "updated_at"]

        for col in timestamp_cols:
            if col in videos_df.columns:
                value = videos_df[col].iloc[0]
                print(f"  {col}:")
                print(f"    Value: {value}")
                print(f"    Type: {type(value)}")
                print(f"    Dtype: {videos_df[col].dtype}")
                if hasattr(value, "tz"):
                    print(f"    Timezone: {value.tz}")


if __name__ == "__main__":
    # Run tests directly if called as script
    pytest.main([__file__, "-v", "-s"])
