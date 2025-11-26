# Vulture whitelist for tonydbc
# Generated to suppress false positives

# ===== MQTT Client - Required by paho-mqtt callback interface =====
# These parameters are required by the MQTT protocol callbacks even if unused
MAX_RECONNECT_ATTEMPTS  # unused variable (src\tonydbc\mqtt_client.py:9)
_.flags  # unused variable (src\tonydbc\mqtt_client.py:90) - required by on_connect signature
_.userdata  # unused variable (src\tonydbc\mqtt_client.py:90) - required by on_connect signature
_.userdata  # unused variable (src\tonydbc\mqtt_client.py:104) - required by on_disconnect signature
_.userdata  # unused variable (src\tonydbc\mqtt_client.py:119) - required by on_message signature

# ===== Utility Functions - Part of Public API =====
refine_dtype  # unused function (src\tonydbc\dataframe_fast.py:94) - utility function
get_current_time_string  # unused function (src\tonydbc\tony_utils.py:230) - public API
iso_timestamp_to_utc  # unused function (src\tonydbc\tony_utils.py:265) - public API
validate_tz_offset  # unused function (src\tonydbc\tony_utils.py:299) - public API

# ===== TonyDBC Attributes - May be used by subclasses or external code =====
_.auto_reconnect  # unused attribute (src\tonydbc\tonydbc.py:322) - configuration attribute

# ===== Test Fixtures - Used by pytest framework =====
pytest_sessionfinish  # unused function (tests\conftest.py:313) - pytest hook
_.exitstatus  # unused variable (tests\conftest.py:313) - pytest hook parameter
_.session  # unused variable (tests\conftest.py:313) - pytest hook parameter
mariadb_container  # unused function (tests\conftest.py:327) - pytest fixture (legacy name)

# ===== Test Mock Attributes - Used by unittest.mock framework =====
# Mock.return_value and Mock.side_effect are special attributes used by the mock framework
_.return_value  # unused attribute - mock framework attribute
_.side_effect  # unused attribute - mock framework attribute
_.lastrowid  # unused attribute - mock database cursor attribute
_._TonyDBCOnlineOnly__production_databases  # unused attribute - testing private attribute
_.pickle_path  # unused attribute - mock attribute for testing
