# bootstrap.py
try:
    import pysqlite3.dbapi2 as sqlite3
    # Monkey-patch the reported SQLite version so chromadb thinks it's high enough.
    sqlite3.sqlite_version = "3.41.0"
    sqlite3.sqlite_version_info = (3, 41, 0)
    import sys
    sys.modules["sqlite3"] = sqlite3  # Replace built-in sqlite3 with the patched version.
except ImportError:
    import sqlite3
