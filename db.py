import sqlite3
import json

DB_NAME = "annotations.db"


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Table for original records (from JSONL)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS source_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        uploaded_file_name TEXT NOT NULL,
        record_index_in_file INTEGER NOT NULL,
        ticker TEXT,
        fy INTEGER,
        full_record_json TEXT, -- Store the original full JSON for reference
        UNIQUE(uploaded_file_name, record_index_in_file)
    )
    """)

    # Table for individual items within the "values" array of a record
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS value_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_record_id INTEGER NOT NULL,
        item_index_in_record INTEGER NOT NULL, -- Original index in the 'values' array
        original_tag TEXT,
        original_value TEXT, -- Store as text, handle type conversion in app
        original_units TEXT,
        original_comments TEXT,
        original_item_json TEXT, -- Store the original item JSON for easy reference
        FOREIGN KEY (source_record_id) REFERENCES source_records (id),
        UNIQUE(source_record_id, item_index_in_record)
    )
    """)

    # Table for annotations
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS annotations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        value_item_id INTEGER NOT NULL UNIQUE, -- Each value_item gets one annotation row
        annotated_value TEXT,
        annotation_comment TEXT,
        status TEXT DEFAULT 'pending', -- e.g., 'pending', 'annotated', 'verified'
        annotator_id TEXT, -- For future use (who made the annotation)
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (value_item_id) REFERENCES value_items (id)
    )
    """)

    # Trigger to update 'updated_at' timestamp on annotation update
    cursor.execute("""
    CREATE TRIGGER IF NOT EXISTS update_annotations_updated_at
    AFTER UPDATE ON annotations
    FOR EACH ROW
    BEGIN
        UPDATE annotations SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
    END;
    """)

    conn.commit()
    conn.close()


def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row  # Access columns by name
    return conn


# --- Query Functions ---


def add_source_record(uploaded_file_name, record_index_in_file, record_data):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO source_records (uploaded_file_name, record_index_in_file, ticker, fy, full_record_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(uploaded_file_name, record_index_in_file) DO NOTHING
        """,
            (
                uploaded_file_name,
                record_index_in_file,
                record_data.get("ticker"),
                record_data.get("fy"),
                json.dumps(record_data),
            ),
        )
        conn.commit()
        record_id = cursor.lastrowid
        if record_id == 0:  # Conflict occurred, record already exists
            cursor.execute(
                """
                SELECT id FROM source_records 
                WHERE uploaded_file_name = ? AND record_index_in_file = ?
            """,
                (uploaded_file_name, record_index_in_file),
            )
            record_id = cursor.fetchone()["id"]
        return record_id
    finally:
        conn.close()


def add_value_item(source_record_id, item_index, item_data):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO value_items (source_record_id, item_index_in_record, original_tag, original_value, original_units, original_comments, original_item_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_record_id, item_index_in_record) DO NOTHING
        """,
            (
                source_record_id,
                item_index,
                item_data.get("tag"),
                str(item_data.get("value", "")),
                item_data.get("units"),
                item_data.get("comments"),
                json.dumps(item_data),
            ),
        )
        conn.commit()
        value_item_id = cursor.lastrowid
        if value_item_id == 0:  # Conflict, item already exists
            cursor.execute(
                """
                SELECT id FROM value_items
                WHERE source_record_id = ? AND item_index_in_record = ?
            """,
                (source_record_id, item_index),
            )
            value_item_id = cursor.fetchone()["id"]

        # Ensure an annotation shell exists
        if value_item_id:
            cursor.execute(
                """
                INSERT INTO annotations (value_item_id, status)
                VALUES (?, 'pending')
                ON CONFLICT(value_item_id) DO NOTHING
            """,
                (value_item_id,),
            )
            conn.commit()
        return value_item_id
    finally:
        conn.close()


def get_record_ids_for_file(uploaded_file_name):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id FROM source_records WHERE uploaded_file_name = ? ORDER BY record_index_in_file ASC",
        (uploaded_file_name,),
    )
    ids = [row["id"] for row in cursor.fetchall()]
    conn.close()
    return ids


def get_source_record_by_id(record_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM source_records WHERE id = ?", (record_id,))
    record = cursor.fetchone()
    conn.close()
    return record  # Returns a Row object or None


def get_value_items_for_record(source_record_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    # Fetch value_items and their corresponding annotations
    cursor.execute(
        """
        SELECT 
            vi.*, 
            a.annotated_value, 
            a.annotation_comment, 
            a.status
        FROM value_items vi
        LEFT JOIN annotations a ON vi.id = a.value_item_id
        WHERE vi.source_record_id = ?
        ORDER BY vi.item_index_in_record ASC
    """,
        (source_record_id,),
    )
    items = cursor.fetchall()
    conn.close()
    return items  # Returns a list of Row objects


def save_annotation(
    value_item_id, annotated_value, annotation_comment, status="annotated"
):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO annotations (value_item_id, annotated_value, annotation_comment, status, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(value_item_id) DO UPDATE SET
                annotated_value = excluded.annotated_value,
                annotation_comment = excluded.annotation_comment,
                status = excluded.status,
                updated_at = CURRENT_TIMESTAMP
        """,
            (value_item_id, annotated_value, annotation_comment, status),
        )
        conn.commit()
    finally:
        conn.close()


def get_distinct_uploaded_file_names():
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT DISTINCT uploaded_file_name FROM source_records ORDER BY uploaded_file_name ASC"
        )
        file_names = [row["uploaded_file_name"] for row in cursor.fetchall()]
        return file_names
    finally:
        conn.close()
