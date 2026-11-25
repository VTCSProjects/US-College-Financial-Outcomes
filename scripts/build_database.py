
import sqlite3
from pathlib import Path
import pandas as pd

csv_path = Path("data/processed/college_clean.csv")
db_path = Path("data/college_outcomes.db")

df = pd.read_csv(csv_path)

assert df["college_id"].notna().all()
assert df["college_id"].is_unique

with sqlite3.connect(db_path) as conn:
    # Build the institution table
    df.to_sql(
        "colleges",
        conn,
        if_exists="replace",
        index=False
    )

    # Enforce unique institution identifiers
    conn.execute("""
        CREATE UNIQUE INDEX idx_college_id
        ON colleges(college_id)
    """)

    # Add indexes for common analytical filters
    conn.execute("""
        CREATE INDEX idx_college_state
        ON colleges(state)
    """)

    conn.execute("""
        CREATE INDEX idx_institution_type
        ON colleges(institution_type)
    """)

    print("DATABASE VALIDATION")

    total = conn.execute(
        "SELECT COUNT(*) FROM colleges"
    ).fetchone()[0]

    unique_ids = conn.execute(
        "SELECT COUNT(DISTINCT college_id) FROM colleges"
    ).fetchone()[0]

    print("Total institutions:", total)
    print("Unique college IDs:", unique_ids)

    assert total == len(df)
    assert unique_ids == total

    print("\nINSTITUTIONS BY TYPE")

    query = """
        SELECT
            institution_type,
            COUNT(*) AS institution_count,
            ROUND(AVG(earnings_10yr), 2) AS avg_earnings
        FROM colleges
        GROUP BY institution_type
        ORDER BY avg_earnings DESC
    """

    print(pd.read_sql_query(query, conn).to_string(index=False))

print("\nSUCCESS!")
print("Database saved:", db_path)
