
import sqlite3
import pandas as pd
from pathlib import Path

db_path = Path("data/college_outcomes.db")
output_dir = Path("outputs/sql")
output_dir.mkdir(parents=True, exist_ok=True)

queries = {
    "state_earnings": """
        SELECT
            state,
            COUNT(earnings_10yr) AS reporting_colleges,
            ROUND(AVG(earnings_10yr), 0) AS avg_earnings,
            ROUND(AVG(net_price), 0) AS avg_net_price
        FROM colleges
        GROUP BY state
        HAVING COUNT(earnings_10yr) >= 20
        ORDER BY avg_earnings DESC
        LIMIT 15
    """,

    "public_private_comparison": """
        SELECT
            institution_type,
            COUNT(*) AS institutions,
            COUNT(net_price) AS price_reporting,
            COUNT(earnings_10yr) AS earnings_reporting,
            ROUND(AVG(net_price), 0) AS avg_net_price,
            ROUND(AVG(graduate_debt), 0) AS avg_grad_debt,
            ROUND(AVG(earnings_10yr), 0) AS avg_earnings
        FROM colleges
        GROUP BY institution_type
        ORDER BY avg_earnings DESC
    """,

    "low_debt_high_earnings": """
        SELECT
            college_name,
            state,
            institution_type,
            graduate_debt,
            earnings_10yr,
            ROUND(
                graduate_debt / NULLIF(earnings_10yr, 0),
                3
            ) AS debt_earnings_ratio
        FROM colleges
        WHERE graduate_debt > 0
          AND earnings_10yr > 0
          AND undergraduate_enrollment >= 1000
        ORDER BY debt_earnings_ratio ASC
        LIMIT 20
    """,

    "above_state_average": """
        WITH state_comparison AS (
            SELECT
                college_name,
                state,
                earnings_10yr,
                AVG(earnings_10yr) OVER (
                    PARTITION BY state
                ) AS state_avg_earnings
            FROM colleges
            WHERE earnings_10yr IS NOT NULL
        )
        SELECT
            college_name,
            state,
            ROUND(earnings_10yr, 0) AS earnings,
            ROUND(state_avg_earnings, 0) AS state_average,
            ROUND(
                earnings_10yr - state_avg_earnings, 0
            ) AS earnings_difference
        FROM state_comparison
        WHERE earnings_10yr > state_avg_earnings
        ORDER BY earnings_difference DESC
        LIMIT 20
    """
}

with sqlite3.connect(db_path) as conn:
    for name, query in queries.items():
        print("\n" + "=" * 60)
        print(name.upper().replace("_", " "))
        print("=" * 60)

        result = pd.read_sql_query(query, conn)
        print(result.to_string(index=False))

        result.to_csv(
            output_dir / f"{name}.csv",
            index=False
        )

print("\nSUCCESS!")
print("SQL reports saved to:", output_dir)
