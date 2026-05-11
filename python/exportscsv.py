import psycopg2
import pandas as pd

conf = {
    "host": "localhost",
    "database": "cancerdata",
    "user": "postgres",
    "password": "password"
}

output_file = "/mnt/c/users/utente/desktop/tesi_clean/python/dataset_totale.csv"

def export_all():
    conn = psycopg2.connect(**conf)

    sql = """
    SELECT
        p.cancer_type,
        s.sample_id,
        b.biomarker_name,
        m.measurement_value,
        CASE WHEN st.type_name ILIKE '%Normal%' THEN 0 ELSE 1 END as target
    FROM Patients p
    JOIN Samples s ON p.individual_id = s.individual_id
    JOIN SampleTypes st ON s.sample_type_id = st.type_id
    JOIN Measurements m ON s.sample_id = m.sample_id
    JOIN Biomarkers b ON m.biomarker_id = b.biomarker_id
    WHERE b.biomarker_id IN (
        SELECT biomarker_id FROM Biomarkers LIMIT 1000
    );
    """

    print("Starting full export... this may take a while.")
    df = pd.read_sql_query(sql, conn)
    df.to_csv(output_file, index=False)
    print(f"Export complete. {len(df)} records saved to {output_file}")
    conn.close()

if __name__ == "__main__":
    export_all()

