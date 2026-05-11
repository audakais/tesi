import psycopg2

conn = psycopg2.connect(host="localhost", database="cancerdata", user="postgres", password="password")
cur = conn.cursor()

query = """
SELECT
    p.cancer_type,
    SUM(CASE WHEN st.type_name ILIKE '%Normal%' THEN 1 ELSE 0 END) as normal_count,
    SUM(CASE WHEN st.type_name NOT ILIKE '%Normal%' THEN 1 ELSE 0 END) as tumor_count,
    COUNT(*) as total
FROM Patients p
JOIN Samples s ON p.individual_id = s.individual_id
JOIN SampleTypes st ON s.sample_type_id = st.type_id
GROUP BY p.cancer_type
ORDER BY total DESC;
"""

cur.execute(query)
for row in cur.fetchall():
    print(f"Type: {row[0][:30]:<30} | Normal: {row[1]:<5} | Tumor: {row[2]:<5} | Total: {row[3]}")

cur.close()
conn.close()
