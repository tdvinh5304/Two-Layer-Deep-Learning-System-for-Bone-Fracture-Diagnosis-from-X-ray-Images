import mysql.connector
import pandas as pd

MYSQL_HOST = "bf2y1sghovmjlpuyvhxi-mysql.services.clever-cloud.com"
MYSQL_DB   = "bf2y1sghovmjlpuyvhxi"
MYSQL_USER = "un87fm6fqztdwuck"
MYSQL_PORT = 3306
MYSQL_PASS = "QFpD3gRbkukuFak11A67"

conn = mysql.connector.connect(
    host=MYSQL_HOST,
    user=MYSQL_USER,
    password=MYSQL_PASS,
    database=MYSQL_DB,
    port=MYSQL_PORT
)
cursor = conn.cursor()

print("✅ Connected to MySQL!")


cursor.execute("""
CREATE TABLE IF NOT EXISTS mura_images (
    id INT AUTO_INCREMENT PRIMARY KEY,
    image_path VARCHAR(255),
    study_path VARCHAR(255),
    label INT,
    dataset VARCHAR(10)
)
""")

print("📌 Table ready.")


def upload_csv(csv_path, dataset_type):
    df = pd.read_csv(csv_path)

    rows = []
    for _, row in df.iterrows():
        rows.append((
            row["image_path"],
            row["study_path"],
            int(row["label"]) if not pd.isna(row["label"]) else None,
            dataset_type
        ))

    sql = """
    INSERT INTO mura_images (image_path, study_path, label, dataset)
    VALUES (%s, %s, %s, %s)
    """

    cursor.executemany(sql, rows)
    conn.commit()

    print(f"✅ Uploaded {len(rows)} rows → {dataset_type}")



upload_csv("MURA-v1.1/train_merged.csv", "train")
upload_csv("MURA-v1.1/valid_merged.csv", "valid")


cursor.close()
conn.close()
print("🎉 DONE! All data uploaded successfully.")
