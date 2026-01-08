import psycopg2

def get_connection():
    return psycopg2.connect(
        host="localhost",
        port=5433,
        database="face_db",
        user="face_user",
        password="face_pass"
    )
