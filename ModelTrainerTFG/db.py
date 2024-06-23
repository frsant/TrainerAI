import psycopg2
from psycopg2.extras import RealDictCursor, Json
import logging


def get_db_connection():
    conn = psycopg2.connect(
        dbname="ModelTrainerDB",
        user="postgres",
        password="admin123",
        host="localhost",
        port="5432"
    )
    return conn
def query_db(query, params=None, is_select=True, one=False):
    dsn = "dbname='ModelTrainerDB' user='postgres' host='localhost' password='admin123' port='5432'"
    with psycopg2.connect(dsn) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(query, params)
            if is_select:
                if one:
                    return cur.fetchone()
                else:
                    return cur.fetchall()
            else:
                conn.commit()
                return cur.rowcount
def save_model(user_id, model_type, model_name, model, evaluation, test_data=None, regression_type=None, input=None, output=None):
    query = """
    INSERT INTO models (user_id, model_type, model_name, model, evaluation, test_data, regression_type, input, output) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    params = (user_id, model_type, model_name, model, evaluation, test_data, regression_type, input, output)
    return query_db(query, params, is_select=False)
def get_user_models(user_id):
    query = "SELECT * FROM models WHERE user_id = %s ORDER BY created_at DESC"
    return query_db(query, (user_id,))
