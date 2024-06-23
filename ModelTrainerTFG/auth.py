from passlib.handlers.argon2 import argon2
from db import query_db
from argon2 import PasswordHasher
import streamlit as st

ph = PasswordHasher()

def get_password_hash(password):
    return ph.hash(password)
def verify_password(plain_password, hashed_password):
    try:
        return ph.verify(hashed_password, plain_password)
    except argon2.exceptions.VerifyMismatchError:
        return False
def create_user(username, password, email):
    hashed_password = get_password_hash(password)
    rows_affected = query_db(
        "INSERT INTO users (username, password, email) VALUES (%s, %s, %s)",
        (username, hashed_password, email),
        is_select=False
    )
def authenticate_user(username, password):
    query = "SELECT * FROM users WHERE username = %s"
    user = query_db(query, (username,), one=True)
    if user and verify_password(password, user['password']):
        return user
    else:
        st.error("These credentials do not exist, sign up first if you do not have an account.")
    return None
def get_user_profile(username):
    query = "SELECT id, username, email FROM users WHERE username = %s"
    return query_db(query, (username,), one=True)
