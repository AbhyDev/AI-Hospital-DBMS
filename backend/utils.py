from bcrypt._bcrypt import hashpw, gensalt, checkpw

def hash(password: str):
    return hashpw(password.encode('utf-8'), gensalt()).decode('utf-8')

def verify(plain_password, hashed_password):
    return checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))