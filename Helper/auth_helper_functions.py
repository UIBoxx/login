from db.userdatabase import users, database, workspaces
from sqlalchemy.sql import select
from fastapi import  Depends, HTTPException, status
from Auth.schemas import User, UserCreate
from jose import jwt, JWTError
from userSecurity import password_context, oauth2_scheme, SECRET_KEY, ALGORITHM
from userSecurity import token_blacklist
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta


def get_db():
    db = database
    try:
        yield db
    finally:
        db.disconnect()

async def get_user(email: str, db=Depends(get_db)):
    query = select([users]).where(users.c.email == email)
    user = await db.fetch_one(query)
    return user



def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    # print(payload)
    # print(token_blacklist)
    email = payload.get("sub")
    token_id = payload.get("jti")
    # print(token_id)
    if token_id in token_blacklist:
        # raise HTTPException(status_code=401, detail="Session Expired.")
        JSONResponse(content={"error": "Session Expired."}, status_code=401)
    return User(username=email, email=email)

def create_access_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_refresh_token(subject: str):
    return create_access_token({"sub": subject}, timedelta(days=30))
