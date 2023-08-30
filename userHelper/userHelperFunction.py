from db.userdatabase import users, database, workspaces
from sqlalchemy.sql import select
from fastapi import  Depends, HTTPException, status
from models.model import User, UserCreate
from jose import jwt, JWTError
from userSecurity import password_context, oauth2_scheme, SECRET_KEY, ALGORITHM
from userSecurity import token_blacklist

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

async def create_user(user: UserCreate, db=Depends(get_db)):
    hashed_password = password_context.hash(user.password)
    query = users.insert().values(username=user.username, email=user.email, hashed_password=hashed_password)
    await db.execute(query)

def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    print(payload)
    print(token_blacklist)
    email = payload.get("sub")
    token_id = payload.get("jti")
    print(token_id)
    if token_id in token_blacklist:
        raise HTTPException(status_code=401, detail="Session Expired.")
    return User(username=email, email=email)
