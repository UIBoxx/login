from Auth.schemas import UserCreate
from databases import Database
from fastapi import Depends, HTTPException
from Helper.auth_helper_functions import get_user, get_db
from userSecurity import password_context
from db.userdatabase import users



async def create_user(user: UserCreate, db=Depends(get_db)):
    hashed_password = password_context.hash(user.password)
    query = users.insert().values(username=user.username, email=user.email, hashed_password=hashed_password)
    await db.execute(query)


async def user_signup(user: UserCreate, db: Database):
    existing_user = await get_user(user.email, db)
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    await create_user(user, db)
    return user