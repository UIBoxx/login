from common_imports import *

async def create_user(user: UserCreate, db=Depends(get_db)):
    hashed_password = password_context.hash(user.password)
    query = users.insert().values(username=user.username, email=user.email, hashed_password=hashed_password)
    await db.execute(query)


async def user_signup(user: UserCreate, db: Database):
    existing_user = await get_user(user.email, db)
    if existing_user:
        JSONResponse(content={"error": "Email already registered"}, status_code=400)
    await create_user(user, db)
    return user