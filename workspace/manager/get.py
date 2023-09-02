# from db.userdatabase import users, database, workspaces
# from Auth.schemas import User, UserCreate, Token, UserCredentials, UserWorkSpace

from common_imports import *



async def view_user_workspaces(current_user: User):
    async with database.transaction():
        query = workspaces.select().where(workspaces.c.owner_email == current_user.email)
        workspace_records = await database.fetch_all(query)
        return workspace_records
    