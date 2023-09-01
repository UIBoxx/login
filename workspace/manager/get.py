from db.userdatabase import users, database, workspaces
from Auth.schemas import User, UserCreate, Token, UserCredentials, UserWorkSpace


async def view_user_workspaces(current_user: User):
    query = workspaces.select().where(workspaces.c.owner_email == current_user.email)
    workspace_records = await database.fetch_all(query)
    return workspace_records