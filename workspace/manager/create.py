from Auth.schemas import User, UserCreate, Token, UserCredentials, UserWorkSpace
from db.userdatabase import users, database, workspaces
from fastapi import FastAPI, Depends, HTTPException

async def user_workspace(workspace: UserWorkSpace, current_user: User):
    query = workspaces.insert().values(
        workspace_name=workspace.workspace_name,
        owner_email=workspace.owner_email,
        model_count=workspace.model_count,
        timestamp=workspace.timestamp,
    )
    workspace_id = await database.execute(query)
    # Create a query to retrieve the inserted workspace
    select_query = workspaces.select().where(workspaces.c.id == workspace_id)
    workspace_record = await database.fetch_one(select_query)
    return workspace_record