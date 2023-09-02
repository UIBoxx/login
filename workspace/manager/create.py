# from Auth.schemas import User, UserCreate, Token, UserCredentials, UserWorkSpace
# from db.userdatabase import users, database, workspaces
# from fastapi import FastAPI, Depends, HTTPException
# import os
from common_imports import *


async def user_workspace(workspace: UserWorkSpace, current_user: User):
    query = workspaces.insert().values(
        workspace_name=workspace.workspace_name,
        owner_email=workspace.owner_email,
        model_count=workspace.model_count,
    )
    workspaceid = await database.execute(query)

    # Create a query to retrieve the inserted workspace
    select_query = workspaces.select().where(workspaces.c.id == workspaceid)
    workspace_record = await database.fetch_one(select_query)

    workspace_path = os.path.join(f"workspace/users/{current_user.email}", workspace_record.workspace_name)
    os.makedirs(workspace_path, exist_ok=True)
    return workspace_record
