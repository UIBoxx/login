# from Auth.schemas import User, UserCreate, Token, UserCredentials, UserWorkSpace
# from db.userdatabase import users, database, workspaces
# from fastapi import FastAPI, Depends, HTTPException
# from sqlalchemy import delete, update

from common_imports import *


async def delete_user_workspace(workspace_id: int, current_user: User):
    # Check if the workspace with the given ID belongs to the current user
    query = workspaces.select().where((workspaces.c.id == workspace_id) & (workspaces.c.owner_email == current_user.email))
    workspace_record = await database.fetch_one(query)

    if not workspace_record:
        raise HTTPException(status_code=404, detail="Workspace not found")

    # Delete the workspace
    delete_query = delete(workspaces).where(workspaces.c.id == workspace_id)
    await database.execute(delete_query)

    return {"message": "Workspace deleted successfully"}