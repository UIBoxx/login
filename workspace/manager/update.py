from common_imports import *

async def update_user_workspace(workspace_id: int, updated_workspace: UserWorkSpace, current_user: User):
    # Check if the workspace with the given ID belongs to the current user
    query = workspaces.select().where((workspaces.c.id == workspace_id) & (workspaces.c.owner_email == current_user.email))
    workspace_record = await database.fetch_one(query)

    if not workspace_record:
        raise HTTPException(status_code=404, detail="Workspace not found")

    # Update the workspace
    update_values = {
        "workspace_name": updated_workspace.workspace_name,
        # "model_count": updated_workspace.model_count,
        # "timestamp": updated_workspace.timestamp,
    }   
    update_query = update(workspaces).where(workspaces.c.id == workspace_id).values(**update_values)
    await database.execute(update_query)

    # Fetch and return the updated workspace
    updated_workspace_query = workspaces.select().where(workspaces.c.id == workspace_id)
    updated_workspace_record = await database.fetch_one(updated_workspace_query)
    return updated_workspace_record