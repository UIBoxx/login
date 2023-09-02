from common_imports import *

workspace_router = APIRouter()

@workspace_router.post("/workspaces/new", response_model=UserWorkSpace)
async def create_workspace(workspace: UserWorkSpace, current_user: User = Depends(get_current_user)):
    return await user_workspace(workspace, current_user)

@workspace_router.get("/workspaces/new", response_model=list[GetUserWorkSpace])
async def view_workspaces(current_user: User = Depends(get_current_user)):
    return await view_user_workspaces(current_user)

@workspace_router.delete("/workspaces/new/{workspace_id}", response_model=dict)
async def delete_workspace(workspace_id: int, current_user: User = Depends(get_current_user)):
    return await delete_user_workspace(workspace_id, current_user)

@workspace_router.put("/workspaces/new/{workspace_id}", response_model=UserWorkSpace)
async def update_workspace(workspace_id: int, updated_workspace: UserWorkSpace, current_user: User = Depends(get_current_user)):
    return await update_user_workspace(workspace_id, updated_workspace, current_user)
