from common_imports import *
import csv

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


@workspace_router.post("/{wname}/{type}/upload", response_model=UserWorkspaceData)
async def add_workspace_data(
    wname: str,
    type: str,
    file: UploadFile=File(...),
    current_user: User = Depends(get_current_user)
):
    # Check if the workspace exists and get its ID
    wid_query = select([workspaces.c.id]).where(
        (workspaces.c.owner_email == current_user.email) & (workspaces.c.workspace_name == wname)
    )
    wid = await database.fetch_val(wid_query)

    if wid is None:
        raise HTTPException(status_code=404, detail="No matching workspace was found")

    wid = int(wid)

    user_dir = os.path.join(f"workspace/user/{current_user.email}/{wname}/{type}", "data")
    os.makedirs(user_dir, exist_ok=True)

    file_path = os.path.join(user_dir, file.filename)

    with open(file_path, "wb") as f:
        f.write(file.file.read())

    try:
        # Insert workspace data into the database
        query = workspaceData.insert().values(
            workspace_id=wid,
            Type=type,
            database_path=file_path,
            model_path="",
            upscaling_path="",
            model_name="",
            is_trained= False,
            models_count=0,
        )

        workspacedataid = await database.execute(query)

        # Create a new UserWorkspaceData instance with the inserted ID
        result = UserWorkspaceData(
            workspace_id=wid,
            Type=type,
            database_path=file_path,
            model_path="",
            upscaling_path="",
            model_name="",
            is_trained=False,
            models_count=0,
        )
        return result

    except Exception as e:
        # Log the error for debugging purposes
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")



@workspace_router.get("/{wname}/{type}/data/", response_model=list[GetUserWorkspaceData])
async def get_workspace_data(wname: str, type: str, current_user: User = Depends(get_current_user)):
    wid_query = select([workspaces.c.id]).where(
        (workspaces.c.owner_email == current_user.email) & (workspaces.c.workspace_name == wname)
    )
    wid = await database.fetch_val(wid_query)

    if wid is None:
        raise HTTPException(status_code=404, detail="No matching workspace was found")

    wid = int(wid)

    query = (
            workspaceData.select()
            .where((workspaceData.c.workspace_id == wid) & (workspaceData.c.Type == type))
            .order_by(desc(workspaceData.c.id)) 
            .limit(1)  
        )
    
    workspace_data = await database.fetch_all(query)
    csv_file_path = workspace_data[0]["database_path"]
    print(csv_file_path)

    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        json_data = [row for row in csv_reader]

        print(json_data)


    if not workspace_data:
        raise HTTPException(status_code=404, detail="No data found for this workspace and type")

    return workspace_data
