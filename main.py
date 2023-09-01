from fastapi import FastAPI, Depends, File, UploadFile
from fastapi.security import OAuth2PasswordRequestForm
from typing import List
from Auth.schemas import User, UserCreate, Token, UserWorkSpace
from db.userdatabase import database
from databases import Database
from Helper.auth_helper_functions import get_current_user, get_db, create_refresh_token
from userSecurity import oauth2_scheme
from default.industries.model1.libraries.prediction import Prediction
from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import JSONResponse
import os



from Auth.login import login_for_access_with_form
from Auth.logout import user_logout
from Auth.register import user_signup
from workspace.manager.create import user_workspace
from workspace.manager.get import view_user_workspaces
from workspace.manager.delete import delete_user_workspace
from workspace.manager.update import update_user_workspace


app = FastAPI()
origins = ["*"]

 

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


    
@app.on_event("startup")
async def startup_db():
    await database.connect()

@app.on_event("shutdown")
async def shutdown_db():
    await database.disconnect()

# Routes
@app.post("/signup/", response_model=User)
async def signup(user: UserCreate, db: Database = Depends(get_db)):
    return await user_signup(user, db)

@app.post("/token/", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Database = Depends(get_db)):
    return await login_for_access_with_form(form_data,db)

@app.get("/protected/")
async def protected_route(current_user: User = Depends(get_current_user)):
    return {"message": "You are authorized to access this route.",
            "user":current_user}

@app.get("/refresh/")
async def refresh_token(current_user: User = Depends(get_current_user)):
    refresh_token = create_refresh_token(current_user.username)
    return {"refresh_token": refresh_token}

@app.post("/logout/")
async def logout(token: str = Depends(oauth2_scheme)):
    return await user_logout(token)

@app.post("/workspace/industries/")
def industries_pred(steam_data: List[float]):
    prediction = Prediction(steam_data = steam_data)
    make_prediction = prediction.make_prediction()
    return{"result":make_prediction}

@app.post("/workspaces/new", response_model=UserWorkSpace)
async def create_workspace(workspace: UserWorkSpace, current_user: User = Depends(get_current_user)):
    return await user_workspace(workspace, current_user)

@app.get("/workspaces/new", response_model=List[UserWorkSpace])
async def view_workspaces(current_user: User = Depends(get_current_user)):
    return await view_user_workspaces(current_user)

@app.delete("/workspaces/new/{workspace_id}", response_model=dict)
async def delete_workspace(workspace_id: int, current_user: User = Depends(get_current_user)):
    return await delete_user_workspace(workspace_id, current_user)

@app.put("/workspaces/new/{workspace_id}", response_model=UserWorkSpace)
async def update_workspace(workspace_id: int, updated_workspace: UserWorkSpace, current_user: User = Depends(get_current_user)):
    return await update_user_workspace(workspace_id,updated_workspace,current_user)


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    try:
        user_dir = os.path.join("workspace/users/databases", current_user.username)
        os.makedirs(user_dir, exist_ok=True)
        
        file_path = os.path.join(user_dir, "data.csv")
        
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        
        return JSONResponse(content={"message": "File uploaded successfully"}, status_code=201)
    except Exception as e:
        return JSONResponse(content={"message": "An error occurred", "error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
