from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta
from typing import List
from models.model import User, UserCreate, Token, UserCredentials, UserWorkSpace
from db.userdatabase import users, database, workspaces
from databases import Database
from userHelper.userHelperFunction import get_user, create_user, get_current_user, get_db, token_blacklist
from userSecurity import password_context, ACCESS_TOKEN_EXPIRE_MINUTES, create_access_token, create_refresh_token, ALGORITHM, SECRET_KEY, oauth2_scheme
import uuid
from jose import jwt
import os
from default.industries.model1.libraries.prediction import Prediction
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import delete, update


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
    existing_user = await get_user(user.email, db)
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    await create_user(user, db)
    return user


# @app.post("/token/", response_model=Token)
# async def login_for_access_login(form_data: OAuth2PasswordRequestForm = Depends(), db: Database = Depends(get_db)):
#     user = await get_user(form_data.username, db)
#     if not user or not password_context.verify(form_data.password, user['hashed_password']):
#         raise HTTPException(status_code=400, detail="Incorrect email or password")
    
#     access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
#     jti = str(uuid.uuid1())
#     print(jti)
#     access_token = create_access_token(data={"sub": user["email"], "jti":jti}, expires_delta=access_token_expires)
#     refresh_token = create_refresh_token(user["email"])

    # Create a directory with the user's email inside the "workspace" folder
    # user_workspace_path = os.path.join("workspace", user["email"])
    # os.makedirs(user_workspace_path, exist_ok=True)

    # # Create "lib" and "models" directories inside the user's email directory
    # lib_path = os.path.join(user_workspace_path, "lib")
#     models_path = os.path.join(user_workspace_path, "models")
#     os.makedirs(lib_path, exist_ok=True)
#     os.makedirs(models_path, exist_ok=True)

    # return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}

@app.post("/token/", response_model=Token)
async def login_for_access_token(user_credentials: UserCredentials, db: Database = Depends(get_db)):
    user = await get_user(user_credentials.username, db)
    if not user or not password_context.verify(user_credentials.password, user['hashed_password']):
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    jti = str(uuid.uuid1())
    access_token = create_access_token(data={"sub": user["email"], "jti": jti}, expires_delta=access_token_expires)
    refresh_token = create_refresh_token(user["email"])    

    return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}



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
    try:
        decoded_token = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        jti = decoded_token.get("jti")
        token_blacklist.add(jti)
        return {"message": "Logout successful"}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Expired token")
    except jwt.DecodeError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/workspace/industries/")
def industries_pred(steam_data: List[float]):
    prediction = Prediction(steam_data = steam_data)
    make_prediction = prediction.make_prediction()
    return{"result":make_prediction}

@app.post("/workspaces/new", response_model=UserWorkSpace)
async def create_workspace(workspace: UserWorkSpace, current_user: User = Depends(get_current_user)):
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


@app.get("/workspaces/new", response_model=List[UserWorkSpace])
async def view_workspaces(current_user: User = Depends(get_current_user)):
    query = workspaces.select().where(workspaces.c.owner_email == current_user.email)
    workspace_records = await database.fetch_all(query)
    return workspace_records

@app.delete("/workspaces/new/{workspace_id}", response_model=dict)
async def delete_workspace(workspace_id: int, current_user: User = Depends(get_current_user)):
    # Check if the workspace with the given ID belongs to the current user
    query = workspaces.select().where((workspaces.c.id == workspace_id) & (workspaces.c.owner_email == current_user.email))
    workspace_record = await database.fetch_one(query)

    if not workspace_record:
        raise HTTPException(status_code=404, detail="Workspace not found")

    # Delete the workspace
    delete_query = delete(workspaces).where(workspaces.c.id == workspace_id)
    await database.execute(delete_query)

    return {"message": "Workspace deleted successfully"}

@app.put("/workspaces/new/{workspace_id}", response_model=UserWorkSpace)
async def update_workspace(workspace_id: int, updated_workspace: UserWorkSpace, current_user: User = Depends(get_current_user)):
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



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.1.147", port=8000)
