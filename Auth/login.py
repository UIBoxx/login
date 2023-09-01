from Auth.schemas import UserCredentials
from databases import Database
from datetime import timedelta
from fastapi import Depends, HTTPException
from Helper.auth_helper_functions import get_user, get_db, create_access_token, create_refresh_token
from userSecurity import password_context, ACCESS_TOKEN_EXPIRE_MINUTES, ALGORITHM, SECRET_KEY, oauth2_scheme
import uuid
from fastapi.security import OAuth2PasswordRequestForm



async def login_for_access_token(user_credentials: UserCredentials, db: Database = Depends(get_db)):
    user = await get_user(user_credentials.username, db)
    if not user or not password_context.verify(user_credentials.password, user['hashed_password']):
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    jti = str(uuid.uuid1())
    access_token = create_access_token(data={"sub": user["email"], "jti": jti}, expires_delta=access_token_expires)
    refresh_token = create_refresh_token(user["email"])    

    return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}



async def login_for_access_with_form(form_data: OAuth2PasswordRequestForm , db: Database):
    user = await get_user(form_data.username, db)
    if not user or not password_context.verify(form_data.password, user['hashed_password']):
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    jti = str(uuid.uuid1())
    print(jti)
    access_token = create_access_token(data={"sub": user["email"], "jti":jti}, expires_delta=access_token_expires)
    refresh_token = create_refresh_token(user["email"])

    # Create a directory with the user's email inside the "workspace" folder
    # user_workspace_path = os.path.join("workspace", user["email"])
    # os.makedirs(user_workspace_path, exist_ok=True)

    # # Create "lib" and "models" directories inside the user's email directory
    # lib_path = os.path.join(user_workspace_path, "lib")
    # models_path = os.path.join(user_workspace_path, "models")
    # os.makedirs(lib_path, exist_ok=True)
    # os.makedirs(models_path, exist_ok=True)

    return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}