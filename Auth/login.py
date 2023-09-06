from common_imports import *

async def login_for_access_token(user_credentials: UserCredentials, db: Database = Depends(get_db)):
    access_token = None  # Default value
    refresh_token = None  # Default value

    user = await get_user(user_credentials.username, db)
    if not user or not password_context.verify(user_credentials.password, user['hashed_password']):
        return JSONResponse(content={"error": "Incorrect email or password"}, status_code=400)
    
    access_token_expires = timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    jti = str(uuid.uuid1())
    if user is not None:
        access_token = create_access_token(data={"sub": user["email"], "name": user["username"], "jti": jti}, expires_delta=access_token_expires)
        refresh_token = create_refresh_token(user["email"])    
    else:
        return JSONResponse(content={"error": "Something went wrong."}, status_code=400)

    return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}


# async def login_for_access_with_form(form_data: OAuth2PasswordRequestForm , db: Database):
#     user = await get_user(form_data.username, db)
#     if not user or not password_context.verify(form_data.password, user['hashed_password']):
#         JSONResponse(content={"error": "Incorrect email or password"}, status_code=400)
    
#     access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_DAYS)
#     jti = str(uuid.uuid1())
#     print(jti)
#     access_token = create_access_token(data={"sub": user["email"], "jti":jti}, expires_delta=access_token_expires)
#     refresh_token = create_refresh_token(user["email"])

#     return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}

