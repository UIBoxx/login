from common_imports import *

auth_router = APIRouter()

@auth_router.post("/signup/", response_model=User)
async def signup(user: UserCreate, db: Database = Depends(get_db)):
    return await user_signup(user, db)

# @auth_router.post("/token/", response_model=Token)
# async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Database = Depends(get_db)):
#     return await login_for_access_with_form(form_data, db)

@auth_router.post("/token/", response_model=Token)
async def login(user_credentials: UserCredentials, db: Database = Depends(get_db)):
    return await login_for_access_token(user_credentials, db)

@auth_router.post("/logout/")
async def logout(token: str = Depends(oauth2_scheme)):
    return await user_logout(token)

@auth_router.get("/protected/")
async def protected_route(current_user: User = Depends(get_current_user)):
    return {"message": "You are authorized to access this route.", "user": current_user}

@auth_router.get("/refresh/")
async def refresh_token(current_user: User = Depends(get_current_user)):
    refresh_token = create_refresh_token(current_user.username)
    return {"refresh_token": refresh_token}
