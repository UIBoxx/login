from fastapi import FastAPI, Depends, HTTPException, Request, APIRouter, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from Auth.schemas import User, UserCreate, Token, UserCredentials, UserWorkSpace, GetUserWorkSpace,UserWorkspaceData, GetUserWorkspaceData
from db.userdatabase import users, database, workspaces, workspaceData
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import delete, update, select, desc
from databases import Database
from datetime import timedelta
from fastapi.responses import JSONResponse
from Helper.auth_helper_functions import (
    get_user,
    get_db,
    create_access_token,
    create_refresh_token,
    token_blacklist,
    get_current_user
)
from userSecurity import (
    password_context,
    ACCESS_TOKEN_EXPIRE_DAYS,
    ALGORITHM,
    SECRET_KEY,
    oauth2_scheme,
)
from pydantic import BaseModel
import uuid
import os
from jose import jwt

from Auth.login import login_for_access_token, login_for_access_with_form
from Auth.logout import user_logout
from Auth.register import user_signup


from workspace.manager.create import user_workspace
from workspace.manager.get import view_user_workspaces
from workspace.manager.delete import delete_user_workspace
from workspace.manager.update import update_user_workspace