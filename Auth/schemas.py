from pydantic import BaseModel

class UserInDB(BaseModel):
    username: str
    email: str
    hashed_password: str

class User(BaseModel):
    username: str
    email: str


class UserCreate(BaseModel):
    username: str
    email: str
    password: str


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str

class TokenData(BaseModel):
    username: str

class UserCredentials(BaseModel):
    username: str
    password: str

class UserWorkSpace(BaseModel):
    workspace_name: str
    owner_email: str
    model_count: int

class GetUserWorkSpace(BaseModel):
    id: int
    workspace_name: str
    owner_email: str
    timestamp: str
    model_count: int

class UserWorkspaceData(BaseModel):
    workspace_id: int
    Type: str
    database_path: str
    model_path: str
    upscaling_path: str
    model_name: str
    is_trained: bool = False  
    models_count: int = 0

class GetUserWorkspaceData(BaseModel):
    id: int
    workspace_id: int
    Type: str
    database_path: str
