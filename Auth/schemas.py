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
    timestamp: str
    model_count: int
