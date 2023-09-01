from jose import jwt
from userSecurity import password_context, ACCESS_TOKEN_EXPIRE_MINUTES, ALGORITHM, SECRET_KEY, oauth2_scheme
from fastapi import Depends, HTTPException
from Helper.auth_helper_functions import get_user, token_blacklist


async def user_logout(token: str):
    try:
        decoded_token = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        jti = decoded_token.get("jti")
        token_blacklist.add(jti)
        return {"message": "Logout successful"}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Expired token")
    except jwt.DecodeError:
        raise HTTPException(status_code=401, detail="Invalid token")