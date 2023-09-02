from common_imports import *

async def user_logout(token: str):
    try:
        decoded_token = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        jti = decoded_token.get("jti")
        token_blacklist.add(jti)
        return {"message": "Logout successful"}
    except jwt.ExpiredSignatureError:
        JSONResponse(content={"error": "Expired token"}, status_code=401)
        # raise HTTPException(status_code=401, detail="Expired token")
    except jwt.DecodeError:
        JSONResponse(content={"error": "Invalid token"}, status_code=401)
        # raise HTTPException(status_code=401, detail="Invalid token")