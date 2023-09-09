from fastapi import UploadFile, Depends, HTTPException
import os
from fastapi.responses import JSONResponse



def get_next_filename(user_dir: str, base_filename: str) -> str:
    index = 0
    while True:
        filename = f"{base_filename}_{index}.csv"
        file_path = os.path.join(user_dir, filename)
        if not os.path.exists(file_path):
            return filename
        index += 1

def save_uploaded_file(file: UploadFile, user_dir: str, filename: str) -> None:
    try:
        os.makedirs(user_dir, exist_ok=True)
        
        file_path = os.path.join(user_dir, filename)
        
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        
    except Exception as e:
        JSONResponse(content={"error": "An error occurred while saving the file"}, status_code=500)



# ... (other imports and code)
