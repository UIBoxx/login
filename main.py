from db.userdatabase import database
from Routers.auth_router import auth_router
from Routers.workspace_router import workspace_router
import uvicorn
from common_imports import *


app = FastAPI()

app.include_router(auth_router)
app.include_router(workspace_router, prefix="/api")

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
def shutdown_db():
    database.disconnect()

if __name__ == "__main__":
    uvicorn.run(app, host="192.168.254.3", port=8000)
