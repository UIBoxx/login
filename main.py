from db.userdatabase import database
from Routers.auth_router import auth_router
from Routers.workspace_router import workspace_router
from Routers.default_workspace_router import Default_workspace_router
import uvicorn
from common_imports import *
from fastapi.staticfiles import StaticFiles


app = FastAPI()

app.mount("/workspace", StaticFiles(directory="workspace"), name="workspace")

app.include_router(auth_router)
app.include_router(workspace_router)
app.include_router(Default_workspace_router)

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
    uvicorn.run(app, host="192.168.1.147", port=PORT)
