from fastapi import FastAPI
from api.workspace import router as workspace_router
from api.feedback import router as feedback_router

app = FastAPI()
app.include_router(workspace_router)
app.include_router(feedback_router)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()