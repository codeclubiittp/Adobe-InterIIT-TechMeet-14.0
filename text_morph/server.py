from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from router import router as sam_router,session_manager
import os


app = FastAPI(title="TextRewrite")

# Add CORS middleware (optional, but useful)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the SAM router
app.include_router(sam_router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse(content={
        "status": "healthy",
        "model_loaded": True,
        "active_sessions": session_manager.get_active_sessions_count()
    })


@app.get("/")
async def root():
    """Root endpoint."""
    return JSONResponse(content={
        "message": "Text Morph",
        "docs": "/docs",
        "version": "1.0.0"
    })


# Note: When running in Docker, uvicorn command is in Dockerfile
# This block is only for local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7000)