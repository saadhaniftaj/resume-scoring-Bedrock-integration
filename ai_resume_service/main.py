"""
TGS Management AI - Main Application Entry Point

FastAPI application for resume-to-job matching with AI-powered analysis.
"""

from fastapi import FastAPI
from app.routes import analyze_routes
from app.processors.local_parser import ResumeParser
from app.config.app_config import ServerConfig, print_config_summary
from app.processors.model_factory import preload_models

# Initialize FastAPI app
app = FastAPI(
    title="TGS Management AI",
    description="AI-powered resume matching and analysis for contractor hiring",
    version="1.0.0"
)

# Initialize parsers
parser = ResumeParser()

# Include routers
app.include_router(analyze_routes.router)

@app.on_event("startup")
async def startup_event():
    """
    Application startup event.
    Preloads models to avoid lazy loading delays on first request.
    """
    from app.utils.logger import get_logger
    logger = get_logger(__name__)
    
    logger.info("="*60)
    logger.info("TGS Management AI - Starting Up")
    logger.info("="*60)
    print_config_summary()  # Still print to console for visibility
    logger.info("Preloading ML models...")
    preload_models()
    logger.info("Application ready to serve requests!")
    logger.info("="*60)

@app.post("/parse-resume")
async def parse_resume(text: str):
    """Parse resume text and extract structured information."""
    return parser.parse(text)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "TGS Management AI"}

if __name__ == "__main__":
    import uvicorn
    
    # Print configuration summary
    print_config_summary()
    
    # Run server with configured host and port
    uvicorn.run(
        app,
        host=ServerConfig.HOST,
        port=ServerConfig.PORT,
        log_level=ServerConfig.LOG_LEVEL.lower()
    )