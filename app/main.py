import sys
from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from loguru import logger

from app.api.routes import api_router
from app.core.config import settings
from app.ui import create_demo

# 1. 로거 설정
logger.remove()
logger.add(
    sys.stdout,
    # 로깅 메시지의 형식을 지정합니다. 시간, 로그 레벨, 모듈 이름, 함수 이름, 라인 번호, 그리고 실제 메시지를 포함하도록 설정되어 있습니다.
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG" if settings.debug else "INFO",
    colorize=True,
)


# 5. @asynccontextmanager 데코레이터
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 4. lifespan 이벤트 핸들러를 정의합니다. 이 함수는 서버가 시작될 때와 종료될 때 실행됩니다. 서버가 시작될 때 설정을 검증하고, 종료될 때 로그 메시지를 출력합니다.
    logger.info("=" * 50)
    logger.info("Lumi Agent 서버를 시작합니다...")
    logger.info(f"환경: {settings.environment}")
    logger.info(f"디버그 모드: {settings.debug}")
    logger.info("=" * 50)

    _validate_settings()

    yield  # 이 지점에서 서버가 요청을 처리함

    logger.info("Lumi Agent 서버를 종료합니다...")


# 6. 설정 검증 함수
def _validate_settings():
    """설정 검증 함수입니다. 필요한 설정이 누락되었는지 확인하고, 누락된 경우 예외를 발생시킵니다."""
    if not settings.upstage_api_key:
        logger.warning(
            "UPSTAGE_API_KEY가 설정되지 않았습니다. LLM 기능을 사용할 수 없습니다."
        )

    if settings.environment == "production" and settings.debug:
        logger.warning("Production 환경에서 DEBUG 모드가 활성화되어 있습니다!")


# 2. FastAPI 애플리케이션 생성
app = FastAPI(
    title=settings.project_name,
    description="A practical FastAPI project",
    version="0.1.0",
    # 3. lifespan 이벤트 핸들러를 등록합니다. 애플리케이션이 시작될 때와 종료될 때 실행되는 비동기 함수입니다.
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# 7. CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")


# 8. endpoint 생성
@app.get("/", tags=["Root"])
async def root() -> RedirectResponse:
    return RedirectResponse(url="/ui")


# 8.1 헬스 체크 및 서버 정보 endpoint 생성
@app.get("/health", tags=["System"])
async def health_check() -> dict:
    return {
        "status": "ok",
        "environment": settings.environment,
    }


@app.get("/info", tags=["System"])
async def server_info() -> dict:
    return {
        "project_name": settings.project_name,
        "environment": settings.environment,
        "debug": settings.debug,
        "host": settings.host,
        "port": settings.port,
        "version": app.version,
    }


app = gr.mount_gradio_app(app, create_demo(), path="/ui")


# 9. 실행 함수
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
