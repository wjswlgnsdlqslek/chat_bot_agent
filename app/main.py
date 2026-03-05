import sys
from contextlib import asynccontextmanager
from pathlib import Path

import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.api.routes import api_router
from app.core.config import settings
from app.ui import create_demo

# ===== 로깅 설정 =====
# loguru를 사용하여 구조화된 로깅 설정
logger.remove()  # 기본 핸들러 제거
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG" if settings.debug else "INFO",
    colorize=True,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    애플리케이션 생명주기 관리

    FastAPI의 lifespan 이벤트 핸들러입니다.
    서버 시작/종료 시 실행되는 로직을 정의합니다.

    시작 시 (yield 이전):
        - 데이터베이스 연결 설정
        - 캐시 연결 설정
        - LangGraph 그래프 컴파일
        - 🆕 LLMOps 2강: 체크포인터 초기화

    종료 시 (yield 이후):
        - 연결 정리
        - 리소스 해제
        - 🆕 LLMOps 2강: 체크포인터 연결 정리

    Args:
        app: FastAPI 애플리케이션 인스턴스
    """
    # ===== 서버 시작 시 실행 =====
    logger.info("=" * 50)
    logger.info("Lumi Agent 서버를 시작합니다...")
    logger.info(f"환경: {settings.environment}")
    logger.info(f"디버그 모드: {settings.debug}")
    logger.info("=" * 50)

    # 설정 검증 (필수 환경변수 확인)
    _validate_settings()

    # LangGraph 그래프 초기화 (워밍업)
    try:
        from app.graph import get_lumi_graph

        _ = get_lumi_graph()  # 워밍업: 그래프 컴파일만 수행
        logger.info("✅ LangGraph 그래프 컴파일 완료")
    except Exception as e:
        logger.error(f"LangGraph 초기화 실패: {e}")

    # 🆕 LLMOps 2강: 체크포인터가 포함된 그래프 워밍업
    if settings.enable_checkpointer:
        try:
            from app.graph.graph import get_lumi_graph_with_memory

            _ = await get_lumi_graph_with_memory()
            logger.info(
                f"✅ 체크포인터 초기화 완료 (타입: {settings.checkpointer_type})"
            )
        except Exception as e:
            logger.error(f"체크포인터 초기화 실패: {e}")
            logger.warning("⚠️ 체크포인터 없이 서버 시작 (대화 이어가기 불가)")

    if settings.enable_langfuse:
        try:
            from app.core.tracing import init_langfuse

            langfuse_client = init_langfuse()
            if langfuse_client:
                logger.info(
                    f"✅ Langfuse 트레이싱 초기화 완료(host: {settings.langfuse_host})"
                )
            else:
                logger.warning("⚠️ Langfuse 트레이싱 초기화 실패")
        except Exception as e:
            logger.error(f"Langfuse 초기화 실패: {e}")
            logger.warning("⚠️ Langfuse 트레이싱 비활성화 (추적 불가)")
    else:
        logger.info("Langfuse 트레이싱 비활성화")

    yield  # 이 지점에서 서버가 요청을 처리함

    if settings.enable_langfuse:
        try:
            from app.core.tracing import flush_langfuse

            flush_langfuse()
            logger.info("✅ Langfuse 트레이싱 데이터 플러시 완료")
        except Exception as e:
            logger.warning(f"Langfuse 트레이싱 데이터 플러시 중 오류: {e}")

    # ===== 서버 종료 시 실행 =====
    logger.info("Lumi Agent 서버를 종료합니다...")

    # 🆕 LLMOps 2강: 체크포인터 연결 정리
    if settings.enable_checkpointer:
        try:
            from app.core.checkpointer import cleanup_checkpointer

            await cleanup_checkpointer()
            logger.info("✅ 체크포인터 연결 정리 완료")
        except Exception as e:
            logger.warning(f"체크포인터 정리 중 오류: {e}")


def _validate_settings():
    """
    필수 설정값 검증

    서버 시작 시 필수 환경변수가 설정되어 있는지 확인합니다.
    설정되지 않은 경우 경고 로그를 출력합니다.
    """
    if not settings.upstage_api_key:
        logger.warning(
            "UPSTAGE_API_KEY가 설정되지 않았습니다. LLM 기능을 사용할 수 없습니다."
        )

    if not settings.supabase_url or not settings.supabase_key:
        logger.warning("Supabase 설정이 완료되지 않았습니다. Mock 데이터를 사용합니다.")

    # LLMOps 2강: PostgreSQL 체크포인터 사용 시 연결 문자열 확인
    if (
        settings.enable_checkpointer
        and settings.checkpointer_type == "postgres"
        and not settings.supabase_connection_string
    ):
        logger.warning(
            "⚠️ SUPABASE_CONNECTION_STRING이 설정되지 않았습니다. "
            "PostgreSQL 체크포인터를 사용할 수 없습니다. "
            "MemorySaver로 대체됩니다."
        )

    # LLMOps 2강: 비용 추적 활성화 시 Supabase 필요
    if settings.enable_cost_tracking and (
        not settings.supabase_url or not settings.supabase_key
    ):
        logger.warning(
            "⚠️ 비용 추적이 활성화되었지만 Supabase가 설정되지 않았습니다. "
            "비용 로그가 저장되지 않습니다."
        )

    # Production 환경에서는 디버그 모드 비활성화 필요
    if settings.environment == "production" and settings.debug:
        logger.warning("⚠️ Production 환경에서 DEBUG 모드가 활성화되어 있습니다!")

    # 🆕 LLMOps 3강: Langfuse 설정 검증
    if settings.enable_langfuse:
        if not settings.langfuse_public_key or not settings.langfuse_secret_key:
            logger.warning(
                "ENABLE_LANGFUSE=true이지만 LANGFUSE_PUBLIC_KEY 또는 LANGFUSE_SECRET_KEY가 설정되지 않았습니다. "
                "Langfuse가 비활성화됩니다."
            )


# ===== FastAPI 애플리케이션 생성 =====
app = FastAPI(
    # 기본 정보
    title="Lumi Agent API",
    description="""
    ## 루미(Lumi) - 버추얼 아이돌 AI 에이전트

    팬들의 덕질을 도와주는 AI 에이전트 서비스입니다.

    ### 주요 기능
    - **대화**: 루미와 자연스러운 대화
    - **정보 제공**: 스케줄, 프로필 조회
    - **액션 수행**: 캘린더 등록, 팬레터 저장

    ### 기술 스택
    - LangGraph: 에이전트 워크플로우
    - Upstage Solar: LLM API
    - FastAPI: 웹 프레임워크
    - Supabase: 데이터베이스

    """,
    version="0.8.0",  # 🆕 LLMOps 2강
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ===== CORS 미들웨어 설정 =====
# Cross-Origin Resource Sharing 설정
# 프론트엔드에서 API를 호출할 수 있도록 허용
app.add_middleware(
    CORSMiddleware,
    # 개발 환경: 모든 origin 허용 (Gradio 포함)
    # Production: 특정 도메인만 허용하도록 변경 필요
    allow_origins=["*"] if settings.debug else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== 라우터 등록 =====
# API 라우터 등록 (/api/v1 prefix)
# 버전 관리를 위해 /api/v1 prefix 사용
app.include_router(api_router, prefix="/api/v1")


# ===== Static 파일 마운트 =====
# favicon, og-image 등 정적 파일 서빙
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ===== Gradio UI 마운트 =====
# /ui 경로에서 Gradio 채팅 인터페이스 제공
gradio_app = create_demo()
app = gr.mount_gradio_app(app, gradio_app, path="/ui")


# ===== 루트 엔드포인트 =====
@app.get("/", tags=["Root"])
async def root():
    """루트 - Gradio UI로 리다이렉트"""
    return RedirectResponse(url="/ui")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """브라우저 기본 favicon 요청 처리"""
    return RedirectResponse(url="/static/favicon.svg")


@app.get("/manifest.json", include_in_schema=False)
async def manifest():
    """PWA manifest (Gradio 자동 요청 처리)"""
    return {
        "name": "Lumi Agent",
        "short_name": "Lumi",
        "start_url": "/ui",
        "display": "standalone",
        "theme_color": "#ff6b9d",
    }


@app.get("/api", tags=["Root"])
async def api_info() -> dict:
    """API 정보"""
    return {
        "message": "Lumi Agent API",
        "docs": "/docs",
        "ui": "/ui",
        "endpoints": {
            "health": "/api/v1/health",
            "chat": "/api/v1/chat",
            "chat_stream": "/api/v1/chat/stream",
        },
    }


# ===== 직접 실행 시 =====
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
