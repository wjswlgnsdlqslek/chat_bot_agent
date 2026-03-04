"""
LangGraph 체크포인터(Checkpointer) 관리 모듈

체크포인터는 LangGraph 그래프의 상태를 영구 저장합니다.
이를 통해 대화를 이어가거나, 서버 재시작 후에도 상태를 복원할 수 있습니다.

핵심 개념:
    - thread_id: 대화 세션을 구분하는 고유 식별자
    - checkpoint: 특정 시점의 그래프 상태 스냅샷
    - checkpointer: 체크포인트를 저장/복원하는 인터페이스

지원하는 체크포인터 타입:
    1. MemorySaver (기본값)
       - 인메모리 저장, 서버 재시작 시 초기화
       - 개발/테스트 용도

    2. AsyncPostgresSaver (Production용)
       - PostgreSQL 데이터베이스에 저장
       - Supabase PostgreSQL 연동
       - 다중 서버 환경에 적합

사용법:
    from app.core.checkpointer import get_checkpointer

    # 설정에 따라 적절한 체크포인터 반환
    checkpointer = await get_checkpointer()

    # LangGraph 그래프 컴파일 시 사용
    graph = builder.compile(checkpointer=checkpointer)

아키텍처:
    ┌─────────────────────────────────────────────────────────────┐
    │                    LangGraph Graph                          │
    │  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐    │
    │  │   State     │ <-> │ Checkpointer│ <-> │  Database   │    │
    │  │  (Memory)   │     │  (Adapter)  │     │ (Postgres)  │    │
    │  └─────────────┘     └─────────────┘     └─────────────┘    │
    └─────────────────────────────────────────────────────────────┘

    thread_id="user-123" 으로 호출하면:
    1. 해당 thread의 이전 상태를 DB에서 로드
    2. 새 메시지와 함께 그래프 실행
    3. 실행 결과를 DB에 저장 (체크포인트)
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from loguru import logger

from app.core.config import settings

# 전역 체크포인터 인스턴스 (싱글톤)
_checkpointer: BaseCheckpointSaver | None = None
_context_manager = None  # PostgreSQL context manager 참조 (cleanup용)


async def get_checkpointer() -> BaseCheckpointSaver:
    """
    🆕 LLMOps 2강: 설정에 따라 적절한 체크포인터를 반환합니다.

    체크포인터 선택 로직:
        1. ENABLE_CHECKPOINTER=false → MemorySaver (기본값)
        2. CHECKPOINTER_TYPE=memory → MemorySaver
        3. CHECKPOINTER_TYPE=postgres → AsyncPostgresSaver (Supabase)

    Returns:
        BaseCheckpointSaver: 체크포인터 인스턴스

    Raises:
        ValueError: 설정이 잘못된 경우

    Example:
        >>> checkpointer = await get_checkpointer()
        >>> graph = builder.compile(checkpointer=checkpointer)
    """
    global _checkpointer

    # 이미 초기화되었으면 재사용
    if _checkpointer is not None:
        return _checkpointer

    if not settings.enable_checkpointer or settings.checkpointer_type == "memory":
        logger.info("MemorySaver 사용 (체크포인터 비활성화 또는 memory 타입)")
        _checkpointer = MemorySaver()
        return _checkpointer

    if settings.checkpointer_type == "postgres":
        try:
            _checkpointer = await _create_postgres_checkpointer()
            return _checkpointer
        except Exception as e:
            logger.error(f"PostgreSQL 체크포인터 생성 실패: {e}")
            logger.warning("MemorySaver로 대체하여 계속 진행")
            _checkpointer = MemorySaver()
            return _checkpointer

    # 알 수 없는 타입 (fallback)
    logger.warning(
        f"알 수 없는 체크포인터 타입: {settings.checkpointer_type}, MemorySaver 사용"
    )
    _checkpointer = MemorySaver()
    return _checkpointer


async def _create_postgres_checkpointer() -> BaseCheckpointSaver:
    """
    🆕 LLMOps 2강: PostgreSQL 체크포인터 생성 (Production용)

    Supabase PostgreSQL에 체크포인트를 저장합니다.
    다중 서버 환경의 프로덕션에 적합합니다.

    Returns:
        AsyncPostgresSaver: PostgreSQL 체크포인터

    Raises:
        ValueError: SUPABASE_CONNECTION_STRING이 설정되지 않은 경우
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    # 연결 문자열 확인
    if not settings.supabase_connection_string:
        error_msg = (
            "SUPABASE_CONNECTION_STRING이 설정되지 않았습니다. "
            "Supabase > Settings > Database > Connection string에서 확인하세요."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info("🐘 AsyncPostgresSaver 사용 (Supabase PostgreSQL)")

    # AsyncIterator (context manager)를 수동으로 열기
    # from_conn_string()은 AsyncIterator를 반환하므로 __aenter__() 호출 필요
    # serde=JsonPlusSerializer(): msgpack 대신 JSON 직렬화 사용 (호환성 향상)
    cm = AsyncPostgresSaver.from_conn_string(
        settings.supabase_connection_string,
        serde=JsonPlusSerializer(),
    )
    checkpointer = await cm.__aenter__()

    # 테이블 생성 (첫 실행 시)
    await checkpointer.setup()
    logger.debug("PostgreSQL 체크포인터 테이블 설정 완료")

    # cleanup_checkpointer()에서 cm.__aexit__() 호출을 위해 저장
    global _context_manager
    _context_manager = cm

    return checkpointer


async def cleanup_checkpointer():
    """
    🆕 LLMOps 2강: 체크포인터 정리

    서버 종료 시 체크포인터 연결을 정리합니다.
    """
    global _checkpointer, _context_manager

    if _checkpointer is None:
        return

    logger.info("체크포인터 연결 정리 중...")

    # PostgreSQL context manager 정리
    if _context_manager is not None:
        try:
            await _context_manager.__aexit__(None, None, None)
            logger.debug("PostgreSQL 연결 종료")
        except Exception as e:
            logger.warning(f"PostgreSQL 연결 종료 중 오류: {e}")
        _context_manager = None

    _checkpointer = None
    logger.info("체크포인터 정리 완료")
