"""
Langfuse 통합 모듈

LangGraph와 Langfuse를 연동하여 Observability를 제공합니다.

🆕 LLMOps 3강 주요 기능:
    1. LangChain CallbackHandler: 노드 실행 흐름 자동 추적
    2. trace_llm_generation(): 토큰/비용 수동 기록 (generation 타입)

사용법:
    from app.core.tracing import init_langfuse, create_langfuse_config

    # 앱 시작 시 1회 초기화
    init_langfuse()

    # LangGraph에서 사용
    config = create_langfuse_config(session_id="user-123")
    result = await graph.ainvoke(state, config=config)

토큰/비용 추적 원리:
    - LangfuseCallbackHandler: 노드를 span으로 기록 (토큰 추적 X)
    - trace_llm_generation(): generation 타입으로 기록 (토큰 추적 O)
    - Langfuse는 generation 타입에서만 토큰/비용을 계산함
"""

from loguru import logger

from app.core.config import settings

# Langfuse 전역 초기화
_langfuse_initialized = False


def init_langfuse():
    """
    Langfuse 전역 초기화 (앱 시작 시 1회)

    Returns:
        Langfuse | None: 초기화된 Langfuse 클라이언트 또는 None
    """
    global _langfuse_initialized

    if _langfuse_initialized:
        return get_langfuse_client()

    # API 키 확인 및 Langfuse 클라이언트 초기화
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.warning(
            "⚠️ Langfuse API 키가 설정되지 않았습니다. 트레이싱이 비활성화됩니다."
        )
        return None

    try:
        from langfuse import Langfuse, get_client

        Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            flush_interval=settings.langfuse_flush_interval,
        )
        _langfuse_initialized = True
        logger.info("✅ Langfuse 초기화 완료")
        return get_client()

    except Exception as e:
        logger.error(f"❌ Langfuse 초기화 실패: {e}")
        return None


def get_langfuse_client():
    """
    Langfuse 클라이언트 싱글톤 반환

    Returns:
        Langfuse | None: Langfuse 클라이언트 또는 None
    """
    global _langfuse_initialized

    if not _langfuse_initialized:
        return init_langfuse()

    try:
        from langfuse import get_client

        return get_client()
    except Exception as e:
        logger.warning(f"⚠️ [Langfuse] 클라이언트 가져오기 실패: {e}")
        return None


def get_langfuse_handler():
    """
    LangChain 콜백 핸들러 생성

    CallbackHandler는 파라미터 없이 생성합니다.
    session_id, user_id는 config["metadata"]에 langfuse_ prefix로 전달합니다.

    Returns:
        CallbackHandler | None: Langfuse 콜백 핸들러 또는 None
    """
    # Langfuse 콜백 핸들러 생성
    if not settings.enable_langfuse:
        return None

    langfuse = get_langfuse_client()

    if langfuse is None:
        return None

    try:
        from langfuse.langchain import CallbackHandler

        handler = CallbackHandler()
        logger.debug("✅ Langfuse 콜백 핸들러 생성 완료")
        return handler
    except Exception as e:
        logger.error(f"❌ Langfuse 콜백 핸들러 생성 실패: {e}")
        return None


def create_langfuse_config(
    session_id: str,
    user_id: str | None = None,
    tags: list[str] | None = None,
) -> dict:
    """
    Langfuse 메타데이터가 포함된 config 생성

    Args:
        session_id: 세션 ID (대화 그룹화용)
        user_id: 사용자 ID (선택)
        tags: 태그 목록 (선택)

    Returns:
        dict: LangGraph config (callbacks + metadata)
    """
    # LangGraph용 config 생성
    handler = get_langfuse_handler()
    if handler is None:
        return {}

    default_tags = [settings.environment, "lumi-chat"]
    if tags:
        default_tags.extend(tags)

    metadata = {
        "langfuse_session_id": session_id,
        "langfuse_tags": default_tags,
    }

    if user_id:
        metadata["langfuse_user_id"] = user_id

    return {"callbacks": [handler], "metadata": metadata}


def trace_llm_generation(
    session_id: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    input_content: str | None = None,
    output_content: str | None = None,
    user_id: str | None = None,
):
    """
    🆕 LLMOps 3강: LLM 호출을 Langfuse generation으로 기록

    Langfuse에서 토큰/비용을 추적하려면 observation 타입이 'generation'이어야 합니다.
    LangfuseCallbackHandler는 span으로 기록되어 토큰 추적이 안 되므로,
    수동으로 generation을 생성합니다.

    Args:
        session_id: 세션 ID
        model: 모델명 (예: "solar-pro2", "openai/solar-pro2")
        input_tokens: 입력 토큰 수
        output_tokens: 출력 토큰 수
        input_content: 입력 내용 (선택)
        output_content: 출력 내용 (선택)
        user_id: 사용자 ID (선택)

    Note:
        - OpenAI 호환 스키마 사용: prompt_tokens, completion_tokens, total_tokens
        - Langfuse가 자동으로 input → prompt_tokens 등으로 매핑
    """
    langfuse = get_langfuse_client()
    if langfuse is None:
        return

    try:
        total_tokens = input_tokens + output_tokens

        # Langfuse v3: start_as_current_observation으로 generation 기록
        # as_type="generation"으로 설정해야 토큰/비용 추적 가능!
        with langfuse.start_as_current_observation(
            as_type="generation",
            name="llm-response",
            model=model,
            input=input_content,
        ) as generation:
            # trace에 session_id, user_id 설정
            generation.update_trace(
                session_id=session_id,
                user_id=user_id,
            )

            # usage_details로 토큰 정보 전송
            # OpenAI 호환 스키마: prompt_tokens, completion_tokens
            generation.update(
                output=output_content,
                usage_details={
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": total_tokens,
                },
            )

        logger.debug(
            f"📊 [Langfuse] LLM generation 기록: {model}, "
            f"tokens={input_tokens}+{output_tokens}={total_tokens}"
        )

    except Exception as e:
        logger.error(f"❌ [Langfuse] LLM generation 기록 실패: {e}")


def flush_langfuse():
    """
    Langfuse 버퍼 플러시 (앱 종료 시 호출)
    """
    langfuse = get_langfuse_client()
    if langfuse is None:
        return

    try:
        langfuse.flush()
        logger.info("🔄 [Langfuse] 버퍼 플러시 완료")
    except Exception as e:
        logger.error(f"❌ [Langfuse] 플러시 실패: {e}")
