# 🆕 7강: 채팅 API 라우트 + 체크포인터 연동
"""
채팅 API 라우트

LangGraph 에이전트를 호출하여 사용자 메시지를 처리합니다.

엔드포인트:
    POST /chat/
    POST /chat/stream
    POST /chat/stream/nodes

🆕 7강 변경사항:
    - 체크포인터 연동 (thread_id로 대화 이어가기)
    - get_lumi_graph_with_memory() 사용
    - config = {"configurable": {"thread_id": session_id}} 전달
"""

from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from loguru import logger

from app.core.config import settings  # 🆕 7강: 설정 확인용
from app.core.tracing import create_langfuse_config
from app.graph import get_lumi_graph
from app.graph.graph import get_lumi_graph_with_memory  # 🆕 7강: 체크포인터 포함 그래프
from app.schemas.chat import ChatRequest, ChatResponse, StreamEvent

router = APIRouter()

# =============================================================
# In-Memory 세션 저장소 (하위 호환성 유지)
# 🆕 7강: 체크포인터 비활성화 시에만 사용
# 체크포인터 활성화 시 체크포인터가 대화 히스토리 관리
# =============================================================
SESSION_STORE: dict[str, list[BaseMessage]] = {}


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    채팅 엔드포인트 (일반 - 비스트리밍)

    사용자 메시지를 LangGraph 에이전트로 처리하고 응답을 반환합니다.

    LLMOps 2강: 체크포인터 연동
        - thread_id로 대화를 이어갈 수 있음
        - 같은 session_id로 호출하면 이전 대화 기억

    Args:
        request: 채팅 요청 (message, session_id, user_id)

    Returns:
        ChatResponse: 루미의 응답

    Raises:
        HTTPException: 에이전트 처리 오류 시

    Example:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/chat/" \\
            -H "Content-Type: application/json" \\
            -d '{"message": "오늘 방송 언제야?", "session_id": "user123"}'
        ```
    """
    logger.info(
        f"📩 채팅 요청: session={request.session_id}, message={request.message[:50]}..."
    )

    try:
        # LLMOps 2강: 체크포인터 활성화 여부에 따라 그래프 선택
        if settings.enable_checkpointer:
            # Step 1: 체크포인터가 포함된 그래프 가져오기
            graph = await get_lumi_graph_with_memory()

            # Step 2: 초기 상태 생성 (체크포인터가 이전 메시지 자동 로드)
            initial_state = {
                "messages": [HumanMessage(content=request.message)],
                "intent": None,
                "retrieved_docs": [],
                "tool_name": None,
                "tool_args": None,
                "tool_result": None,
                "session_id": request.session_id,
                "user_id": request.user_id,
            }

            # LLMOps 2강: thread_id로 대화 세션 구분
            # Langfuse config 생성 및 병합
            # create_langfuse_config()로 Langfuse config를 생성하고 아래 config에 병합하세요 (session_id, user_id 전달)
            langfuse_config = create_langfuse_config(
                session_id=request.session_id,
                user_id=request.user_id,
            )

            config = {
                "configurable": {"thread_id": request.session_id},
                **langfuse_config,
            }

            # Step 3: 그래프 실행 (체크포인터가 자동으로 상태 저장/복원)
            logger.debug(f"🔄 LangGraph 실행 시작 (thread_id: {request.session_id})")
            final_state = await graph.ainvoke(initial_state, config=config)
            logger.debug("✅ LangGraph 실행 완료 (체크포인터 저장됨)")
        else:
            # 체크포인터 비활성화 시 기존 방식 사용
            graph = get_lumi_graph()

            initial_state = {
                "messages": [HumanMessage(content=request.message)],
                "intent": None,
                "retrieved_docs": [],
                "tool_name": None,
                "tool_args": None,
                "tool_result": None,
                "session_id": request.session_id,
                "user_id": request.user_id,
            }

            # Langfuse config 생성 (체크포인터 없을 때)
            config = create_langfuse_config(
                session_id=request.session_id,
                user_id=request.user_id,
            )

            logger.debug("🔄 LangGraph 실행 시작")
            final_state = await graph.ainvoke(initial_state, config=config)
            logger.debug("✅ LangGraph 실행 완료")

        # Step 4: 최종 응답 추출
        messages = final_state["messages"]
        if len(messages) < 2:
            raise ValueError("응답 메시지가 없습니다.")

        ai_response = messages[-1].content
        tool_used = final_state.get("tool_name")

        logger.info(f"📤 응답 전송: tool_used={tool_used}")

        return ChatResponse(
            message=ai_response,
            tool_used=tool_used,
            cached=False,
        )

    except Exception as e:
        logger.error(f"채팅 처리 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"에이전트 처리 중 오류가 발생했습니다: {str(e)}",
        )


# =============================================================
# 🆕 3강: SSE 스트리밍 - Helper 함수
# =============================================================


async def stream_with_status(
    message: str,
    session_id: str,
    user_id: str | None = None,
) -> AsyncGenerator[tuple[str | None, str | None, str | None, str | None], None]:
    """
    🆕 3강: 노드 상태 + 토큰 스트리밍 결합
    🆕 7강: 체크포인터 연동

    진행 상황을 표시하면서 토큰도 스트리밍합니다.
    Gradio UI에서 "생각 중...", "Tool 실행 중..." 표시에 사용됩니다.

    🔑 핵심: stream_mode=["updates", "messages"]
        - updates: 노드 완료 시 이벤트 → 진행 상태 표시
        - messages: 토큰 단위 이벤트 → ChatGPT처럼 글자 스트리밍

    🆕 7강: 체크포인터 연동
        - 체크포인터 활성화 시 thread_id로 대화 이어가기
        - SESSION_STORE 대신 체크포인터가 히스토리 관리

    Yields:
        tuple[status, token, final_response, tool_used]:
            - (status, None, None, None): 진행 상황 메시지
            - (None, token, None, None): 스트리밍 중인 토큰
            - (None, None, final_response, tool_used): 최종 응답
    """
    session_id = session_id or "default"

    # 🆕 7강: 체크포인터 활성화 여부에 따라 그래프 선택
    if settings.enable_checkpointer:
        graph = await get_lumi_graph_with_memory()
        # 체크포인터가 히스토리 관리 → 새 메시지만 전달
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "intent": None,
            "retrieved_docs": [],
            "tool_name": None,
            "tool_args": None,
            "tool_result": None,
            "session_id": session_id,
            "user_id": user_id,
        }
        # thread_id로 대화 세션 구분
        # 스트리밍용 Langfuse config 병합
        langfuse_config = create_langfuse_config(
            session_id=session_id,
            user_id=user_id,
        )

        config = {"configurable": {"thread_id": session_id}, **langfuse_config}
        logger.debug(f"📜 [StreamWithStatus] 체크포인터 모드 (thread_id: {session_id})")
    else:
        graph = get_lumi_graph()
        # 체크포인터 비활성화 → SESSION_STORE에서 히스토리 가져오기
        history = SESSION_STORE.get(session_id, [])
        initial_state = {
            "messages": history + [HumanMessage(content=message)],
            "intent": None,
            "retrieved_docs": [],
            "tool_name": None,
            "tool_args": None,
            "tool_result": None,
            "session_id": session_id,
            "user_id": user_id,
        }
        # 스트리밍용 Langfuse config
        config = create_langfuse_config(
            session_id=session_id,
            user_id=user_id,
        )
        logger.debug(
            f"📜 [StreamWithStatus] 인메모리 모드, 히스토리: {len(history)}개 메시지"
        )

    new_message = HumanMessage(content=message)
    final_response = ""
    final_tool_name = None
    current_node = None

    # 노드 이름 → 사용자 친화적 메시지
    node_status = {
        "router": "🔀 루미 생각 중...",
        "rag": "📚 정보 검색 중...",
        "tool": "🔧 도구 실행 중...",
        "response": "💬 응답 작성 중...",
    }

    # 핵심: 두 모드 동시 사용! (updates + messages)
    # stream_mode가 리스트일 때: (mode_name, event) 튜플로 반환됨
    # LLMOps 2강: config 전달 (체크포인터 사용 시)
    try:
        async for mode, event in graph.astream(
            initial_state, config=config, stream_mode=["updates", "messages"]
        ):
            # 노드 스트리밍 (stream_mode="updates") : 노드가 완료될 때마다 이벤트 발생
            if mode == "updates":
                # event = {"node_name": {출력 상태}}
                for node_name, node_output in event.items():
                    if node_name != current_node and node_name in node_status:
                        current_node = node_name
                        # 진행 상황 메시지 yield
                        yield (node_status[node_name], None, None, None)
                        logger.debug(f"🔄 [StreamWithStatus] 노드 진입: {node_name}")

                    # tool 노드에서 tool_name 추출
                    if node_name == "tool" and node_output:
                        final_tool_name = node_output.get("tool_name")

                    # response 노드에서 fallback 응답 확인
                    # LLM 스트리밍이 실패하면 토큰이 안 오지만,
                    # response_node는 에러 시 fallback 메시지를 반환함
                    if node_name == "response" and node_output:
                        messages = node_output.get("messages", [])
                        if messages and not final_response:
                            # 스트리밍 토큰이 없었지만 응답이 있다면 (에러 fallback)
                            last_msg = messages[-1]
                            if hasattr(last_msg, "content") and last_msg.content:
                                final_response = last_msg.content
                                logger.debug(
                                    f"📍 [StreamWithStatus] Fallback 응답 감지: {final_response[:50]}..."
                                )

            # 토큰 스트리밍 (stream_mode="messages") : LLM이 토큰을 생성할 때마다 이벤트 발생
            elif mode == "messages":
                # event = (message, metadata) 튜플
                msg, meta = event
                node_name = meta.get("langgraph_node", "")

                # response 노드의 토큰만 스트리밍 (router 노드 토큰은 무시)
                if node_name != "response":
                    continue

                # AIMessageChunk = 토큰 하나
                if isinstance(msg, AIMessageChunk):
                    token = msg.content or ""
                    if token:
                        final_response += token
                        yield (None, token, None, None)

    except Exception as e:
        # LLMOps 1강: 스트리밍 중 에러 발생 시 사용자에게 알림
        logger.error(f"❌ [StreamWithStatus] 스트리밍 오류: {e}")
        error_message = (
            "😢 AI 서비스에 일시적인 문제가 발생했어요. 잠시 후 다시 시도해주세요!"
        )
        yield (None, None, error_message, None)
        return  # 에러 발생 시 여기서 종료

    # LLMOps 2강: 체크포인터 비활성화 시에만 SESSION_STORE에 저장
    # 체크포인터 활성화 시 체크포인터가 자동으로 상태 저장
    if final_response and not settings.enable_checkpointer:
        if session_id not in SESSION_STORE:
            SESSION_STORE[session_id] = []
        SESSION_STORE[session_id].append(new_message)
        SESSION_STORE[session_id].append(AIMessage(content=final_response))
        logger.debug(f"💾 [StreamWithStatus] 인메모리 세션 저장: {session_id}")
    elif final_response and settings.enable_checkpointer:
        logger.debug(f"💾 [StreamWithStatus] 체크포인터 자동 저장: {session_id}")

    # 마지막에 최종 응답 yield
    yield (None, None, final_response, final_tool_name)


# SSE 스트리밍 엔드포인트
@router.post("/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    SSE 노드 + 토큰 스트리밍 채팅 엔드포인트

    stream_with_status를 사용하여 노드 상태(thinking)와 토큰을 동시에 스트리밍합니다.

    SSE 이벤트 타입:
        - thinking: 노드 진행 상황 ("🔀 루미 생각 중...")
        - token: LLM 토큰 (글자 단위)
        - response: 최종 응답
        - error: 에러
        - done: 스트리밍 종료

    Example:
        ```bash
        curl -N -X POST "http://localhost:8000/api/v1/chat/stream" \\
            -H "Content-Type: application/json" \\
            -d '{"message": "오늘 방송 언제야?", "session_id": "user123"}'
        ```
    """
    logger.info(f"📩 [Stream] 노드+토큰 스트리밍 요청: session={request.session_id}")

    async def generate() -> AsyncGenerator[str, None]:
        """SSE 이벤트 생성기 - 노드 상태 + 토큰 스트리밍!"""
        try:
            async for status, token, final, tool_used in stream_with_status(
                request.message,
                request.session_id,
                request.user_id,
            ):
                # 노드 상태 (thinking 이벤트)
                if status:
                    yield StreamEvent(type="thinking", content=status).to_sse()

                # 토큰 스트리밍 (token 이벤트)
                if token:
                    yield StreamEvent(type="token", content=token).to_sse()

                # 최종 응답 (response 이벤트)
                if final:
                    yield StreamEvent(
                        type="response", content=final, tool_used=tool_used
                    ).to_sse()

            yield StreamEvent(type="done").to_sse()
            logger.info(f"✅ [Stream] 완료: session={request.session_id}")

        except Exception as e:
            logger.error(f"❌ [Stream] 오류: {e}")
            yield StreamEvent(type="error", error=str(e)).to_sse()
            yield StreamEvent(type="done").to_sse()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
