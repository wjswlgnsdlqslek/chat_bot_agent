"""
채팅 API 라우트

LangGraph 에이전트를 호출하여 사용자 메시지를 처리합니다.

엔드포인트:
    POST /chat/          - 채팅 메시지 전송
"""

from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from loguru import logger

from app.schemas.chat import ChatRequest, ChatResponse, StreamEvent
from app.graph import get_lumi_graph

router = APIRouter()

# In-Memory 세션 저장소 (서버 재시작 시 초기화됨)
SESSION_STORE: dict[str, list[BaseMessage]] = {}


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    채팅 엔드포인트

    사용자 메시지를 LangGraph 에이전트로 처리하고 응답을 반환합니다.

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
    logger.info(f"📩 채팅 요청: session={request.session_id}")

    try:
        # TODO 1: LangGraph 그래프 가져오기
        graph = get_lumi_graph()

        # TODO 2: 초기 상태 생성
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

        # TODO 3: 그래프 실행 (비동기)
        final_state = await graph.ainvoke(initial_state)

        # TODO 4: 최종 응답 추출
        messages = final_state["messages"]
        if not messages:
            raise ValueError("응답 메시지가 없습니다.")

        ai_response = messages[-1].content
        tool_used = final_state.get("tool_name")

        logger.info(f"📤 응답 전송: tool_used={tool_used}")
        
        # TODO 5: ChatResponse 반환
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

# SSE 스트리밍 - Helper 함수
async def stream_with_status(
    message: str,
    session_id: str,
    user_id: str | None = None,
) -> AsyncGenerator[tuple[str | None, str | None, str | None, str | None], None]:
    """
    노드 상태 + 토큰 스트리밍 결합

    진행 상황을 표시하면서 토큰도 스트리밍합니다.
    Gradio UI에서 "생각 중...", "Tool 실행 중..." 표시에 사용됩니다.

    핵심: stream_mode=["updates", "messages"]
        - updates: 노드 완료 시 이벤트 → 진행 상태 표시
        - messages: 토큰 단위 이벤트 → ChatGPT처럼 글자 스트리밍

    Yields:
        tuple[status, token, final_response, tool_used]:
            - (status, None, None, None): 진행 상황 메시지
            - (None, token, None, None): 스트리밍 중인 토큰
            - (None, None, final_response, tool_used): 최종 응답
    """
    graph = get_lumi_graph()

    # 세션에서 이전 메시지 히스토리 가져오기
    session_id = session_id or "default"
    history = SESSION_STORE.get(session_id, [])
    new_message = HumanMessage(content=message)

    # 초기 상태 생성
    initial_state = {
        "messages": history + [new_message],
        "intent": None,
        "retrieved_docs": [],
        "tool_name": None,
        "tool_args": None,
        "tool_result": None,
        "session_id": session_id,
        "user_id": user_id,
    }

    logger.debug(f"📜 [StreamWithStatus] 세션 히스토리: {len(history)}개 메시지")

    final_response = ""
    final_tool_name = None
    current_node = None

    # TODO 1: 노드별 상태 메시지 정의
    node_status = {
        "router": "루미 생각 중 ...",
        "rag": "정보 검색 중 ...",
        "tool": "도구 실행 중 ...",
        "response": "응답 생성 중 ...",
    }

    # TODO 2: 스트리밍 모드 설정
    async for mode, event in graph.astream(initial_state, stream_mode=["updates", "messages"]):  # type: ignore # stream_mode 수정!
        # TODO 3: 노드 스트리밍 (mode == "updates")
        if mode == "updates":
            for node_name, node_output in event.items():
                # 여기에 구현하세요!
                if node_name != current_node and node_name in node_status:
                    current_node = node_name
                    yield (node_status[node_name], None, None, None)
                    logger.debug(f"[stream_with_status] 노드 진입: {node_name}")
                    
                if node_name =="tool" and node_output:
                    final_tool_name = node_output.get("tool_name")
                    
        # TODO 4: 토큰 스트리밍 (mode == "messages")
        elif mode == "messages":    
            msg, meta = event
            # 여기에 구현하세요!
            node_name = meta.get("langgraph_node", "")
            
            if node_name != "response":
                continue
            if isinstance(msg, AIMessageChunk):
                token = msg.content or ""
                if token:
                    final_response += token
                    yield (None, token, None, None)

    # 세션 히스토리 저장
    if final_response:
        if session_id not in SESSION_STORE:
            SESSION_STORE[session_id] = []
        SESSION_STORE[session_id].append(new_message)
        SESSION_STORE[session_id].append(AIMessage(content=final_response))
        logger.debug(f"💾 [StreamWithStatus] 세션 저장: {session_id}")

    # 마지막에 최종 응답 yield : status, token, final_response, final_tool_name
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
    logger.info(f"📩 [Stream] 스트리밍 요청: session={request.session_id}")

    async def generate() -> AsyncGenerator[str, None]:
        """SSE 이벤트 생성기 - 노드 상태 + 토큰 스트리밍"""
        try:
            async for status, token, final, tool_used in stream_with_status(
                request.message,
                request.session_id,
                request.user_id,
            ):
                # TODO 5: 이벤트 타입별(thinking, token, response) SSE 전송
                if status:
                    yield StreamEvent(type="thinking", content=status).to_sse()
                    
                    
                    
                if token:
                    yield StreamEvent(type="token", content=token).to_sse()
                if final:
                    yield StreamEvent(type="response", content=final, tool_used=tool_used).to_sse()

            # TODO 6: 완료 이벤트 전송
            yield StreamEvent(type="done").to_sse()
            logger.info(f"✅ [Stream] 완료: session={request.session_id}")


        except Exception as e:
            logger.error(f"❌ [Stream] 오류: {e}")
            yield StreamEvent(type="error", error=str(e)).to_sse()
            yield StreamEvent(type="done").to_sse()

    # TODO 7: StreamingResponse 반환
    return StreamingResponse(
        generate(),
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-buffering": "no",
            "Content-Type": "text/event-stream",
        }
    )  