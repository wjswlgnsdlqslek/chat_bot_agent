"""
LangGraph 에이전트의 상태(State) 타입 정의

State는 LangGraph에서 노드 간에 전달되는 데이터 구조입니다.
TypedDict를 사용하여 타입 안전하게 정의합니다.

핵심 개념:
    - State는 불변(immutable) 딕셔너리처럼 동작
    - 각 노드는 State를 받아서 업데이트할 필드만 반환
    - Annotated[list, add_messages]를 사용하면 메시지가 자동으로 추가됨
"""

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class LumiState(TypedDict):
    """
    루미 에이전트의 상태 정의

    LangGraph에서 각 노드는 이 State를 입력받고,
    필요한 필드를 업데이트하여 반환합니다.

    Attributes:
        messages: 대화 기록 (LangGraph 메시지 리듀서 사용)
        intent: 라우팅 결과 (chat, rag, tool 중 하나)
        retrieved_docs: RAG 검색 결과 문서 목록
        tool_name: 실행할 Tool 이름
        tool_args: Tool 실행에 필요한 인자
        tool_result: Tool 실행 결과
        session_id: 세션 식별자
        user_id: 사용자 식별자 (선택)

    Example:
        >>> state = {
        ...     "messages": [HumanMessage(content="오늘 방송 언제야?")],
        ...     "session_id": "user123",
        ...     "intent": None,
        ...     "retrieved_docs": [],
        ...     "tool_name": None,
        ...     "tool_args": None,
        ...     "tool_result": None,
        ... }
    """

    # TODO 1: messages 필드 정의
    messages: Annotated[list[BaseMessage], add_messages]
    # TODO 2: intent 필드 정의
    intent: Literal["chat", "rag", "tool"] | None
    # TODO 3: retrieved_docs 필드 정의
    retrieved_docs: list[str]
    # TODO 4: tool 관련 필드 정의 (3개)
    tool_name: str | None
    tool_args: dict[str, Any] | None
    tool_result: dict[str, Any] | None
    # TODO 5: 세션 정보 필드 정의 (2개)
    session_id: str
    user_id: str | None


def create_initial_state(
    session_id: str,
    user_id: str | None = None,
    messages: list[BaseMessage] | None = None,
) -> LumiState:
    """
    초기 상태를 생성합니다.

    Args:
        session_id: 세션 식별자
        user_id: 사용자 식별자 (선택)
        messages: 초기 메시지 목록 (선택)

    Returns:
        LumiState: 초기화된 상태 딕셔너리

    Example:
        >>> state = create_initial_state("session-123")
        >>> print(state["session_id"])
        'session-123'
    """
    return LumiState(
        messages=messages or [],
        intent=None,
        retrieved_docs=[],
        tool_name=None,
        tool_args=None,
        tool_result=None,
        session_id=session_id,
        user_id=user_id,
    )
