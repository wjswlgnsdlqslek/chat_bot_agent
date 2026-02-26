from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """user -> server chat request schema."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="사용자의 메시지입니다. 1자 이상, 1000자 이하로 입력해주세요.",
        examples=["오늘 방송 언제야?", "노래 추천해줘"],
    )

    session_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="세션 ID입니다.",
        examples=["session_1234567890"],
    )

    user_id: str | None = Field(
        default=None,
        max_length=100,
        description="사용자 식별자 (선택)",
    )


class ChatResponse(BaseModel):
    """server -> user chat response schema."""

    message: str = Field(
        ...,
        description="서버의 응답 메시지입니다.",
    )

    tool_used: str | None = Field(
        default=None,
        description="사용된 도구의 이름입니다. 도구가 사용되지 않은 경우 None입니다.",
        examples=["search", "calculator", None],
    )

    cached: bool = Field(
        default=False,
        description="이 응답이 캐시된 결과인지 여부입니다.",
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="응답이 생성된 시간입니다.",
    )


# TODO 1: StreamEventType 정의
StreamEventType = Literal["thinking", "tool", "token", "response", "error", "done"]


class StreamEvent(BaseModel):
    """
    SSE 스트리밍 이벤트 스키마

    Server-Sent Events로 전송되는 이벤트 형식입니다.
    각 이벤트 타입에 따라 다른 필드가 채워집니다.

    Attributes:
        type: 이벤트 타입
            - thinking: 노드 실행 시작 (어떤 노드가 실행 중인지)
            - tool: Tool 실행 결과
            - token: LLM 토큰 스트리밍
            - response: 최종 응답 완료
            - error: 에러 발생
            - done: 스트리밍 종료
        node: 현재 실행 중인 노드 이름 (thinking, tool 이벤트)
        content: 텍스트 내용 (token, response 이벤트)
        tool_name: 실행된 Tool 이름 (tool 이벤트)
        tool_result: Tool 실행 결과 (tool 이벤트)
        error: 에러 메시지 (error 이벤트)

    Example:
        >>> # 노드 실행 시작 이벤트
        >>> event = StreamEvent(type="thinking", node="router")

        >>> # Tool 실행 결과 이벤트
        >>> event = StreamEvent(
        ...     type="tool",
        ...     node="tool",
        ...     tool_name="get_schedule",
        ...     tool_result={"schedules": [...]}
        ... )

        >>> # 토큰 스트리밍 이벤트
        >>> event = StreamEvent(type="token", content="안녕")

        >>> # 최종 응답 이벤트
        >>> event = StreamEvent(
        ...     type="response",
        ...     content="금요일에 뮤직뱅크 나와!",
        ...     tool_used="get_schedule"
        ... )
    """

    # TODO 2: type 필드 정의
    type: StreamEventType = Field(..., description="envet type")

    node: str | None = Field(
        default=None,
        description="g현재 실행중인 노드 이름",
        examples=["router", "rag", "tool", "response"],
    )

    # TODO 3: content 필드 정의
    content: str | None = Field(
        default=None,
        description="텍스트 내용 (토큰 또는 최종 응답)",
    )

    tool_name: str | None = Field(
        default=None,
        description="실행된 Tool 이름",
        examples=["get_schedule", "recommend_song"],
    )

    tool_result: Any | None = Field(
        default=None,
        description="Tool 실행 결과",
    )

    tool_used: str | None = Field(
        default=None,
        description="최종 응답에서 사용된 Tool",
    )

    error: str | None = Field(
        default=None,
        description="에러 메시지",
    )

    def to_sse(self) -> str:
        """
        SSE 형식 문자열로 변환

        Returns:
            str: SSE 형식 (data: {...}\n\n)
        """
        import orjson

        # TODO 4: SSE 형식으로 변환. utf-8로 디코딩
        data = {k: v for k, v in self.model_dump().items() if v is not None}
        json_str = orjson.dumps(data).decode("utf-8")

        return f"data: {json_str}\n\n"
