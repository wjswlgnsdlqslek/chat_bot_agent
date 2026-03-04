"""
대화 기록 데이터 접근 계층

Supabase에 대화 기록을 저장합니다.
체크포인터와 달리 대화 분석/디버깅용 별도 테이블입니다.
"""

from typing import Literal

from loguru import logger

from app.core.config import settings

# Supabase 클라이언트 (싱글톤)
_supabase_client = None


def get_supabase_client():
    """Supabase 클라이언트를 반환합니다 (싱글톤)."""
    global _supabase_client

    if _supabase_client is None and settings.supabase_url and settings.supabase_key:
        try:
            from supabase import create_client

            _supabase_client = create_client(
                settings.supabase_url,
                settings.supabase_key,
            )
            logger.info("✅ Supabase 클라이언트 초기화 완료 (Conversation)")
        except Exception as e:
            logger.warning(f"Supabase 초기화 실패: {e}")
            _supabase_client = None

    return _supabase_client


class ConversationRepository:
    """
    대화 기록 Repository

    Supabase에 대화 기록을 저장/조회합니다.

    Example:
        >>> repo = ConversationRepository()
        >>> await repo.save_message(
        ...     session_id="user123",
        ...     role="user",
        ...     content="안녕!",
        ...     intent="chat"
        ... )
    """

    def __init__(self):
        """ConversationRepository 초기화"""
        self.client = get_supabase_client()

        if self.client is None:
            logger.warning("💬 Supabase 미설정 - 대화 저장 비활성화")
        else:
            logger.info("💬 Supabase 연결됨 (Conversation)")

    async def save_message(
        self,
        session_id: str,
        role: Literal["user", "assistant", "system"],
        content: str,
        intent: str | None = None,
        tool_name: str | None = None,
        tool_result: dict | None = None,
        user_id: str | None = None,
        metadata: dict | None = None,
    ) -> str | None:
        """
        대화 메시지를 저장합니다.

        Args:
            session_id: 세션 ID
            role: 메시지 역할 (user, assistant, system)
            content: 메시지 내용
            intent: 라우팅 의도 (chat, rag, tool)
            tool_name: 사용된 Tool 이름
            tool_result: Tool 실행 결과
            user_id: 사용자 ID (선택)
            metadata: 추가 메타데이터 (토큰 수, 응답 시간 등)

        Returns:
            Optional[str]: 저장된 메시지 ID (실패 시 None)
        """
        if self.client is None:
            return None

        try:
            data = {
                "session_id": session_id,
                "user_id": user_id,
                "role": role,
                "content": content,
                "intent": intent,
                "tool_name": tool_name,
                "tool_result": tool_result,
                "metadata": metadata or {},
            }

            result = self.client.table("conversations").insert(data).execute()

            if result.data:
                message_id = result.data[0]["id"]
                logger.debug(f"💬 대화 저장: {role} - {content[:30]}...")
                return message_id
            else:
                logger.warning("대화 저장 결과 없음")
                return None

        except Exception as e:
            logger.error(f"대화 저장 실패: {e}")
            return None

    async def save_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        intent: str | None = None,
        tool_name: str | None = None,
        tool_result: dict | None = None,
        user_id: str | None = None,
        metadata: dict | None = None,
    ) -> tuple[str | None, str | None]:
        """
        한 턴의 대화(user + assistant)를 저장합니다.

        Args:
            session_id: 세션 ID
            user_message: 사용자 메시지
            assistant_message: AI 응답
            intent: 라우팅 의도
            tool_name: 사용된 Tool
            tool_result: Tool 결과
            user_id: 사용자 ID
            metadata: 추가 메타데이터

        Returns:
            tuple[Optional[str], Optional[str]]: (user_message_id, assistant_message_id)
        """
        # 사용자 메시지 저장
        user_id_result = await self.save_message(
            session_id=session_id,
            role="user",
            content=user_message,
            intent=intent,
            user_id=user_id,
        )

        # AI 응답 저장
        assistant_id_result = await self.save_message(
            session_id=session_id,
            role="assistant",
            content=assistant_message,
            intent=intent,
            tool_name=tool_name,
            tool_result=tool_result,
            user_id=user_id,
            metadata=metadata,
        )

        return user_id_result, assistant_id_result

    async def get_history(
        self,
        session_id: str,
        limit: int = 50,
    ) -> list[dict]:
        """
        세션의 대화 기록을 조회합니다.

        Args:
            session_id: 세션 ID
            limit: 최대 조회 개수

        Returns:
            list[dict]: 대화 기록 목록 (시간순)
        """
        if self.client is None:
            return []

        try:
            result = (
                self.client.table("conversations")
                .select("*")
                .eq("session_id", session_id)
                .order("created_at", desc=False)
                .limit(limit)
                .execute()
            )
            return result.data or []

        except Exception as e:
            logger.error(f"대화 기록 조회 실패: {e}")
            return []

    async def get_stats(self, days: int = 7) -> dict:
        """
        대화 통계를 조회합니다.

        Args:
            days: 조회 기간 (일)

        Returns:
            dict: 통계 정보
        """
        if self.client is None:
            return {"intent_distribution": {}}

        try:
            result = self.client.rpc("get_conversation_stats").execute()
            return {
                "intent_distribution": {
                    row["intent"]: row["count"] for row in (result.data or [])
                }
            }

        except Exception as e:
            logger.error(f"통계 조회 실패: {e}")
            return {"intent_distribution": {}}

    async def delete_session(self, session_id: str) -> int:
        """
        세션의 모든 대화를 삭제합니다 (GDPR 대응).

        Args:
            session_id: 삭제할 세션 ID

        Returns:
            int: 삭제된 메시지 수
        """
        if self.client is None:
            return 0

        try:
            result = (
                self.client.table("conversations")
                .delete()
                .eq("session_id", session_id)
                .execute()
            )
            deleted = len(result.data) if result.data else 0
            logger.info(f"💬 세션 {session_id} 대화 {deleted}개 삭제")
            return deleted

        except Exception as e:
            logger.error(f"세션 삭제 실패: {e}")
            return 0


# 싱글톤 인스턴스
_conversation_repo: ConversationRepository | None = None


def get_conversation_repository() -> ConversationRepository:
    """ConversationRepository 싱글톤 인스턴스를 반환합니다."""
    global _conversation_repo
    if _conversation_repo is None:
        _conversation_repo = ConversationRepository()
    return _conversation_repo
