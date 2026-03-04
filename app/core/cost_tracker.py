"""
LLM API 비용 추적 및 관리 모듈

API 호출마다 토큰 사용량과 비용을 기록하고,
일일 한도 초과 시 알림을 발송합니다.

핵심 기능:
    1. 토큰 사용량 기록 (Supabase usage_logs 테이블)
    2. 세션별/일별 비용 집계
    3. 일일 한도 초과 시 Discord 알림

데이터베이스 스키마 (Supabase):
    ```sql
    CREATE TABLE usage_logs (
        id SERIAL PRIMARY KEY,
        session_id TEXT NOT NULL,
        model TEXT NOT NULL,
        input_tokens INTEGER NOT NULL,
        output_tokens INTEGER NOT NULL,
        total_tokens INTEGER NOT NULL,
        cost_usd DECIMAL(10, 6) NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX idx_usage_logs_session ON usage_logs(session_id);
    CREATE INDEX idx_usage_logs_created_at ON usage_logs(created_at);
    ```

사용법:
    from app.core.cost_tracker import CostTracker

    tracker = CostTracker()

    # 사용량 기록
    await tracker.track_usage(
        session_id="user-123",
        model="solar-pro",
        input_tokens=100,
        output_tokens=50
    )

    # 일일 비용 조회
    daily_cost = await tracker.get_daily_cost()
"""

from datetime import UTC, datetime
from typing import Any

import httpx
from loguru import logger

from app.core.config import settings
from app.core.token_counter import estimate_cost


class CostTracker:
    """
    🆕 LLMOps 2강: LLM API 비용 추적 클래스

    Supabase에 사용량을 기록하고, 비용을 추적합니다.
    일일 한도 초과 시 Discord로 알림을 발송합니다.

    Attributes:
        enabled: 비용 추적 활성화 여부
        daily_limit: 일일 비용 한도 (USD)
    """

    def __init__(self):
        """CostTracker 초기화"""
        self.enabled = settings.enable_cost_tracking
        self.daily_limit = settings.daily_cost_limit
        self._supabase_client = None
        self._already_alerted_today = False

        if self.enabled:
            logger.info(f"💰 CostTracker 활성화 (일일 한도: ${self.daily_limit})")
        else:
            logger.info("💰 CostTracker 비활성화 (ENABLE_COST_TRACKING=false)")

    def _get_supabase_client(self):
        """
        Supabase 클라이언트를 반환합니다 (지연 초기화)
        """
        if (
            self._supabase_client is None
            and settings.supabase_url
            and settings.supabase_key
        ):
            from supabase import create_client

            self._supabase_client = create_client(
                settings.supabase_url, settings.supabase_key
            )
        return self._supabase_client

    async def track_usage(
        self,
        session_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> dict[str, Any] | None:
        """
        🆕 LLMOps 2강: 토큰 사용량을 기록합니다.

        Args:
            session_id: 세션 식별자
            model: 사용한 모델명
            input_tokens: 입력 토큰 수
            output_tokens: 출력 토큰 수

        Returns:
            Optional[Dict]: 기록된 사용량 정보 (비활성화 시 None)

        Example:
            >>> await tracker.track_usage(
            ...     session_id="user-123",
            ...     model="solar-pro",
            ...     input_tokens=150,
            ...     output_tokens=80
            ... )
        """
        if not self.enabled:
            return None

        # 비용 계산
        cost_usd = estimate_cost(input_tokens, output_tokens, model)
        total_tokens = input_tokens + output_tokens

        logger.debug(
            f"💰 [CostTracker] 사용량 기록: "
            f"session={session_id}, model={model}, "
            f"tokens={total_tokens}, cost=${cost_usd:.6f}"
        )

        # Supabase에 기록
        try:
            record = await self._save_usage_log(
                session_id=session_id,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
            )

            # 일일 한도 체크
            await self._check_daily_limit(session_id)

            return record

        except Exception as e:
            logger.error(f"❌ [CostTracker] 사용량 기록 실패: {e}")
            return None

    async def _save_usage_log(
        self,
        session_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        cost_usd: float,
    ) -> dict[str, Any] | None:
        """
        🆕 LLMOps 2강: Supabase usage_logs 테이블에 기록합니다.
        """
        client = self._get_supabase_client()

        if client is None:
            logger.warning("Supabase 클라이언트가 초기화되지 않았습니다.")
            return None

        try:
            record = {
                "session_id": session_id,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "cost_usd": cost_usd,
                "created_at": datetime.now(UTC).isoformat(),
            }

            result = client.table("usage_logs").insert(record).execute()

            if result.data:
                logger.debug(
                    f"✅ [CostTracker] 기록 완료: id={result.data[0].get('id')}"
                )
                return result.data[0]
            return None

        except Exception as e:
            # 테이블이 없는 경우 등 에러 처리
            logger.warning(f"Supabase 기록 실패: {e}")
            return None

    async def _check_daily_limit(self, session_id: str):
        """
        🆕 LLMOps 2강: 일일 비용 한도를 체크하고, 초과 시 알림을 발송합니다.
        """
        if self._already_alerted_today:
            return

        daily_cost = await self.get_daily_cost()

        if daily_cost > self.daily_limit:
            logger.warning(
                f"⚠️ [CostTracker] 일일 비용 한도 초과! "
                f"현재: ${daily_cost:.4f}, 한도: ${self.daily_limit}"
            )

            # Discord 알림 발송
            await self._send_discord_alert(
                title="비용 한도 초과 경고",
                message=(
                    f"일일 비용 한도를 초과했습니다.\n"
                    f"- 현재 비용: ${daily_cost:.4f}\n"
                    f"- 한도: ${self.daily_limit}\n"
                    f"- 마지막 세션: {session_id}"
                ),
                color=0xFF0000,  # 빨간색
            )

            self._already_alerted_today = True

    async def get_daily_cost(self) -> float:
        """
        🆕 LLMOps 2강: 오늘의 총 비용을 조회합니다.

        Returns:
            float: 오늘 사용한 총 비용 (USD)
        """
        client = self._get_supabase_client()

        if client is None:
            return 0.0

        try:
            # 오늘 날짜 (UTC)
            today = datetime.now(UTC).date()
            today_start = datetime(today.year, today.month, today.day, tzinfo=UTC)

            # Supabase에서 오늘 비용 합계 조회
            result = (
                client.table("usage_logs")
                .select("cost_usd")
                .gte("created_at", today_start.isoformat())
                .execute()
            )

            if result.data:
                total_cost = sum(row.get("cost_usd", 0) for row in result.data)
                return total_cost

            return 0.0

        except Exception as e:
            logger.warning(f"일일 비용 조회 실패: {e}")
            return 0.0

    async def _send_discord_alert(
        self,
        title: str,
        message: str,
        color: int = 0xFF9900,  # 주황색
    ):
        """
        🆕 LLMOps 2강: Discord Webhook으로 알림을 발송합니다.

        Args:
            title: 알림 제목
            message: 알림 내용
            color: 임베드 색상 (16진수)
        """
        webhook_url = settings.discord_webhook_url

        if not webhook_url:
            logger.debug("Discord Webhook URL이 설정되지 않았습니다.")
            return

        try:
            # Discord Webhook payload 구성
            payload = {
                "embeds": [
                    {
                        "title": f"[Lumi Agent] {title}",
                        "description": message,
                        "color": color,
                        "timestamp": datetime.now(UTC).isoformat(),
                        "footer": {"text": f"environment: {settings.environment}"},
                    }
                ]
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(webhook_url, json=payload)
                response.raise_for_status()

            logger.info(f"✅ Discord 알림 발송 완료: {title}")

        except Exception as e:
            logger.error(f"❌ Discord 알림 발송 실패: {e}")


# 전역 CostTracker 인스턴스 (싱글톤)
_cost_tracker: CostTracker | None = None


def get_cost_tracker() -> CostTracker:
    """
    🆕 LLMOps 2강: CostTracker 싱글톤 인스턴스를 반환합니다.

    Returns:
        CostTracker: 비용 추적 인스턴스
    """
    global _cost_tracker

    if _cost_tracker is None:
        _cost_tracker = CostTracker()

    return _cost_tracker
