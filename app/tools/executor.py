from datetime import date, timedelta
from typing import Any

from loguru import logger

from app.repositories.fan_letter import FanLetterRepository
from app.repositories.schedule import ScheduleRepository


class ToolExecutor:
    """Router가 결정한 Tool을 실행합니다."""

    async def execute(
        self,
        tool_name: str,
        tool_args: dict[str, Any] | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        args = tool_args or {}

        try:
            if tool_name == "get_schedule":
                return await self._get_schedule(args)

            if tool_name == "send_fan_letter":
                return await self._send_fan_letter(args, session_id, user_id)

            if tool_name == "recommend_song":
                return self._recommend_song(args)

            if tool_name == "get_weather":
                return {
                    "success": False,
                    "tool": tool_name,
                    "error": "get_weather Tool은 아직 준비 중이야.",
                }

            return {
                "success": False,
                "tool": tool_name,
                "error": f"지원하지 않는 Tool이야: {tool_name}",
            }

        except Exception as e:
            logger.error(f"Tool 실행 오류 ({tool_name}): {e}")
            return {
                "success": False,
                "tool": tool_name,
                "error": str(e),
            }

    async def _get_schedule(self, args: dict[str, Any]) -> dict[str, Any]:
        today = date.today()
        start_date = str(args.get("start_date") or today.isoformat())
        end_date = str(args.get("end_date") or (today + timedelta(days=7)).isoformat())
        event_type = args.get("event_type")

        repo = ScheduleRepository()
        schedules = await repo.get_schedules(
            start_date=start_date,
            end_date=end_date,
            event_type=event_type,
        )

        return {
            "success": True,
            "tool": "get_schedule",
            "count": len(schedules),
            "params": {
                "start_date": start_date,
                "end_date": end_date,
                "event_type": event_type,
            },
            "data": schedules,
        }

    async def _send_fan_letter(
        self,
        args: dict[str, Any],
        session_id: str | None,
        user_id: str | None,
    ) -> dict[str, Any]:
        category = str(args.get("category") or "general")
        message = str(args.get("message") or "").strip()

        if not message:
            return {
                "success": False,
                "tool": "send_fan_letter",
                "error": "팬레터 메시지가 비어 있어.",
            }

        repo = FanLetterRepository()
        letter_id = await repo.create(
            session_id=session_id or "anonymous",
            category=category,
            message=message,
            user_id=user_id,
        )

        if not letter_id:
            return {
                "success": False,
                "tool": "send_fan_letter",
                "error": "팬레터 저장에 실패했어.",
            }

        return {
            "success": True,
            "tool": "send_fan_letter",
            "letter_id": letter_id,
        }

    def _recommend_song(self, args: dict[str, Any]) -> dict[str, Any]:
        mood = str(args.get("mood") or "신남")

        playlist = {
            "신남": ["NewJeans - Super Shy", "IVE - I AM", "LE SSERAFIM - ANTIFRAGILE"],
            "잔잔": [
                "AKMU - 어떻게 이별까지 사랑하겠어",
                "아이유 - 밤편지",
                "Paul Kim - 모든 날, 모든 순간",
            ],
            "집중": [
                "Nujabes - Aruarian Dance",
                "Yiruma - River Flows in You",
                "Lofi Girl Mix",
            ],
        }

        picks = playlist.get(mood, playlist["신남"])

        return {
            "success": True,
            "tool": "recommend_song",
            "mood": mood,
            "recommendations": picks,
        }
