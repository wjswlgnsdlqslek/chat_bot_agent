"""
HTTP 엔드포인트 정의

각 라우터는 특정 도메인의 API를 담당합니다:
    - chat.py: 채팅 API
"""

from fastapi import APIRouter

from app.api.routes import chat

api_router = APIRouter()
api_router.include_router(chat.router, prefix="/chat", tags=["Chat"])
