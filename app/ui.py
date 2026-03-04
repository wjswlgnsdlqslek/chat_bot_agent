"""
Gradio 기반 채팅 인터페이스

접속:
    - 로컬: http://localhost:8000/ui
"""

import re
import uuid

import gradio as gr
from loguru import logger

from app.core.config import settings


def sanitize_for_gradio_markdown(text: str) -> str:
    """
    Gradio 마크다운 렌더링 문제 수정

    문제 1: 단일 틸다(~)가 취소선(~~)으로 해석됨
        - "루미너스~! 😄 아마..." → 취소선 발생
        - 해결: 단일 ~를 전각 물결표(～)로 변환

    문제 2: 볼드(**)가 특수문자와 붙으면 렌더링 실패
        - **"텍스트"** → 볼드 안 됨
        - 해결: 따옴표 위치 조정
    """
    # 1. 단일 틸다 → 전각 물결표 (취소선 방지)
    text = re.sub(r"(?<!~)~(?!~)", "～", text)

    # 2. 볼드 마크다운 정리 (따옴표와 충돌 방지)
    text = re.sub(r'\*\*"', '"**', text)
    text = re.sub(r'"\*\*', '**"', text)

    return text


# ✨ 커스텀 CSS - 버추얼 아이돌 채팅앱 테마
CUSTOM_CSS = """
/* ===== Gradio Footer 숨김 ===== */
footer { display: none !important; }

/* ===== 폰트 임포트 ===== */
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&family=Quicksand:wght@400;500;600;700&display=swap');

/* ===== 전역 변수 ===== */
:root {
    --lumi-pink: #ff6b9d;
    --lumi-purple: #c44eff;
    --lumi-blue: #4ecaff;
    --lumi-gradient: linear-gradient(135deg, #ff6b9d 0%, #c44eff 50%, #4ecaff 100%);
    --glass-bg: rgba(255, 255, 255, 0.15);
    --glass-border: rgba(255, 255, 255, 0.3);
    --chat-user: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --chat-lumi: linear-gradient(135deg, #ff6b9d 0%, #ff8a80 100%);
}

/* ===== 메인 컨테이너 ===== */
.gradio-container {
    font-family: 'Noto Sans KR', 'Quicksand', sans-serif !important;
    background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #0f0c29) !important;
    background-size: 400% 400% !important;
    animation: gradientShift 15s ease infinite !important;
    min-height: 100vh !important;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ===== 오로라 오버레이 효과 ===== */
.gradio-container::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background:
        radial-gradient(ellipse at 20% 20%, rgba(255, 107, 157, 0.15) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(196, 78, 255, 0.15) 0%, transparent 50%),
        radial-gradient(ellipse at 40% 80%, rgba(78, 202, 255, 0.1) 0%, transparent 40%);
    pointer-events: none;
    z-index: 0;
}

/* ===== 반짝이는 별 효과 ===== */
.gradio-container::after {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-image:
        radial-gradient(2px 2px at 20px 30px, rgba(255, 255, 255, 0.8), transparent),
        radial-gradient(2px 2px at 40px 70px, rgba(255, 107, 157, 0.6), transparent),
        radial-gradient(1px 1px at 90px 40px, rgba(196, 78, 255, 0.8), transparent),
        radial-gradient(2px 2px at 130px 80px, rgba(78, 202, 255, 0.6), transparent),
        radial-gradient(1px 1px at 160px 120px, white, transparent);
    background-repeat: repeat;
    background-size: 200px 200px;
    animation: twinkle 4s ease-in-out infinite;
    pointer-events: none;
    z-index: 0;
    opacity: 0.5;
}

@keyframes twinkle {
    0%, 100% { opacity: 0.3; }
    50% { opacity: 0.7; }
}

/* ===== 메인 콘텐츠 영역 ===== */
.main, .contain {
    position: relative;
    z-index: 1;
}

/* ===== 헤더 스타일 ===== */
.header-container {
    text-align: center;
    padding: 2rem 1rem;
    margin-bottom: 1rem;
}

.header-container h1 {
    font-family: 'Quicksand', sans-serif !important;
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    background: var(--lumi-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: 0 0 40px rgba(255, 107, 157, 0.5);
    margin-bottom: 0.5rem;
    animation: glow 2s ease-in-out infinite alternate;
}

@keyframes glow {
    from { filter: drop-shadow(0 0 20px rgba(255, 107, 157, 0.4)); }
    to { filter: drop-shadow(0 0 30px rgba(196, 78, 255, 0.6)); }
}

.header-container p {
    color: rgba(255, 255, 255, 0.8) !important;
    font-size: 1rem;
    font-weight: 300;
}

/* ===== 기능 태그 스타일 ===== */
.feature-tags {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 0.5rem;
    margin-top: 1rem;
}

.feature-tag {
    background: var(--glass-bg);
    backdrop-filter: blur(10px);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    padding: 0.4rem 1rem;
    font-size: 0.85rem;
    color: white;
    transition: all 0.3s ease;
}

.feature-tag:hover {
    background: rgba(255, 107, 157, 0.3);
    transform: translateY(-2px);
}

/* ===== 채팅 컨테이너 (글래스모피즘) ===== */
.chat-container {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 24px !important;
    box-shadow:
        0 8px 32px rgba(0, 0, 0, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
    overflow: hidden;
}

/* ===== 채팅창 스타일 ===== */
.chatbot {
    background: transparent !important;
    border: none !important;
}

.chatbot .messages {
    background: transparent !important;
    padding: 1.5rem !important;
}

/* ===== 메시지 말풍선 (Gradio 6.0) ===== */
.chatbot .message-row {
    padding: 0.5rem 0 !important;
}

.chatbot .message-bubble {
    max-width: 80% !important;
    padding: 1rem 1.25rem !important;
    border-radius: 20px !important;
    line-height: 1.6 !important;
}

/* 사용자 메시지 */
.chatbot .message-row.user-row .message-bubble {
    background: var(--chat-user) !important;
    color: white !important;
    border-radius: 20px 20px 4px 20px !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}

/* 루미(어시스턴트) 메시지 */
.chatbot .message-row.bot-row .message-bubble {
    background: linear-gradient(135deg, rgba(255, 107, 157, 0.3) 0%, rgba(196, 78, 255, 0.3) 100%) !important;
    border: 1px solid rgba(255, 107, 157, 0.4) !important;
    color: white !important;
    border-radius: 20px 20px 20px 4px !important;
    box-shadow: 0 4px 15px rgba(255, 107, 157, 0.2) !important;
    backdrop-filter: blur(10px) !important;
}

/* 메시지 텍스트 색상 강제 적용 */
.chatbot .message-bubble p,
.chatbot .message-bubble span,
.chatbot .message-bubble {
    color: white !important;
}

/* 아바타 스타일 */
.chatbot .avatar-container,
.chatbot .avatar-image {
    width: 40px !important;
    height: 40px !important;
    border-radius: 50% !important;
    overflow: hidden !important;
    border: 2px solid var(--glass-border) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
}

.chatbot .bot-row .avatar-container {
    background: var(--lumi-gradient) !important;
    border-color: rgba(255, 107, 157, 0.5) !important;
}

/* ===== 입력 영역 ===== */
.input-row {
    padding: 1rem 1.5rem 1.5rem !important;
    background: rgba(0, 0, 0, 0.2) !important;
    border-top: 1px solid rgba(255, 255, 255, 0.1) !important;
}

/* 텍스트박스 - 모든 Gradio 입력창 타겟 */
.input-row textarea,
.input-row input[type="text"],
textarea,
input[type="text"],
.textbox textarea {
    background: rgba(255, 255, 255, 0.95) !important;
    border: 1px solid rgba(255, 107, 157, 0.4) !important;
    border-radius: 16px !important;
    color: #1a1a1a !important;
    font-family: 'Noto Sans KR', sans-serif !important;
    font-size: 1rem !important;
    padding: 1rem 1.25rem !important;
    transition: all 0.3s ease !important;
    caret-color: var(--lumi-pink) !important;
}

.input-row textarea:focus,
.input-row input[type="text"]:focus,
textarea:focus,
input[type="text"]:focus,
.textbox textarea:focus {
    outline: none !important;
    border-color: var(--lumi-pink) !important;
    box-shadow: 0 0 20px rgba(255, 107, 157, 0.3) !important;
    background: #ffffff !important;
    color: #1a1a1a !important;
}

.input-row textarea::placeholder,
.input-row input[type="text"]::placeholder,
textarea::placeholder,
.textbox textarea::placeholder {
    color: rgba(150, 100, 120, 0.7) !important;
}

/* 전송 버튼 */
.send-btn {
    background: var(--lumi-gradient) !important;
    border: none !important;
    border-radius: 14px !important;
    color: white !important;
    font-weight: 600 !important;
    font-family: 'Noto Sans KR', sans-serif !important;
    padding: 0.8rem 1.5rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(255, 107, 157, 0.4) !important;
    text-transform: none !important;
}

.send-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(255, 107, 157, 0.5) !important;
}

.send-btn:active {
    transform: translateY(0) !important;
}

/* ===== 빠른 응답 버튼 ===== */
.quick-buttons {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    justify-content: center;
    padding: 1rem;
}

.quick-btn {
    background: rgba(255, 255, 255, 0.1) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 20px !important;
    color: white !important;
    font-family: 'Noto Sans KR', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.2rem !important;
    transition: all 0.3s ease !important;
    cursor: pointer !important;
}

.quick-btn:hover {
    background: rgba(255, 107, 157, 0.3) !important;
    border-color: rgba(255, 107, 157, 0.5) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 15px rgba(255, 107, 157, 0.3) !important;
}

/* ===== 초기화 버튼 ===== */
.clear-btn {
    background: transparent !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    border-radius: 12px !important;
    color: rgba(255, 255, 255, 0.7) !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.3s ease !important;
    margin-top: 0.5rem !important;
}

.clear-btn:hover {
    background: rgba(255, 255, 255, 0.1) !important;
    border-color: rgba(255, 255, 255, 0.5) !important;
    color: white !important;
}

/* ===== 푸터 ===== */
.footer {
    text-align: center;
    padding: 1.5rem;
    color: rgba(255, 255, 255, 0.5);
    font-size: 0.8rem;
}

.footer code {
    background: rgba(255, 255, 255, 0.1);
    padding: 0.2rem 0.5rem;
    border-radius: 6px;
    font-family: 'Monaco', 'Consolas', monospace;
    font-size: 0.75rem;
}

/* ===== 섹션 라벨 숨기기 ===== */
.chatbot > label,
.block > label span {
    display: none !important;
}

/* ===== 스크롤바 스타일 ===== */
::-webkit-scrollbar {
    width: 6px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background: rgba(255, 107, 157, 0.3);
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 107, 157, 0.5);
}

/* ===== 복사 버튼 숨기기 ===== */
.chatbot button[aria-label="Copy"],
.chatbot .copy-btn,
.chatbot .message-buttons,
.chatbot [data-testid="copy-btn"],
.chatbot svg.copy-icon,
button.copy {
    display: none !important;
}

/* ===== Processing 타이머 숨기기 ===== */
/* 어차피 첫 yield에서 사라지므로 깔끔하게 숨김 */
.generating,
.progress-text,
.eta-bar,
.progress-bar,
.progress-level,
.meta-text,
.meta-text-center,
.timer {
    display: none !important;
}

/* ===== 반응형 ===== */
@media (max-width: 768px) {
    .header-container h1 {
        font-size: 1.8rem !important;
    }

    .chatbot .user .message-content,
    .chatbot .bot .message-content {
        max-width: 85% !important;
    }

    .feature-tags {
        gap: 0.3rem;
    }

    .feature-tag {
        font-size: 0.75rem;
        padding: 0.3rem 0.7rem;
    }
}
"""

# 테마 설정
THEME = gr.themes.Base(
    primary_hue="pink",
    secondary_hue="purple",
    neutral_hue="slate",
)

# OG 이미지용 BASE_URL (상대 경로 사용)
BASE_URL = ""

# 메타 태그 (Open Graph, favicon)
# - 카카오톡, 슬랙 등에서 링크 공유 시 미리보기 표시
# - 브라우저 탭 아이콘 설정
# - ⚠️ OG 이미지는 절대 URL + PNG/JPG 권장 (SVG는 일부 플랫폼에서 미지원)
META_TAGS = f"""
<!-- Favicon -->
<link rel="icon" type="image/svg+xml" href="/static/favicon.svg">
<link rel="apple-touch-icon" href="/static/favicon.svg">

<!-- Primary Meta Tags -->
<meta name="title" content="루미(LUMI) - 버추얼 아이돌 AI 에이전트">
<meta name="description" content="버추얼 아이돌 루미와 대화하고, 스케줄 확인하고, 팬레터도 보내보세요!">
<meta name="theme-color" content="#ff6b9d">

<!-- Open Graph / Facebook -->
<meta property="og:type" content="website">
<meta property="og:url" content="{BASE_URL}">
<meta property="og:title" content="루미(LUMI) - 버추얼 아이돌 AI 에이전트">
<meta property="og:description" content="버추얼 아이돌 루미와 대화하고, 스케줄 확인하고, 팬레터도 보내보세요!">
<meta property="og:image" content="{BASE_URL}/static/og-image.png">
<meta property="og:site_name" content="Lumi Agent">

<!-- Twitter -->
<meta property="twitter:card" content="summary_large_image">
<meta property="twitter:url" content="{BASE_URL}">
<meta property="twitter:title" content="루미(LUMI) - 버추얼 아이돌 AI 에이전트">
<meta property="twitter:description" content="버추얼 아이돌 루미와 대화하고, 스케줄 확인하고, 팬레터도 보내보세요!">
<meta property="twitter:image" content="{BASE_URL}/static/og-image.png">
"""


def create_chat_handler():
    """
    🆕 3강: 스트리밍 채팅 핸들러 생성 (Direct Call 방식)

    HTTP 요청 대신 stream_with_status 함수를 직접 호출하여
    네트워크(localhost/port) 문제 없이 동작합니다.

    🆕 진행 상태 + 토큰 스트리밍:
        - 노드 상태: "🔀 루미 생각 중..." 채팅창에 표시
        - 토큰 스트리밍: 상태가 토큰으로 대체됨

    🔧 수정: 세션 ID를 파라미터로 받아 사용자별 격리
    """
    # 🆕 Direct Call - stream_with_status 직접 호출 (노드 상태 + 토큰)
    from app.api.routes.chat import stream_with_status

    async def chat_with_lumi_stream(message: str, history: list, session_id: str):
        """
        🆕 3강: 진행 상태 + 토큰 스트리밍으로 루미와 대화합니다. (Direct Call)

        stream_with_status 함수를 직접 호출하여
        진행 상태와 토큰을 실시간으로 받아 yield합니다.

        Args:
            message: 사용자 메시지
            history: 대화 히스토리
            session_id: 사용자별 고유 세션 ID (gr.BrowserState로 관리)

        이벤트 흐름:
            1. status: "🔀 루미 생각 중..." → 채팅창에 표시
            2. token: 토큰이 오면 상태를 대체
            3. final: 최종 응답
        """
        if not message.strip():
            yield "메시지를 입력해주세요!"
            return

        try:
            # 🆕 Direct Call - stream_with_status 함수 직접 호출
            current_response = ""

            async for status, token, final, tool_used in stream_with_status(
                message=message,
                session_id=session_id,
                user_id=None,
            ):
                # 🆕 진행 상태 메시지 (토큰 스트리밍 전에만 표시)
                if status and not current_response:
                    yield status

                # 토큰 스트리밍 - 글자 단위로 누적 (상태 메시지 대체)
                if token:
                    current_response += token
                    yield sanitize_for_gradio_markdown(current_response)

                if final:
                    # 최종 응답 (마크다운 수정 적용)
                    final_content = final
                    if tool_used:
                        final_content += f"\n\n✨ _{tool_used}_"
                    yield sanitize_for_gradio_markdown(final_content)

        except Exception as e:
            logger.error(f"채팅 오류: {e}")
            yield f"앗, 오류가 발생했어요: {str(e)}"

    return chat_with_lumi_stream


# =============================================================
# 🆕 3강: SSE 방식 - 프론트/백엔드 분리 시 사용
# =============================================================


def create_chat_handler_sse(api_base_url: str = "http://localhost:8000"):
    """
    🆕 3강: SSE 스트리밍 채팅 핸들러 (HTTP 방식)

    FastAPI의 /chat/stream 엔드포인트를 SSE로 호출합니다.
    프론트엔드와 백엔드가 분리된 실무 환경에서 사용하는 방식입니다.

    ⚠️ 주의:
        - localhost 연결 문제가 있을 수 있음 (Docker 등)
        - 같은 프로세스면 Direct Call 방식이 더 간단함

    🔧 수정: 세션 ID를 파라미터로 받아 사용자별 격리

    Args:
        api_base_url: FastAPI 서버 주소 (기본값: http://localhost:8000)
    """
    import json

    import httpx

    async def chat_with_lumi_sse(message: str, history: list, session_id: str):
        """
        🆕 3강: SSE로 루미와 대화합니다. (HTTP 방식)

        /chat/stream 엔드포인트를 호출하여 SSE 이벤트를 수신합니다.
        실무에서 프론트/백엔드 분리 시 이 방식을 사용합니다.

        Args:
            message: 사용자 메시지
            history: 대화 히스토리
            session_id: 사용자별 고유 세션 ID (gr.State로 관리)

        SSE 이벤트 타입:
            - thinking: 노드 진행 상황 ("router", "tool", "response")
            - token: LLM 토큰 (글자 단위)
            - tool: Tool 실행 결과
            - response: 최종 응답
            - error: 에러 발생
            - done: 스트리밍 종료
        """
        if not message.strip():
            yield "메시지를 입력해주세요!"
            return

        try:
            current_response = ""

            # 🔑 핵심: httpx로 SSE 스트리밍 연결!
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    f"{api_base_url}/api/v1/chat/stream",
                    json={
                        "message": message,
                        "session_id": session_id,
                    },
                ) as response:
                    # 🔑 SSE 이벤트를 한 줄씩 읽기
                    async for line in response.aiter_lines():
                        # SSE 형식: "data: {...}"
                        if not line.startswith("data: "):
                            continue

                        # JSON 파싱
                        try:
                            event = json.loads(line[6:])  # "data: " 제거
                        except json.JSONDecodeError:
                            continue

                        event_type = event.get("type")

                        # 📍 thinking: 노드 진행 상황
                        # content에 이미 "🔀 루미 생각 중..." 같은 메시지가 들어있음
                        if event_type == "thinking":
                            status_msg = event.get("content", "")
                            if status_msg and not current_response:
                                yield status_msg

                        # 📍 token: LLM 토큰 스트리밍
                        elif event_type == "token":
                            token = event.get("content", "")
                            if token:
                                current_response += token
                                yield sanitize_for_gradio_markdown(current_response)

                        # 📍 tool: Tool 실행 결과
                        elif event_type == "tool":
                            tool_name = event.get("tool_name", "")
                            if tool_name and not current_response:
                                yield f"🔧 {tool_name} 실행 완료!"

                        # 📍 response: 최종 응답
                        elif event_type == "response":
                            final_content = event.get("content", "")
                            tool_used = event.get("tool_used")
                            if tool_used:
                                final_content += f"\n\n✨ _{tool_used}_"
                            yield sanitize_for_gradio_markdown(final_content)

                        # 📍 error: 에러
                        elif event_type == "error":
                            error_msg = event.get("error", "알 수 없는 오류")
                            yield f"❌ 오류: {error_msg}"

                        # 📍 done: 종료
                        elif event_type == "done":
                            break

        except httpx.ConnectError as e:
            logger.error(f"SSE 연결 실패: {e}")
            yield f"❌ 서버 연결 실패: {api_base_url}\n\n💡 Direct Call 방식을 사용해보세요."
        except Exception as e:
            logger.error(f"SSE 오류: {e}")
            yield f"앗, 오류가 발생했어요: {str(e)}"

    return chat_with_lumi_sse


# =============================================================
# 🎛️ 스트리밍 방식 선택
# =============================================================
# 기본값: Direct Call (같은 프로세스, 네트워크 문제 없음)
# 옵션: SSE (프론트/백엔드 분리 시)
#
# SSE 방식을 쓰려면:
#   chat_with_lumi = create_chat_handler_sse("http://localhost:8000")
# =============================================================


def create_demo(api_base_url: str | None = None) -> gr.Blocks:
    """
    Gradio 데모 앱 생성

    Args:
        api_base_url: FastAPI 서버 URL (None일 경우 settings.port 사용)

    Returns:
        gr.Blocks: Gradio 앱
    """

    # API URL이 없으면 settings의 host/port 사용
    if not api_base_url:
        host = "localhost" if settings.host == "0.0.0.0" else settings.host
        api_base_url = f"http://{host}:{settings.port}"

    # ===========================================
    # 🎛️ 스트리밍 방식 선택 (둘 중 하나만 활성화!)
    # ===========================================
    #
    # 방식 1: Direct Call (같은 프로세스일 때) - 기본값
    #   - Gradio가 FastAPI와 같은 프로세스에서 실행될 때
    #   - 네트워크 없이 함수 직접 호출 → 빠르고 간단!
    #
    # 방식 2: SSE (프론트/백엔드 분리 시)
    #   - React, Vue, Next.js 등 별도 프론트엔드 사용 시
    #   - HTTP로 /chat/stream 엔드포인트 호출
    #   - 실무 표준 방식!
    #
    # ===========================================

    # ✅ 방식 1: Direct Call (마운트 구조에서는 이것 사용!)
    chat_with_lumi = create_chat_handler()

    # 🔄 방식 2: SSE (Gradio를 별도 프로세스로 실행할 때만!)
    # chat_with_lumi = create_chat_handler_sse(api_base_url)

    # LLMOps 2강: 세션 ID는 gr.BrowserState로 localStorage에서 관리
    # - 페이지 로드 시: localStorage에서 세션 ID 로드
    # - 값 변경 시: localStorage에 자동 저장
    # - 새로고침해도 유지됨!

    def generate_session_id() -> str:
        """새 세션 ID 생성"""
        return f"lumi-{uuid.uuid4().hex[:8]}"

    with gr.Blocks(
        title="루미(LUMI) - 버추얼 아이돌 AI 에이전트",
        head=META_TAGS,
        analytics_enabled=False,
    ) as demo:
        # CSS 직접 삽입 (마운트 시에도 적용되도록)
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")

        # LLMOps 2강: gr.BrowserState로 세션 ID 관리 (localStorage 자동 연동)
        # - storage_key와 secret을 하드코딩해야 서버 재시작 후에도 유지됨
        session_storage = gr.BrowserState(
            "",
            storage_key="lumi_session_id",  # localStorage 키 고정
            secret="lumi-agent-session-secret-key",  # 암호화 키 고정 (서버 재시작 후에도 복호화 가능)
        )
        session_state = gr.State(value=None)

        # 헤더
        gr.HTML(
            """
            <div class="header-container">
                <h1>✨ LUMI ✨</h1>
                <p>버추얼 아이돌 루미와 대화해보세요!</p>
            </div>
            """
        )

        # 채팅 컨테이너
        with gr.Column(elem_classes="chat-container"):
            # 채팅 인터페이스
            chatbot = gr.Chatbot(
                label="루미와 대화",
                height=450,
                elem_classes="chatbot",
                avatar_images=(
                    None,
                    "https://api.dicebear.com/9.x/adventurer/svg?seed=Lumi&hair=long16&hairColor=f06292&skinColor=fce4ec&backgroundColor=ff6b9d&eyes=variant01&eyebrows=variant01&mouth=variant01",
                ),
            )

            # 입력 영역
            with gr.Row(elem_classes="input-row"):
                msg = gr.Textbox(
                    placeholder="루미에게 메시지를 보내세요... 💭",
                    scale=4,
                    show_label=False,
                    container=False,
                )
                submit_btn = gr.Button(
                    "전송 ✨",
                    variant="primary",
                    scale=1,
                    elem_classes="send-btn",
                )

        # 빠른 응답 버튼
        gr.HTML(
            '<div style="text-align: center; margin-top: 1rem; color: rgba(255,255,255,0.6); font-size: 0.9rem;">💡 빠른 질문</div>'
        )
        with gr.Row(elem_classes="quick-buttons"):
            btn1 = gr.Button("👋 안녕!", elem_classes="quick-btn")
            btn2 = gr.Button("🔮 MBTI 뭐야?", elem_classes="quick-btn")
            btn3 = gr.Button("📅 이번 주 방송?", elem_classes="quick-btn")
            btn4 = gr.Button("🎵 노래 추천!", elem_classes="quick-btn")

        # 초기화 버튼
        with gr.Row():
            clear_btn = gr.Button("🗑️ 대화 초기화", elem_classes="clear-btn")

        # 🆕 3강: 스트리밍 이벤트 핸들러
        def add_user_message(message: str, chat_history: list) -> tuple:
            """1단계: 사용자 메시지 먼저 표시"""
            if not message.strip():
                return "", chat_history
            chat_history.append({"role": "user", "content": message})
            return "", chat_history

        async def get_bot_response_stream(chat_history: list, session_id: str | None):
            """
            🆕 3강: 스트리밍 봇 응답 생성

            chat_with_lumi가 응답을 yield할 때마다 채팅창 업데이트.
            - 먼저 "🔀 루미 생각 중..." 표시
            - 토큰이 오면 응답으로 대체

            🔧 수정: session_id를 파라미터로 받아 사용자별 격리
            """
            # LLMOps 2강: session_id가 없으면 기본값 사용
            if not session_id:
                session_id = "default"
                logger.warning("⚠️ 세션 ID가 없어 기본값 사용")
            else:
                logger.debug(f"🔑 세션 ID: {session_id}")
            if not chat_history:
                yield chat_history
                return

            # 마지막 사용자 메시지 가져오기
            last_msg = chat_history[-1]

            # 메시지 내용 추출 (다양한 형식 처리)
            if isinstance(last_msg, dict):
                content = last_msg.get("content", "")
                # Gradio 멀티모달 형식: [{'text': '...', 'type': 'text'}]
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            last_user_msg = part.get("text", "")
                            break
                    else:
                        last_user_msg = str(content)
                else:
                    last_user_msg = str(content)
            else:
                last_user_msg = str(last_msg)

            if not last_user_msg:
                yield chat_history
                return

            # 🆕 스트리밍 응답 생성
            chat_history.append({"role": "assistant", "content": ""})

            # 🔧 수정: session_id를 chat_with_lumi에 전달
            async for partial_response in chat_with_lumi(
                str(last_user_msg), chat_history, session_id
            ):
                # 마지막 assistant 메시지 업데이트
                chat_history[-1] = {"role": "assistant", "content": partial_response}
                yield chat_history

        # 🆕 3강: 전송 이벤트 - 스트리밍 체이닝
        # 🔧 수정: session_state 추가 및 concurrency_limit=None으로 병렬 처리 허용
        msg.submit(add_user_message, [msg, chatbot], [msg, chatbot]).then(
            get_bot_response_stream,
            [chatbot, session_state],
            [chatbot],
            concurrency_limit=None,  # 🔧 여러 요청 병렬 처리 허용
        )
        submit_btn.click(add_user_message, [msg, chatbot], [msg, chatbot]).then(
            get_bot_response_stream,
            [chatbot, session_state],
            [chatbot],
            concurrency_limit=None,  # 🔧 여러 요청 병렬 처리 허용
        )

        # 빠른 응답 버튼 이벤트
        btn1.click(lambda: "안녕!", outputs=msg)
        btn2.click(lambda: "너 MBTI 뭐야?", outputs=msg)
        btn3.click(lambda: "이번 주 방송 일정 알려줘", outputs=msg)
        btn4.click(lambda: "신나는 노래 추천해줘", outputs=msg)

        # LLMOps 2강: 페이지 로드 시 localStorage에서 세션 ID 로드
        def on_load(stored_session_id: str):
            """페이지 로드 시 세션 ID 로드 (BrowserState에서)"""
            if stored_session_id:
                logger.info(f"🔑 기존 세션 로드 (localStorage): {stored_session_id}")
                return stored_session_id, stored_session_id
            else:
                new_session_id = generate_session_id()
                logger.info(f"🔑 새 세션 생성: {new_session_id}")
                return new_session_id, new_session_id

        demo.load(
            fn=on_load,
            inputs=[session_storage],
            outputs=[session_storage, session_state],  # 둘 다 업데이트
        )

        # LLMOps 2강: 클리어 시 새 세션 ID 생성
        def clear_chat():
            """대화 초기화 및 새 세션 ID 생성"""
            new_session_id = generate_session_id()
            logger.info(f"🗑️ 대화 초기화, 새 세션: {new_session_id}")
            return [], new_session_id, new_session_id

        clear_btn.click(
            fn=clear_chat,
            inputs=None,
            outputs=[chatbot, session_storage, session_state],
        )

    return demo
