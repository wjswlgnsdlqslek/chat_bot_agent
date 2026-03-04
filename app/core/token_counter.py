"""
tiktoken을 사용한 토큰 수 계산 모듈

LLM API 호출 시 토큰 수를 미리 계산하여:
1. Context Window 제한을 초과하지 않도록 관리
2. 비용을 사전에 예측
3. 메시지 트리밍 시 정확한 토큰 수 계산

지원 모델:
    - Upstage Solar: cl100k_base 인코딩 사용 (GPT-4와 동일)
    - OpenAI GPT-4/GPT-3.5: cl100k_base 인코딩
    - 기타 모델: cl100k_base로 근사치 계산

사용법:
    from app.core.token_counter import count_tokens, count_messages_tokens

    # 단일 텍스트 토큰 수 계산
    tokens = count_tokens("안녕하세요!")

    # 메시지 리스트 전체 토큰 수 계산
    messages = [HumanMessage(content="안녕"), AIMessage(content="반가워!")]
    total_tokens = count_messages_tokens(messages)

주의:
    - tiktoken은 영어 기준으로 최적화되어 있음
    - 한국어는 토큰당 문자 수가 적어 더 많은 토큰 소모
    - 정확한 비용은 API 응답의 usage 정보 사용
"""

from functools import lru_cache

from langchain_core.messages import BaseMessage
from loguru import logger


@lru_cache(maxsize=1)
def get_encoding():
    """
    🆕 LLMOps 2강: tiktoken 인코딩 객체를 반환합니다 (싱글톤)

    cl100k_base 인코딩을 사용합니다.
    이 인코딩은 GPT-4, GPT-3.5-turbo, text-embedding-ada-002에서 사용됩니다.

    Returns:
        tiktoken.Encoding: 토큰 인코딩 객체

    Note:
        첫 호출 시 인코딩 파일을 다운로드할 수 있음 (약 1.5MB)
    """
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except ImportError:
        logger.warning("tiktoken이 설치되지 않았습니다. 근사치 계산을 사용합니다.")
        return None
    except Exception as e:
        logger.warning(f"tiktoken 초기화 실패: {e}. 근사치 계산을 사용합니다.")
        return None


def count_tokens(text: str) -> int:
    """
    🆕 LLMOps 2강: 텍스트의 토큰 수를 계산합니다.

    tiktoken 라이브러리를 사용하여 정확한 토큰 수를 계산합니다.
    tiktoken이 없는 경우 근사치를 반환합니다.

    Args:
        text: 토큰 수를 계산할 텍스트

    Returns:
        int: 토큰 수

    Example:
        >>> count_tokens("Hello, world!")
        4
        >>> count_tokens("안녕하세요!")
        5  # 한국어는 더 많은 토큰 사용
    """
    if not text:
        return 0

    encoding = get_encoding()

    # tiktoken으로 토큰 수 계산
    if encoding is not None:
        return len(encoding.encode(text))
    else:
        # tiktoken 없으면 근사치 계산
        # 영어: 약 4글자당 1토큰
        # 한국어: 약 2글자당 1토큰 (더 많은 토큰 사용)
        # 보수적으로 2글자당 1토큰으로 계산
        return len(text) // 2 + 1


def count_messages_tokens(messages: list[BaseMessage]) -> int:
    """
    🆕 LLMOps 2강: 메시지 리스트의 총 토큰 수를 계산합니다.

    각 메시지의 content와 role 정보를 포함하여 토큰 수를 계산합니다.
    ChatGPT API와 유사한 방식으로 계산합니다.

    Args:
        messages: LangChain 메시지 리스트

    Returns:
        int: 총 토큰 수

    Note:
        메시지당 약 4토큰의 오버헤드 추가 (role, name 등)
    """
    if not messages:
        return 0

    total_tokens = 0

    # 각 메시지별 토큰 수 계산
    for message in messages:
        # 메시지 content 토큰 수
        content_tokens = count_tokens(message.content)

        # 메시지 메타데이터 오버헤드 (role 등)
        # OpenAI API 기준: 메시지당 약 4토큰 오버헤드
        overhead_tokens = 4

        total_tokens += content_tokens + overhead_tokens

    # 전체 대화 오버헤드 (시작/끝 토큰)
    total_tokens += 2

    return total_tokens


def estimate_cost(
    input_tokens: int, output_tokens: int, model: str = "solar-pro2"
) -> float:
    """
    🆕 LLMOps 2강: 토큰 사용량 기반 비용을 추정합니다.

    모델별 토큰당 가격을 기반으로 비용을 계산합니다.
    가격은 공식 문서 기준이며, 실제 가격과 다를 수 있습니다.

    Args:
        input_tokens: 입력 토큰 수
        output_tokens: 출력 토큰 수
        model: 모델 이름

    Returns:
        float: 예상 비용 (USD)

    모델별 가격 (1M 토큰당, 2025년 기준):
        - solar-pro2: 입력 $0.15, 출력 $0.6
        - solar-mini: 입력 $0.15, 출력 정보 없음 (solar-pro2와 동일 적용)
        - gpt-4o-mini: 입력 $0.15, 출력 $0.6
        - gpt-4o: 입력 $5, 출력 $15
    """
    # 모델별 가격 테이블 (1K 토큰당 USD)
    # 참고: https://console.upstage.ai/docs/capabilities/chat
    # solar-pro2: Input $0.15/MTok, Output $0.6/MTok
    PRICING = {
        # Upstage Solar (다양한 포맷 지원)
        "solar-pro2": {"input": 0.00015, "output": 0.0006},  # settings.llm_model
        "solar-mini": {
            "input": 0.00015,
            "output": 0.00015,
        },  # settings.router_llm_model
        "openai/solar-pro2": {"input": 0.00015, "output": 0.0006},  # LiteLLM 포맷
        "openai/solar-mini": {"input": 0.00015, "output": 0.00015},
        "upstage/solar-pro2": {"input": 0.00015, "output": 0.0006},
        "upstage/solar-mini": {"input": 0.00015, "output": 0.00015},
        # OpenAI (참고용)
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "openai/gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "openai/gpt-4o": {"input": 0.005, "output": 0.015},
        # Gemini (Fallback용) - $0.10/MTok input, $0.40/MTok output
        "gemini-2.5-flash": {"input": 0.0001, "output": 0.0004},
        "openai/gemini-2.5-flash": {"input": 0.0001, "output": 0.0004},
        # 기본값 (solar-pro2 가격 사용)
        "default": {"input": 0.00015, "output": 0.0006},
    }

    # 모델 가격 조회 (없으면 기본값)
    pricing = PRICING.get(model, PRICING["default"])

    # 비용 계산 (1K 토큰 단위)
    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]

    total_cost = input_cost + output_cost

    logger.debug(
        f"💰 비용 추정: 입력={input_tokens}토큰(${input_cost:.6f}), "
        f"출력={output_tokens}토큰(${output_cost:.6f}), "
        f"총=${total_cost:.6f}"
    )

    return total_cost
