"""
LiteLLM Router를 LangChain ChatModel처럼 사용할 수 있는 래퍼

LiteLLM Router는 다양한 LLM 프로바이더를 통합된 인터페이스로 제공합니다.
이 모듈은 LiteLLM Router의 장점을 LangChain 생태계에서 사용할 수 있게 해줍니다.

주요 기능:
    1. Retry 전략: 일시적 오류 시 자동 재시도 (num_retries=3)
    2. Fallback 전략: Router의 fallbacks 설정으로 Upstage 실패 시 Gemini로 자동 전환
    3. Timeout 설정: 30초 타임아웃으로 무한 대기 방지
    4. LangChain 호환: ChatUpstage와 동일한 인터페이스

사용법:
    from app.core.llm import get_llm

    # 환경변수에 따라 ChatUpstage 또는 LiteLLM 래퍼 반환
    llm = get_llm()
    response = await llm.ainvoke([HumanMessage(content="안녕!")])

아키텍처 (Router 기반):
    ┌─────────────────────────────────────────────────────────────┐
    │                    LiteLLM Router                           │
    │  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐   │
    │  │ model_list  │ --> │  fallbacks  │ --> │   Retry     │   │
    │  │ (설정 정의)   │     │  (전환 규칙)   │     │ (재시도)     │   │
    │  └─────────────┘     └─────────────┘     └─────────────┘   │
    └─────────────────────────────────────────────────────────────┘
              │                                        │
              ▼                                        ▼
    ┌─────────────────┐                    ┌─────────────────┐
    │  Upstage Solar  │   실패 시 전환 →      │     Gemini      │
    │   (Primary)     │                    │   (Fallback)    │
    └─────────────────┘                    └─────────────────┘
"""

import json
from collections.abc import AsyncIterator
from typing import Any, TypeVar

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_upstage import ChatUpstage
from loguru import logger
from pydantic import BaseModel

from app.core.config import settings

# Pydantic 스키마 타입 변수
T = TypeVar("T", bound=BaseModel)

# 모듈 레벨 Router 캐시 (콜백 누적 방지)
# key: (model, fallback_model) → value: Router 인스턴스
_router_cache: dict[tuple[str, str], Any] = {}


class LiteLLMChatModel(BaseChatModel):
    """
    LiteLLM Router를 LangChain ChatModel처럼 사용하는 래퍼 클래스

    LiteLLM Router를 사용하여 Retry/Fallback 기능을 제공합니다.
    Router는 model_list에 정의된 모델들 간의 전환을 지원합니다.

    특징:
        - Router: model_list로 여러 모델 설정, fallbacks로 전환 규칙 정의
        - num_retries: 3번 재시도 (일시적 오류 대응)
        - fallbacks: Upstage(primary) 실패 시 Gemini(fallback)로 자동 전환
        - timeout: 30초 타임아웃
        - streaming: 스트리밍 응답 지원

    예시:
        llm = LiteLLMChatModel()
        response = await llm.ainvoke([HumanMessage(content="안녕!")])
        print(response.content)  # "안녕하세요! 반가워요~"
    """

    # LiteLLM 설정
    model: str = "openai/solar-pro2"  # Primary 모델 (OpenAI 호환 모드)
    api_base: str = "https://api.upstage.ai/v1/solar"  # Upstage 엔드포인트
    num_retries: int = 3  # 재시도 횟수
    timeout: int = 30  # 타임아웃 (초)
    fallback_model: str = "openai/gemini-2.5-flash"  # 폴백 모델 (OpenAI 호환)
    fallback_api_base: str = "https://generativelanguage.googleapis.com/v1beta/openai/"

    @property
    def _llm_type(self) -> str:
        """LangChain에서 사용하는 LLM 타입 식별자"""
        return "litellm-router"

    def _get_router(self):
        """
        LiteLLM Router 인스턴스 반환 (모듈 레벨 싱글톤)

        Router는 model_list에 정의된 모델들과 fallbacks 규칙을 관리합니다.
        - primary: Upstage Solar (model_name="primary")
        - fallback: Gemini (model_name="fallback")
        - fallbacks=[{"primary": ["fallback"]}]: primary 실패 시 fallback으로 전환

        콜백 누적 방지를 위해 모듈 레벨 캐시 사용
        - 동일한 (model, fallback_model) 조합은 Router를 재사용
        - LiteLLM 내부 콜백이 MAX_CALLBACKS(30) 초과하는 문제 해결
        """
        global _router_cache

        # 캐시 키: (primary_model, fallback_model)
        cache_key = (self.model, self.fallback_model)

        if cache_key in _router_cache:
            return _router_cache[cache_key]

        from litellm import Router
        from litellm.router import RetryPolicy

        # TODO 1: RetryPolicy 설정하기
        retry_policy = RetryPolicy(
            AuthenticationErrorRetries=3,
            BadRequestErrorRetries=0,
        )

        # TODO 2: Primary 모델 설정 (model_list)
        model_list = [
            {
                "model_name": "primary",
                "litellm_params": {
                    "model": self.model,
                    "api_key": settings.upstage_api_key,
                    "api_base": self.api_base,
                    "timeout": self.timeout,
                },
            }
        ]  # 이 리스트에 Primary 모델 설정을 추가해주세요

        # TODO 3: Fallback 모델 설정 및 fallbacks 규칙
        fallbacks = None
        if settings.gemini_api_key:
            model_list.append(
                {
                    "model_name": "fallback",
                    "litellm_params": {
                        "model": self.fallback_model,
                        "api_key": settings.gemini_api_key,
                        "api_base": self.fallback_api_base,
                        "timeout": self.timeout,
                    },
                }
            )
            fallbacks = [{"primary": ["fallback"]}]  # primary 실패 시 fallback으로 전환
            logger.debug("Fallback 모델 설정: Gemini 사용")

        # TODO 4: Router 생성
        router = Router(
            model_list=model_list,
            fallbacks=fallbacks,
            retry_policy=retry_policy,
            num_retries=self.num_retries,
        )

        # 캐시에 저장
        _router_cache[cache_key] = router
        logger.info(
            f"LiteLLM Router 초기화 완료 (models: {len(model_list)}, fallbacks: {fallbacks is not None})"
        )

        return router

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """동기 생성 (BaseChatModel 필수 구현, 실제 사용 안 함)"""
        raise NotImplementedError("Use ainvoke() instead")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        비동기 생성 메서드 (메인 메서드)

        Router의 acompletion을 사용하여 비동기로 LLM을 호출합니다.

        Retry 전략:
            - RateLimitError: 자동 재시도 (최대 3회)
            - ServiceUnavailableError: 자동 재시도
            - Timeout: 30초 초과 시 실패

        Fallback 전략:
            - Router가 primary 실패 시 fallback으로 자동 전환
            - GEMINI_API_KEY가 설정된 경우에만 활성화
        """
        # LangChain 메시지를 LiteLLM 형식으로 변환
        litellm_messages = self._convert_messages(messages)
        router = self._get_router()

        try:
            logger.debug(f"LiteLLM Router 호출: primary ({self.model})")
            logger.debug(
                f"   - Fallback: {self.fallback_model if settings.gemini_api_key else 'None'}"
            )

            # Router의 acompletion 사용 (Fallback 자동 처리)
            response = await router.acompletion(
                model="primary",  # model_list에서 정의한 model_name
                messages=litellm_messages,
            )

            # 실제 사용된 모델 로그
            used_model = getattr(response, "model", "unknown")
            is_fallback = "gemini" in used_model.lower() if used_model else False
            if is_fallback:
                logger.warning(f"Fallback 사용: {used_model}")
            else:
                logger.info(f"Primary 사용: {used_model}")

            content = response.choices[0].message.content
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content=content,
                            response_metadata={
                                "model_name": used_model
                            },  # 실제 사용된 모델
                        )
                    )
                ]
            )

        except Exception as e:
            logger.error(f"LiteLLM Router 호출 실패: {e}")
            if "rate limit" in str(e).lower():
                raise RuntimeError(
                    "AI 서비스가 바쁩니다. 잠시 후 다시 시도해주세요."
                ) from e
            elif "timeout" in str(e).lower():
                raise RuntimeError(
                    "AI 응답 시간이 초과되었습니다. 다시 시도해주세요."
                ) from e
            else:
                raise RuntimeError(f"AI 서비스 오류: {e}") from e

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """
        스트리밍 생성 메서드 (Router 기반)

        Router의 acompletion을 stream=True로 호출하여
        토큰 단위로 응답을 스트리밍합니다.

        LangGraph의 stream_mode=["messages"]와 함께 사용하면
        실시간으로 토큰이 전달됩니다.

        동작 원리:
            1. router.acompletion(stream=True) 호출
            2. 각 청크의 delta.content를 AIMessageChunk로 변환
            3. ChatGenerationChunk로 감싸서 yield
            4. Primary 실패 시 Router가 자동으로 Fallback으로 전환
            5. 스트리밍 중 타임아웃 시 fallback으로 재시도

        사용 예시:
            async for chunk in llm._astream(messages):
                print(chunk.message.content, end="", flush=True)
        """
        # LangChain 메시지를 LiteLLM 형식으로 변환
        litellm_messages = self._convert_messages(messages)
        router = self._get_router()

        # 스트리밍 타임아웃 시 fallback 재시도를 위한 모델 리스트
        models_to_try = ["primary"]
        if settings.gemini_api_key:
            models_to_try.append("fallback")

        last_error = None

        for model_name in models_to_try:
            try:
                if model_name == "fallback":
                    logger.warning("Primary 타임아웃, Fallback으로 재시도...")

                logger.debug(
                    f"LiteLLM Router 스트리밍 호출: {model_name} ({self.model if model_name == 'primary' else self.fallback_model})"
                )

                # Router의 acompletion 사용
                response = await router.acompletion(
                    model=model_name,
                    messages=litellm_messages,
                    stream=True,
                )

                # 스트리밍 응답 처리
                used_model = None
                async for chunk in response:
                    # 첫 청크에서 실제 사용된 모델 확인
                    if used_model is None and hasattr(chunk, "model"):
                        used_model = chunk.model
                        is_fallback = (
                            "gemini" in used_model.lower() if used_model else False
                        )
                        if is_fallback:
                            logger.warning(f"Fallback 사용: {used_model}")
                        else:
                            logger.info(f"Primary 사용: {used_model}")

                    delta = chunk.choices[0].delta
                    content = delta.content if delta and delta.content else ""
                    if content:
                        chunk_msg = AIMessageChunk(content=content)
                        yield ChatGenerationChunk(message=chunk_msg)
                        if run_manager:
                            await run_manager.on_llm_new_token(content)

                logger.debug(f"LiteLLM Router 스트리밍 완료 (model: {used_model})")

                # 마지막에 모델 정보를 포함한 빈 청크 전송
                if used_model:
                    final_chunk = AIMessageChunk(
                        content="",
                        response_metadata={"model_name": used_model},
                    )
                    yield ChatGenerationChunk(message=final_chunk)

                return  # 성공하면 종료

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # 타임아웃이면 다음 모델로 시도
                if "timeout" in error_str and model_name != models_to_try[-1]:
                    logger.warning(f"{model_name} 타임아웃, 다음 모델 시도...")
                    continue

                # Rate limit이면 다음 모델로 시도
                if "rate limit" in error_str and model_name != models_to_try[-1]:
                    logger.warning(f"{model_name} Rate limit, 다음 모델 시도...")
                    continue

                # 마지막 모델이거나 다른 에러면 종료
                break

        # 모든 모델 실패
        logger.error(f"LiteLLM Router 스트리밍 실패 (모든 모델): {last_error}")
        if last_error:
            if "rate limit" in str(last_error).lower():
                raise RuntimeError(
                    "AI 서비스가 바쁩니다. 잠시 후 다시 시도해주세요."
                ) from last_error
            elif "timeout" in str(last_error).lower():
                raise RuntimeError(
                    "AI 응답 시간이 초과되었습니다. 다시 시도해주세요."
                ) from last_error
            else:
                raise RuntimeError(f"AI 서비스 오류: {last_error}") from last_error
        raise RuntimeError("AI 서비스 오류: 알 수 없는 오류")

    def _convert_messages(self, messages: list[BaseMessage]) -> list[dict]:
        """
        LangChain 메시지를 LiteLLM 형식으로 변환

        LangChain:
            [HumanMessage(content="안녕"), AIMessage(content="반가워")]

        LiteLLM:
            [{"role": "user", "content": "안녕"}, {"role": "assistant", "content": "반가워"}]
        """
        converted = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                converted.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                converted.append({"role": "assistant", "content": msg.content})
            else:
                # SystemMessage 등 기타 메시지 타입
                converted.append({"role": "system", "content": msg.content})
        return converted

    def with_structured_output(self, schema: type[T]) -> "StructuredLiteLLMRunnable[T]":
        """
        Structured Output 지원

        Pydantic 스키마를 받아서 구조화된 출력을 반환하는 Runnable을 생성합니다.
        LiteLLM의 response_format을 사용하여 JSON 응답을 강제하고,
        결과를 Pydantic 모델로 파싱합니다.

        Args:
            schema: Pydantic 모델 클래스 (예: RouterOutput)

        Returns:
            StructuredLiteLLMRunnable: ainvoke 시 Pydantic 모델 인스턴스 반환

        사용 예시:
            llm = LiteLLMChatModel()
            structured_llm = llm.with_structured_output(RouterOutput)
            result = await structured_llm.ainvoke(messages)
            # result는 RouterOutput 타입!

        Note:
            - LangChain의 with_structured_output과 동일한 인터페이스
            - 내부적으로 response_format={"type": "json_object"} 사용
        """
        return StructuredLiteLLMRunnable(llm=self, schema=schema)


class StructuredLiteLLMRunnable(Runnable[list[BaseMessage], T]):
    """
    Structured Output을 위한 Runnable 래퍼 (Router 기반)

    LiteLLMChatModel의 with_structured_output이 반환하는 클래스입니다.
    ainvoke 호출 시 JSON 응답을 Pydantic 모델로 파싱하여 반환합니다.

    동작 원리:
        1. Router의 completion/acompletion 호출 (response_format={"type": "json_object"})
        2. 응답 JSON을 Pydantic 스키마로 파싱
        3. 파싱된 Pydantic 모델 인스턴스 반환
        4. Primary 실패 시 Router가 자동으로 Fallback으로 전환
    """

    def __init__(self, llm: "LiteLLMChatModel", schema: type[T]):
        self.llm = llm
        self.schema = schema

    def invoke(self, input: list[BaseMessage], config: Any = None) -> T:
        """동기 호출 (권장하지 않음)"""
        litellm_messages = self.llm._convert_messages(input)
        router = self.llm._get_router()

        # Router의 completion 사용 (Fallback 자동 처리)
        response = router.completion(
            model="primary",  # model_list에서 정의한 model_name
            messages=litellm_messages,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        return self.schema.model_validate_json(content)

    async def ainvoke(self, input: list[BaseMessage], config: Any = None) -> T:
        """
        비동기 호출 (메인 메서드)

        Router를 사용하여 LLM에게 JSON 응답을 요청하고, Pydantic 모델로 파싱합니다.
        """
        litellm_messages = self.llm._convert_messages(input)
        router = self.llm._get_router()

        try:
            logger.debug(
                f"LiteLLM Router Structured Output 호출: primary ({self.llm.model})"
            )
            logger.debug(f"   - Schema: {self.schema.__name__}")

            # Router의 acompletion 사용 (Fallback 자동 처리)
            response = await router.acompletion(
                model="primary",  # model_list에서 정의한 model_name
                messages=litellm_messages,
                response_format={"type": "json_object"},
            )

            # 실제 사용된 모델 로그
            used_model = getattr(response, "model", "unknown")
            is_fallback = "gemini" in used_model.lower() if used_model else False
            if is_fallback:
                logger.warning(f"Fallback 사용: {used_model}")
            else:
                logger.info(f"Primary 사용: {used_model}")

            content = response.choices[0].message.content
            logger.debug(f"LiteLLM Router 응답: {content[:100]}...")

            # JSON → Pydantic 모델 파싱
            result = self.schema.model_validate_json(content)
            logger.debug(f"Pydantic 파싱 성공: {result}")

            return result

        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 실패: {e}")
            raise ValueError(f"LLM 응답이 올바른 JSON 형식이 아닙니다: {e}") from e
        except Exception as e:
            logger.error(f"Structured Output 호출 실패: {e}")
            raise


def get_llm(model: str | None = None) -> BaseChatModel:
    """
    환경변수에 따라 적절한 LLM 클라이언트 반환

    USE_LITELLM=true  -> LiteLLM 래퍼 (Retry + Fallback 지원)
    USE_LITELLM=false -> ChatUpstage (기존 방식)

    Args:
        model: 사용할 모델명 (기본값: settings.llm_model)

    Returns:
        BaseChatModel: LLM 클라이언트 (LangChain 호환)

    사용 예시:
        # 기본 모델 (solar-pro2)
        llm = get_llm()

        # 특정 모델 지정
        llm = get_llm(model="solar-mini")

    주의:
        - LiteLLM 사용 시 UPSTAGE_API_KEY 필수
        - Fallback 사용 시 GEMINI_API_KEY 필요
    """
    # 모델이 지정되지 않으면 기본값 사용
    model_name = model or settings.llm_model

    if settings.use_litellm:
        logger.info(f"LiteLLM Router 사용 (model: {model_name})")

        # Upstage는 LiteLLM에서 직접 지원 안함 → OpenAI 호환 모드 사용
        # 모델 형식: "openai/{model}" + api_base로 엔드포인트 지정
        litellm_model = f"openai/{model_name}"

        # Fallback 모델도 OpenAI 호환 모드로 변환
        # "gemini/gemini-2.5-flash" → "openai/gemini-2.5-flash"
        fallback_model = settings.litellm_fallback_model
        if fallback_model.startswith("gemini/"):
            fallback_model = f"openai/{fallback_model.split('/')[-1]}"

        return LiteLLMChatModel(
            model=litellm_model,
            api_base="https://api.upstage.ai/v1/solar",
            num_retries=settings.litellm_num_retries,
            timeout=settings.litellm_timeout,
            fallback_model=fallback_model,
        )
    else:
        logger.info(f"ChatUpstage 직접 사용 (model: {model_name})")
        return ChatUpstage(
            api_key=settings.upstage_api_key,
            model=model_name,
        )


def get_router_llm() -> BaseChatModel:
    """
    Router 노드 전용 LLM 클라이언트 반환

    의도 분류는 간단한 작업이므로 빠르고 저렴한 모델을 사용합니다.
    settings.router_llm_model (기본값: solar-mini)을 사용합니다.

    성능 비교 (테스트 기준):
        - solar-pro2: ~1.0초
        - solar-mini: ~0.4초 (2.5배 빠름!)

    Returns:
        BaseChatModel: Router용 LLM 클라이언트

    사용 예시:
        llm = get_router_llm()  # solar-mini 반환
        structured_llm = llm.with_structured_output(RouterOutput)
    """
    return get_llm(model=settings.router_llm_model)
