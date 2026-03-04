"""
LangGraph 그래프 구성

이 모듈에서 노드와 엣지를 조합하여 완전한 그래프를 구성합니다.

🆕 LLMOps 2강 변경사항:
    - 체크포인터 통합 (대화 이어가기 지원)
    - create_lumi_graph() 체크포인터 파라미터 추가
    - get_lumi_graph_with_memory() 비동기 함수 추가
    - thread_id로 대화 세션 구분

그래프 구조:
    Entry -> router -> (조건부) -> rag/tool/response -> response -> END

    1. router: 의도 분류
    2. 조건부 라우팅:
       - chat -> response
       - rag -> rag -> response
       - tool -> tool -> response
    3. response: 최종 응답 생성
    4. END: 그래프 종료

체크포인터 동작:
    thread_id="user-123"으로 호출하면:
    1. 해당 thread의 이전 상태를 체크포인터에서 로드
    2. 새 메시지와 함께 그래프 실행
    3. 실행 결과를 체크포인터에 저장
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from loguru import logger

from app.graph.edges import route_by_intent
from app.graph.nodes import rag_node, response_node, router_node, tool_node
from app.graph.state import LumiState

# 전역 그래프 인스턴스 (싱글톤)
_compiled_graph = None
_compiled_graph_with_memory = None


def create_lumi_graph(checkpointer: BaseCheckpointSaver | None = None) -> StateGraph:
    """
    루미 에이전트 그래프를 생성하고 컴파일합니다.

    🆕 LLMOps 2강: checkpointer 파라미터 추가

    Args:
        checkpointer: 체크포인터 인스턴스 (선택)
                     None이면 체크포인터 없이 컴파일

    그래프 구조:
        ```
                      ┌─────────┐
                      │  START  │
                      └────┬────┘
                           │
                           ▼
                      ┌─────────┐
                      │ router  │
                      └────┬────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
         ┌────────┐   ┌────────┐   ┌──────────┐
         │  rag   │   │  tool  │   │ response │
         └────┬───┘   └────┬───┘   └────┬─────┘
              │            │            │
              └────────────┼────────────┘
                           │
                           ▼
                      ┌─────────┐
                      │ response│ (rag/tool에서 온 경우)
                      └────┬────┘
                           │
                           ▼
                      ┌─────────┐
                      │   END   │
                      └─────────┘
        ```

    Returns:
        CompiledStateGraph: 컴파일된 LangGraph 그래프
    """
    logger.info("🔧 LangGraph 그래프 생성 시작")

    # Step 1: StateGraph 빌더 생성
    builder = StateGraph(LumiState)

    # Step 2: 노드 추가
    # 각 노드는 (이름, 함수) 형태로 등록
    builder.add_node("router", router_node)
    builder.add_node("rag", rag_node)
    builder.add_node("tool", tool_node)
    builder.add_node("response", response_node)

    logger.debug("노드 추가 완료: router, rag, tool, response")

    # Step 3: 진입점 설정 (LangGraph 1.0 스타일)
    # START에서 router 노드로 연결
    builder.add_edge(START, "router")

    # Step 4: 조건부 엣지 추가
    # router 노드 이후, intent에 따라 다른 노드로 분기
    builder.add_conditional_edges(
        source="router",  # 출발 노드
        path=route_by_intent,  # 라우팅 함수
        path_map={  # 반환값 -> 목적지 노드 매핑
            "rag": "rag",
            "tool": "tool",
            "response": "response",
        },
    )

    logger.debug("조건부 엣지 추가 완료: router -> (rag/tool/response)")

    # Step 5: 일반 엣지 추가
    # rag/tool 노드 이후에는 response 노드로 이동
    builder.add_edge("rag", "response")
    builder.add_edge("tool", "response")

    # response 노드 이후에는 그래프 종료
    builder.add_edge("response", END)

    logger.debug("일반 엣지 추가 완료: rag->response, tool->response, response->END")

    # Step 6: 그래프 컴파일
    # 🆕 LLMOps 2강: 체크포인터가 있으면 포함하여 컴파일
    if checkpointer:
        compiled = builder.compile(checkpointer=checkpointer)
        logger.info("✅ LangGraph 그래프 컴파일 완료 (체크포인터 활성화)")
    else:
        compiled = builder.compile()
        logger.info("✅ LangGraph 그래프 컴파일 완료 (체크포인터 없음)")

    return compiled


def get_lumi_graph():
    """
    싱글톤 패턴으로 컴파일된 그래프를 반환합니다.
    (체크포인터 없는 버전 - 하위 호환성)

    그래프 컴파일은 비용이 있는 작업이므로,
    한 번 컴파일된 그래프를 재사용합니다.

    Returns:
        CompiledStateGraph: 컴파일된 LangGraph 그래프

    Example:
        >>> graph = get_lumi_graph()
        >>> result = await graph.ainvoke(initial_state)
    """
    global _compiled_graph

    if _compiled_graph is None:
        _compiled_graph = create_lumi_graph()

    return _compiled_graph


async def get_lumi_graph_with_memory():
    """
    🆕 LLMOps 2강: 체크포인터가 포함된 그래프를 반환합니다.

    체크포인터를 사용하면 thread_id로 대화를 이어갈 수 있습니다.
    설정에 따라 MemorySaver, PostgreSQL 중 선택됩니다.

    Returns:
        CompiledStateGraph: 체크포인터가 포함된 컴파일된 그래프

    Example:
        >>> graph = await get_lumi_graph_with_memory()
        >>> config = {"configurable": {"thread_id": "user-123"}}
        >>> result = await graph.ainvoke(initial_state, config=config)
    """
    global _compiled_graph_with_memory

    if _compiled_graph_with_memory is None:
        # 체크포인터 초기화
        from app.core.checkpointer import get_checkpointer

        checkpointer = await get_checkpointer()

        # 그래프 생성 및 컴파일
        _compiled_graph_with_memory = create_lumi_graph(checkpointer=checkpointer)

    return _compiled_graph_with_memory
