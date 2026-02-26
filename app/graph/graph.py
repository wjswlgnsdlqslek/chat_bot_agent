"""
LangGraph 그래프 구성

이 모듈에서 노드와 엣지를 조합하여 완전한 그래프를 구성합니다.

그래프 구조:
    Entry -> router -> (조건부) -> rag/tool/response -> response -> END

    1. router: 의도 분류
    2. 조건부 라우팅:
       - chat -> response
       - rag -> rag -> response
       - tool -> tool -> response
    3. response: 최종 응답 생성
    4. END: 그래프 종료
"""

from langgraph.graph import StateGraph, START, END
from loguru import logger

from app.graph.state import LumiState
from app.graph.nodes import router_node, rag_node, tool_node, response_node
from app.graph.edges import route_by_intent

# 전역 그래프 인스턴스 (싱글톤)
_compiled_graph = None


def create_lumi_graph() -> StateGraph:
    """
    루미 에이전트 그래프를 생성하고 컴파일합니다.

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

    # TODO 1: StateGraph 빌더 생성
    builder = StateGraph(LumiState)

    # TODO 2: 노드 추가 (4개)
    builder.add_node("router", router_node)
    builder.add_node("rag", rag_node)
    builder.add_node("tool", tool_node)
    builder.add_node("response", response_node)

    # TODO 3: 진입점 설정
    builder.add_edge(START, "router")

    # TODO 4: 조건부 엣지 추가
    builder.add_conditional_edges(
        "router",
        route_by_intent,
        {
            "rag": "rag",
            "tool": "tool",
            "response": "response",
        },
    )

    # TODO 5: 일반 엣지 추가 (3개)
    builder.add_edge("rag", "response")
    builder.add_edge("tool", "response")
    builder.add_edge("response", END)

    # TODO 6: 그래프 컴파일
    compiled = builder.compile()

    logger.info("✅ LangGraph 그래프 컴파일 완료")

    return compiled


def get_lumi_graph():
    """
    싱글톤 패턴으로 컴파일된 그래프를 반환합니다.

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
