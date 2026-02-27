# 전지훈 실습

## 실행

```bash
uv run uvicorn app.main:app --reload
```

## 1일차

### 학습 목표:

1. FastAPI와 pydantic-settings를 사용하여 설정 관리 시스템을 구축한다.
2. 환경변수 기반 설정과 타입 안전성을 이해한다.
3. FastAPI의 기본 앱 구조를 학습한다.

### 미션 개요:

1. 문제 상황: AI 챗봇 "Lumi Agent"를 개발하려고 한다. 프로젝트를 시작하기 전에 확장 가능하고 유지보수하기 쉬운 기반 구조가 필요하다.
2. 목적: pydantic-settings를 활용한 설정 관리와 FastAPI 기본 앱 구조를 완성한다.

### 미션 수행:

| 단계   | 수행 내용                    | 예상 시간 |
| ------ | ---------------------------- | --------- |
| Step 1 | 환경 설정(uv sync, env 설정) | 10분      |
| Step 2 | config.py 구현               | 60분      |
| Step 3 | main.py 구현                 | 30분      |
| Step 4 | schemas/chat.py 구현         | 15분      |
| Step 5 | 테스트 및 검증               | 15분      |

### 체크리스트

1. config.py
   - environment 필드가 Literal 타입으로 정의되어 있는가?
   - 모든 필드에 적절한 타입 힌트와 기본값이 있는가?
   - model_config가 SettingsConfigDict로 설정되어 있는가?
   - get_settings() 함수에 @lru_cache 데코레이터가 있는가?
2. main.py
   - FastAPI 앱에 title, version, lifespan이 설정되어 있는가?
   - CORS 미들웨어가 추가되어 있는가?
   - 루트 엔드포인트가 올바른 JSON을 반환하는가?
3. schemas/chat.py
   - ChatResponse에 message 필드가 정의되어 있는가?
   - tool_used, cached, timestamp 필드가 올바르게 정의되어 있는가?
   - 각 필드에 Field()를 사용하여 description이 있는가?

- 실행 확인
  - `uv run uvicorn app.main:app --reload`로 서버가 실행되는가?
  - `http://localhost:8000/`에 접속이 되는가? -`http://localhost:8000/docs`에서 Swagger UI가 표시되는가?

### 심화 미션

1. 환경별 설정 분리
   - `config_dev.py`, `config_prod.py`로 환경별 설정 파일 분리
   - 환경변수 `ENVIRONMENT`에 따라 다른 설정 로드
2. 추가 엔드포인트
   - `/health` 헬스체크 엔드포인트 추가
   - `/info` 서버 정보 반환 엔드포인트 추가

## 2일차

### 학습 목표:

1. LangGraph의 State, Node, Edge 개념을 이해하고 구현한다.
2. 조건부 라우팅(Conditional Edge)으로 워크플로우 분기를 구현한다.
3. Tool Calling 패턴을 활용하여 외부 기능을 연동한다.

### 미션 개요:

1. 문제 상황: Day 1에서 만든 FastAPI 서버에 AI 챗봇 기능을 추가해야 한다. 사용자 의도에 따라 다른 처리 로직을 실행하는 "에이전트" 구조가 필요하다.
2. 목적: LangGraph를 활용하여 Router → RAG/Tool → Response 흐름을 갖춘 에이전트를 구현한다.

### 미션 수행:

| 단계   | 수행 내용                     | 예상 시간 |
| ------ | ----------------------------- | --------- |
| Step 1 | 환경 설정(uv sync, .env 파일) | 10분      |
| Step 2 | state.py 구현                 | 60분      |
| Step 3 | nodes.py 구현                 | 30분      |
| Step 4 | edges.py 구현                 | 15분      |
| Step 5 | graph.py 구현                 | 15분      |
| Step 6 | chat.py 구현                  | 15분      |
| Step 7 | schedule.py 구현              | 15분      |
| Step 8 | 테스트 및 검증                | 15분      |

### 체크리스트

1. state.py
   - messages 필드에 add_messages 리듀서가 적용되어 있는가?
   - intent 필드가 Literal 타입으로 정의되어 있는가?
   - 모든 필드에 적절한 타입 힌트가 있는가?
2. edges.py
   - route_by_intent 함수가 intent에 따라 올바른 노드를 반환하는가?
   - chat intent일 때 "response"를 반환하는가?
3. graph.py
   - 4개 노드가 모두 추가되어 있는가? (router, rag, tool, response)
   - 조건부 엣지가 router 노드에서 올바르게 분기되는가?
   - response 노드가 END로 연결되어 있는가?
4. nodes.py
   - router_node가 with_structured_output을 사용하는가?
   - rag_node가 filter_status="active"로 검색하는가?
   - response_node가 AIMessage를 반환하는가?
5. chat.py
   - 초기 상태에 HumanMessage가 포함되어 있는가?
   - 그래프 실행 후 마지막 메시지를 추출하는가?
6. repositories/schedule.py
   - client가 None일 때 ValueError를 발생시키는가?
   - 연결 성공 시 로그를 출력하는가?
7. 실행 확인
   - 서버가 정상 실행되는가?
   - `/api/v1/chat/` 엔드포인트가 응답하는가?
   - `/ui` Gradio UI가 표시되는가?

### 심화 미션(구현완료)

1. 대화 히스토리 구현
   - 세션별 대화 기록 저장
   - 이전 대화 컨텍스트를 LLM에 전달
2. 에러 핸들링 강화
   - 노드별 타임아웃 처리
   - 실패 시 폴백 응답 구현

## 3일차

### 학습 목표:

1. SSE(Server-Sent Events)를 이용한 실시간 응답 전송을 구현한다.
2. LangGraph의 astream 메서드와 stream_mode를 이해한다.
3. 토큰 스트리밍으로 ChatGPT 같은 UX를 구현한다.

### 미션 개요:

1. 문제 상황: 에이전트가 응답하는 데 3~5초가 걸린다. 사용자는 그동안 빈 화면만 봐야 할까?
2. 목적: 진행 상황을 실시간으로 보여주고, 토큰 단위로 글자가 나타나는 스트리밍 UX를 구현한다.

### 미션 수행:

| 단계   | 수행 내용                    | 예상 시간 |
| ------ | ---------------------------- | --------- |
| Step 1 | 환경 설정(uv sync, env 설정) | 5분       |
| Step 2 | schema/chat.py 구현          | 20분      |
| Step 3 | api/routes/chat.py 구현      | 70분      |
| Step 4 | 테스트 및 검증               | 20분      |

### 체크리스트

1. schemas/chat.py
   - StreamEventType이 Literal로 정의되어 있는가?
   - StreamEvent에 type, content 필드가 있는가?
   - to_sse() 메서드가 `data: {...}\n\n` 형식을 반환하는가?
2. chat.py (stream_with_status)
   - stream_mode=["updates", "messages"]로 두 모드를 동시에 사용하는가?
   - updates 모드에서 노드 상태를 yield하는가?
   - messages 모드에서 토큰을 yield하는가?
3. chat.py (chat_stream)
   - StreamingResponse를 반환하는가?
   - media_type="text/event-stream"인가?
   - done 이벤트로 스트리밍을 종료하는가?

- 실행 확인
  - `/api/v1/chat/stream` 엔드포인트가 동작하는가?

### 심화 미션

1. 토큰 카운트 표시- 스트리밍 중 생성된 토큰 수를 UI에 표시
   - 예: "42 tokens generated"
2. 취소 기능 구현- 스트리밍 중 사용자가 취소할 수 있는 기능

## 4일차

### 학습 목표:

1. GitHub Actions를 사용하여 CI 파이프라인을 구축한다.
2. Pre-commit 훅으로 로컬에서 코드 품질을 관리한다.
3. pytest로 API 엔드포인트를 테스트한다.

### 미션 개요:

1. 문제 상황: 코드를 수정할 때마다 수동으로 린트 검사와 테스트를 실행해야 한다. PR마다 리뷰어가 같은 실수를 반복적으로 지적하고 있다.
2. 목적: 자동화된 CI 파이프라인으로 코드 품질을 보장한다.

### 환경 설정

```bash
# 의존성 설치 (개발 도구 포함)
uv sync --extra dev

# Pre-commit 훅 설치
uv run pre-commit install

# 환경변수 설정
cp .env.example .env
```

### 미션 수행:

| 단계   | 수행 내용                                                                                       | 예상 시간 |
| ------ | ----------------------------------------------------------------------------------------------- | --------- |
| Step 1 | 환경 설정(uv sync, env 설정)                                                                    | 5분       |
| Step 2 | ci.yml 구현(1. 트리거 이벤트 설정 (main 브랜치 push/PR) -> 2.lint job 작성 -> 3. test job 작성) | 60분      |
| Step 3 | GitHub에 Push 및 CI 실행 확인                                                                   | 60분      |
| Step 4 | 테스트 및 검증                                                                                  | 20분      |

### 체크리스트

1. ci.yml
   - push, pull_request 트리거가 설정되어 있는가?
   - lint → test 순서로 실행되는가? (needs: lint)
   - Secrets를 사용하여 API 키를 전달하는가?

- GitHub Actions CI 실행 확인
  - GitHub의 브랜치에 Push -> Pull Request -> Actions 탭에서 워크플로우가 실행되는가?
  - lint job이 성공적으로 통과하는가?
  - test job이 성공적으로 통과하는가?
  - 전체 워크플로우가 녹색 체크로 완료되는가?

### 실행

```bash
# 테스트 실행
uv run pytest tests/ -v

# 린트 검사
uv run ruff check app/ tests/

# Pre-commit 전체 실행
uv run pre-commit run --all-files
```

### 심화 미션

1. pytest-cov 연동
2. CI에서 커버리지 리포트 생성

## 5일차

### 학습 목표:

1. Docker를 사용하여 애플리케이션을 컨테이너화합니다.
2. Docker Compose로 컨테이너 환경을 구성합니다.
3. GitHub Actions CD 파이프라인으로 EC2에 자동 배포합니다.

### 미션 개요:

1. 문제 상황: 로컬에서 잘 작동하던 애플리케이션이 서버에서는 동작하지 않습니다. "내 컴퓨터에서는 되는데..."라는 말이 반복됩니다.
2. 목적: Docker 컨테이너로 일관된 실행 환경을 보장하고, CD 파이프라인으로 자동 배포를 구현합니다.

### 미션 수행:

| 단계   | 수행 내용                          | 예상 시간 |
| ------ | ---------------------------------- | --------- |
| Step 1 | 환경 설정(uv sync, env 설정)       | 5분       |
| Step 2 | Dockerfile 작성                    | 30분      |
| Step 3 | docker-compose.yml 작성            | 30분      |
| Step 4 | healty.py 헬스체크 엔드포인트 작성 | 10분      |
| Step 5 | cd.yml CD 파이프라인 작성          | 30분      |
| Step 6 | Docker 빌드 및 EC2 배포 테스트     | 30분      |

### 체크리스트

1. Dockerfile
   - 멀티스테이지 빌드를 사용하는가? (builder → runtime)
   - uv를 사용하여 의존성을 설치하는가?
   - non-root 유저로 실행하는가?
   - HEALTHCHECK가 설정되어 있는가?
2. docker-compose.yml
   - 이미지와 빌드 설정이 되어 있는가?
   - 포트 매핑이 올바른가? (8000:8000)
   - 환경변수가 올바르게 전달되는가?
   - 헬스체크와 재시작 정책이 설정되어 있는가?
3. health.py
   - APIRouter가 생성되어 있는가?
   - GET / 엔드포인트가 구현되어 있는가?
   - 응답에 status, timestamp, version이 포함되어 있는가?
4. cd.yml
   - push, workflow_dispatch 트리거가 설정되어 있는가?
   - GHCR 로그인 설정이 되어 있는가?
   - 이미지 빌드 & Push 설정이 되어 있는가?

- 실행 확인
  - `docker build -t lumi-agent .`가 성공하는가?
  - `docker-compose up`으로 서버가 정상 시작하는가?
  - `curl http://localhost:8000/api/v1/health/`가 200을 반환하는가?

### 심화 미션

1. 멀티 환경 배포- staging, production 환경 분리
   - 환경별 docker-compose 파일 작성
