"""
기존 core/config.py 사용 시, 해당 파일을 통해서 핵심 설정 및 공통 모듈 설명을 했으나,
.
├── README.md
├── app
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-312.pyc
│   │   └── main.cpython-312.pyc
│   ├── core
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-312.pyc
│   │   │   └── config.cpython-312.pyc
│   │   ├── config
│   │   │   ├── __init__.py
│   │   │   ├── config_dev.py
│   │   │   └── config_prod.py
│   │   └── config.py
│   ├── main.py
│   └── schemas
│       ├── __init__.py
│       └── chat.py
├── data
├── pyproject.toml
├── result_iamges
│   └── 0223 결과.png
└── uv.lock
위처럼 환경별 설정 분리로 수정함.
"""
