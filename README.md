# 대학교 질의처리 Chatbot RAG 시스템

대학교 학사 관련 질의응답을 위한 RAG(Retrieval-Augmented Generation) 기반 Chatbot 시스템입니다.

## 🏗️ 시스템 아키텍처

### 핵심 구성요소
- **FastAPI 서버**: 메인 웹 서버
- **OpenAI GPT**: GPT 모델 (답변 생성)
- **Chroma DB**: 벡터 데이터베이스 (문서 검색)
- **OpenAI Embeddings**: text-embedding-3-small 모델 (벡터 임베딩)
- **다국어 지원**: 언어 감지 및 번역 기능
- **위치 서비스**: 지도 액션 기능

### 워크플로우
1. 사용자 질의 수신
2. 다국어 지원: 외국어 질의인 경우 한국어로 번역
3. Chroma DB에서 관련 문서 검색
4. Retrieval 결과를 포함하여 OpenAI API로 질의
5. 스트리밍으로 답변 전송
6. 위치 관련 답변인 경우 지도 액션 제공

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 프로젝트 클론
git clone <repository-url>
cd chatirumae

# 환경 변수 설정
cp env.example .env
# .env 파일을 편집하여 필요한 설정 수정
```

### 2. Docker Compose로 서비스 시작
```bash
# 모든 서비스 빌드 및 시작
docker-compose up -d

# 로그 확인
docker-compose logs -f
```

### 3. OpenAI API 키 설정
```bash
# .env 파일에서 OpenAI API 키 설정
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=gpt-4
```

### 4. Chroma DB 초기화
```bash
# Chroma DB에 샘플 문서 추가
python scripts/init_chroma.py
```

## 📡 API 사용법

### API 엔드포인트
```bash
# API 문서 (Swagger UI)
# 브라우저에서 http://localhost/docs 접속
```

### WebSocket 스트리밍 채팅
```javascript
const ws = new WebSocket('ws://localhost/login');

ws.onopen = function() {
    ws.send(JSON.stringify({
        user_id: "user123",
        username: "홍길동",
        user_info: {
            major: "컴퓨터공학과",
            grade: "3학년"
        }
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log(data.type, data.content);
};
```

### API 문서
브라우저에서 `http://localhost/docs`에 접속하여 Swagger UI로 API 문서를 확인할 수 있습니다.

## 🔧 개발 환경 설정

### 로컬 개발
```bash
# Python 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 서버 실행
python main.py
```

### 필요한 외부 서비스
- OpenAI API: `https://api.openai.com/v1`
- Chroma DB: 로컬 파일 시스템 기반

## 📁 프로젝트 구조

```
chatirumae/
├── main.py                 # FastAPI 메인 서버
├── requirements.txt        # Python 의존성
├── env.example            # 환경변수 예제 파일
├── Dockerfile             # LangChain 서버 컨테이너
├── docker-compose.yml     # 전체 서비스 오케스트레이션
├── models/
│   ├── __init__.py
│   └── chat_models.py     # 데이터 모델 정의
├── services/
│   ├── __init__.py
│   ├── openai_service.py  # OpenAI API 서비스
│   ├── langchain_chroma_service.py  # LangChain Chroma 서비스
│   ├── websocket_manager.py # WebSocket 연결 관리
│   ├── language_service.py # 다국어 지원 서비스
│   └── location_service.py # 위치 서비스
├── prompts/
│   ├── __init__.py
│   ├── chat_prompts.py    # 프롬프트 템플릿
│   └── prompt_config.py   # 프롬프트 설정
└── scripts/
    └── init_chroma.py     # Chroma DB 초기화
```

## 🔍 주요 기능

### 1. RAG 기반 질의 처리
- Chroma DB에서 관련 문서 검색 후 답변 생성
- 대학교 학사 규정에 특화된 정확한 정보 제공

### 2. 다국어 지원
- 외국어 질의를 자동으로 한국어로 번역
- 언어 감지 및 번역 알림 기능

### 3. 위치 서비스
- 위치 관련 질문에 대한 지도 액션 제공
- 대학교 내 주요 건물/시설 위치 정보

### 4. 실시간 스트리밍
- WebSocket을 통한 실시간 응답 스트리밍
- 사용자 경험 향상을 위한 점진적 응답 제공

## ⚙️ 설정 옵션

### 환경 변수
- `OPENAI_API_KEY`: OpenAI API 키 (필수)
- `OPENAI_MODEL`: OpenAI 모델명 (기본값: gpt-4)
- `UNIVERSITY_LAT`: 대학교 위도 (기본값: 37.5665)
- `UNIVERSITY_LNG`: 대학교 경도 (기본값: 126.9780)
- `HOST`: 서버 호스트 (기본값: 0.0.0.0)
- `PORT`: 서버 포트 (기본값: 80)

### 모델 설정
- GPT-4: 답변 생성 및 언어 감지/번역
- text-embedding-3-small: 벡터 임베딩용
- OpenAI API 기반 클라우드 서비스

## 🔒 보안 고려사항

- 모든 API 키는 환경변수를 통해 관리
- 사용자 질의 내용은 로깅에서 제외
- 민감한 정보는 하드코딩하지 않음
- 좌표 정보도 환경변수로 설정 가능

## 🐛 문제 해결

### 일반적인 문제들

1. **OpenAI API 연결 실패**
   ```bash
   # API 키 확인
   echo $OPENAI_API_KEY
   
   # API 키 테스트
   curl -H "Authorization: Bearer $OPENAI_API_KEY" \
        https://api.openai.com/v1/models
   ```

2. **Chroma DB 연결 실패**
   ```bash
   # Chroma DB 상태 확인
   curl http://localhost:8001/api/v1/heartbeat
   
   # 컨테이너 재시작
   docker-compose restart chroma
   ```

3. **GPU 메모리 부족**
   ```bash
   # GPU 메모리 사용량 확인
   nvidia-smi
   
   # 모델 크기 조정 또는 배치 크기 감소
   ```

### 로그 확인
```bash
# 특정 서비스 로그 확인
docker-compose logs -f irumae-llm-server
```

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 📞 지원

문제가 발생하거나 질문이 있으시면 이슈를 생성해 주세요.
