# 대학교 질의처리 Chatbot LLM 시스템

대학교 학사 관련 질의응답을 위한 LLM 기반 Chatbot 시스템입니다.

## 🏗️ 시스템 아키텍처

### 핵심 구성요소
- **LangChain 서버**: FastAPI 기반의 메인 서버
- **Ollama 3B**: Llama 3.2 3B 모델 (구조화된 답변 생성)
- **Ollama 1B**: Llama 3.2 1B 모델 (플레이스홀더 매핑)
- **Chroma DB**: 벡터 데이터베이스 (문서 검색)

### 워크플로우
1. 사용자 질의 수신
2. 동시 처리:
   - Ollama 3B: 구조화된 답변 생성 (플레이스홀더 포함)
   - Chroma DB: 관련 문서 검색
3. 스트리밍으로 답변 전송
4. Ollama 1B: 플레이스홀더를 실제 값으로 매핑
5. 최종 답변 완성 및 전송

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

### 3. Ollama 모델 설치
```bash
# 모델 설치 스크립트 실행
chmod +x scripts/setup_models.sh
./scripts/setup_models.sh
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
const ws = new WebSocket('ws://localhost/ws/chat');

ws.onopen = function() {
    ws.send(JSON.stringify({
        message: "학사 경고 기준이 뭐야?"
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
- Ollama 3B 서버: `http://ollama-3b:11434`
- Ollama 1B 서버: `http://ollama-1b:11434`
- Chroma DB: `http://chroma:8000`

## 📁 프로젝트 구조

```
chatirumae/
├── main.py                 # FastAPI 메인 서버
├── requirements.txt        # Python 의존성
├── Dockerfile             # LangChain 서버 컨테이너
├── docker-compose.yml     # 전체 서비스 오케스트레이션
├── nginx.conf             # 리버스 프록시 설정
├── models/
│   ├── __init__.py
│   └── chat_models.py     # 데이터 모델 정의
├── services/
│   ├── __init__.py
│   ├── ollama_service.py  # Ollama API 서비스
│   ├── chroma_service.py  # Chroma DB 서비스
│   └── response_processor.py # 응답 처리 로직
└── scripts/
    ├── setup_models.sh    # 모델 설치 스크립트
    └── init_chroma.py     # Chroma DB 초기화
```

## 🔍 주요 기능

### 1. 스마트 질의 처리
- 자연어 질문을 구조화된 답변으로 변환
- 대학교 학사 규정에 특화된 응답 생성

### 2. 실시간 스트리밍
- WebSocket을 통한 실시간 응답 스트리밍
- 사용자 경험 향상을 위한 점진적 응답 제공

### 3. 지식 베이스 검색
- Chroma DB를 통한 관련 문서 검색
- 벡터 유사도 기반 정확한 정보 제공

### 4. 플레이스홀더 매핑
- 구조화된 답변의 플레이스홀더를 실제 값으로 대체
- 더 정확하고 구체적인 정보 제공

## ⚙️ 설정 옵션

### 환경 변수
- `LANGCHAIN_HOST`: 서버 호스트 (기본값: 0.0.0.0)
- `LANGCHAIN_PORT`: 서버 포트 (기본값: 80)
- `OLLAMA_3B_URL`: Ollama 3B 서버 URL
- `OLLAMA_1B_URL`: Ollama 1B 서버 URL
- `CHROMA_HOST`: Chroma DB 호스트
- `CHROMA_PORT`: Chroma DB 포트

### 모델 설정
- 3B 모델: 구조화된 답변 생성용
- 1B 모델: 플레이스홀더 매핑용
- GPU 가속 지원 (NVIDIA Docker 필요)

## 🐛 문제 해결

### 일반적인 문제들

1. **Ollama 모델이 로드되지 않는 경우**
   ```bash
   # 모델 상태 확인
   docker exec ollama-3b ollama list
   docker exec ollama-1b ollama list
   
   # 모델 재설치
   ./scripts/setup_models.sh
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
docker-compose logs -f langchain-server
docker-compose logs -f ollama-3b
docker-compose logs -f ollama-1b
docker-compose logs -f chroma
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
