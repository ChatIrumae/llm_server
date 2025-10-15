# WebSocket 연결 구조 변경 사용법

## 변경된 구조

기존에는 채팅 요청 시에만 WebSocket 연결을 했지만, 이제는 사용자 로그인 시에 WebSocket 연결을 초기화하고, 채팅은 별도 엔드포인트로 처리합니다.

## 사용 흐름

### 1. 사용자 로그인
```bash
curl -X POST "http://localhost:80/login" \
     -H "Content-Type: application/json" \
     -d '{
       "user_id": "user123",
       "username": "홍길동"
     }'
```

### 2. WebSocket 연결 수립
```javascript
// 클라이언트 측 JavaScript 예제
const ws = new WebSocket('ws://localhost:80/ws/connect');

ws.onopen = function() {
    // 로그인 정보 전송
    ws.send(JSON.stringify({
        user_id: "user123",
        username: "홍길동"
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('수신된 메시지:', data);
    
    if (data.type === 'connection_established') {
        console.log('WebSocket 연결이 성공적으로 수립되었습니다.');
    }
};

// 연결 유지를 위한 ping 전송
setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'ping' }));
    }
}, 30000); // 30초마다 ping
```

### 3. 채팅 요청
```bash
curl -X POST "http://localhost:80/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "user_id": "user123",
       "message": "안녕하세요, 대학교 입학에 대해 문의드립니다."
     }'
```

## 새로운 엔드포인트

### `/login` (WebSocket)
- **목적**: 사용자 로그인 시 WebSocket 연결 수립
- **입력**: `user_info`, `username`
- **출력**: 연결 확인 메시지

### `/chat` (POST)
- **목적**: 채팅 요청 처리
- **입력**: `ChatRequest` (user_id, message)
- **출력**: 채팅 응답이 WebSocket을 통해 전송됨

## WebSocket 메시지 타입

- `connection_established`: 연결 수립 확인
- `chat_response`: 채팅 응답 (스트리밍)
- `complete`: 응답 완료
- `error`: 오류 메시지
- `pong`: ping에 대한 응답

## 장점

1. **연결 지속성**: 로그인 후 WebSocket 연결이 유지되어 실시간 응답 가능
2. **사용자별 관리**: 각 사용자의 연결 상태를 개별적으로 관리
3. **확장성**: 다중 사용자 환경에서 효율적인 연결 관리
4. **오류 처리**: 연결 상태 확인 및 자동 재연결 지원
