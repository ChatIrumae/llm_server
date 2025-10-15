from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
import asyncio
import json
import logging
from datetime import datetime

from services.ollama_service import OllamaService
from services.langchain_chroma_service import LangChainChromaService
from services.websocket_manager import websocket_manager

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="대학교 질의처리 Chatbot API",
    description="LLM 기반 대학교 질의응답 시스템",
    version="1.0.0"
)


ollama_service = OllamaService()
chroma_service = LangChainChromaService()

class ChatRequest(BaseModel):
    user_id: str
    message: str

@app.get("/")
async def root():
    """헬스체크 엔드포인트"""
    return {"message": "대학교 질의처리 Chatbot API가 정상 작동 중입니다."}

@app.websocket("/login")
async def websocket_connect(websocket: WebSocket):
    """사용자 로그인 시 WebSocket 연결을 수립하는 엔드포인트"""
    try:
        # 연결 요청에서 user_id와 user_info를 받음
        data = await websocket.receive_text()
        message_data = json.loads(data)
        user_id = message_data.get("user_id")
        username = message_data.get("username", "")
        user_info = message_data.get("user_info", {})
        
        if not user_id:
            await websocket.close(code=1008, reason="user_id가 필요합니다.")
            return
        
        # WebSocket 연결 수립 (user_info 포함)
        await websocket_manager.connect(websocket, user_id, user_info)
        
        # 연결 유지를 위한 루프
        while True:
            try:
                # ping 메시지 대기 (연결 상태 확인용)
                data = await websocket.receive_text()
                ping_data = json.loads(data)
                
                if ping_data.get("type") == "ping":
                    await websocket_manager.send_personal_message({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }, user_id)
                    
            except WebSocketDisconnect:
                break
                
    except WebSocketDisconnect:
        logger.info(f"사용자 {user_id}의 WebSocket 연결 해제")
    except Exception as e:
        logger.error(f"WebSocket 연결 처리 중 오류: {str(e)}")
    finally:
        # 연결 해제 처리
        if 'user_id' in locals():
            websocket_manager.disconnect(user_id)

@app.post("/chat")
async def chat_endpoint(chat_request: ChatRequest):
    """채팅 요청을 처리하는 엔드포인트"""
    user_id = chat_request.user_id
    user_message = chat_request.message
    
    # WebSocket 연결 상태 확인
    if not websocket_manager.is_connected(user_id):
        raise HTTPException(
            status_code=400, 
            detail="WebSocket 연결이 필요합니다. 먼저 /login에 연결해주세요."
        )
    
    if not user_message:
        await websocket_manager.send_error(user_id, "메시지가 비어있습니다.")
        raise HTTPException(status_code=400, detail="메시지가 비어있습니다.")
    
    logger.info(f"사용자 {user_id}의 채팅 요청: {user_message}")
    
    try:
        # 1. Chroma DB 검색을 백그라운드에서 시작 (병렬 처리)
        chroma_task = asyncio.create_task(
            chroma_service.search_documents(user_message)
        )
        
        # 2. Ollama 3B로 답변 구조 생성 (스트리밍) 및 전체 답변 저장
        # 저장된 사용자 정보 가져오기
        stored_user_info = websocket_manager.get_user_info(user_id)
        streamed_response = await ollama_service.stream_response(
            websocket_manager.active_connections[user_id], 
            user_message, 
            model="llama3.2:3b",
            user_info=stored_user_info
        )
        
        # 3. Chroma DB 검색 결과 대기 (이미 백그라운드에서 실행 중)
        chroma_results = await chroma_task
        
        # 4. 같은 3B 모델로 플레이스홀더 매핑 (모델 재사용)
        if chroma_results and streamed_response:
            await ollama_service.map_placeholders(
                websocket_manager.active_connections[user_id], 
                streamed_response, 
                chroma_results, 
                model="llama3.2:3b"
            )
        
        await websocket_manager.send_personal_message({
            "type": "complete",
            "message": "응답 완료",
            "timestamp": datetime.now().isoformat()
        }, user_id)
        
        return {"status": "success", "message": "채팅 응답이 전송되었습니다."}
        
    except Exception as e:
        error_msg = f"채팅 처리 중 오류: {str(e)}"
        logger.error(error_msg)
        await websocket_manager.send_error(user_id, error_msg)
        raise HTTPException(status_code=500, detail=error_msg)




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
