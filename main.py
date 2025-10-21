from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
import asyncio
import json
import logging
from datetime import datetime

from services.openai_service import OpenAIService
from services.langchain_chroma_service import LangChainChromaService
from services.websocket_manager import websocket_manager
from services.language_service import LanguageService
from services.location_service import LocationService

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="대학교 질의처리 Chatbot API",
    description="LLM 기반 대학교 질의응답 시스템",
    version="1.0.0"
)


# OpenAI API 키 및 모델 설정 (환경변수에서 가져오기)
import os
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_model = os.getenv("OPENAI_MODEL", "gpt-5")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

openai_service = OpenAIService(api_key=openai_api_key, model=openai_model)
chroma_service = LangChainChromaService()
language_service = LanguageService(api_key=openai_api_key, model=openai_model)
location_service = LocationService(api_key=openai_api_key, model=openai_model)

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
        # 저장된 사용자 정보 가져오기
        stored_user_info = websocket_manager.get_user_info(user_id)
        
        # 1. 다국어 지원: 외국어 질의인 경우 한국어로 번역
        language_result = await language_service.process_multilingual_query(user_message)
        processed_message = language_result["processed_query"]
        
        # 번역된 경우 사용자에게 알림
        if language_result["was_translated"]:
            await websocket_manager.send_personal_message({
                "type": "translation",
                "original": language_result["original_query"],
                "translated": language_result["translated_query"],
                "message": f"질의가 {language_result['detected_language']}에서 한국어로 번역되었습니다."
            }, user_id)
        
        # 2. Chroma DB에서 검색 (번역된 질의로)
        chroma_results = await chroma_service.search_documents(processed_message)
        
        # 3. Retrieval 결과를 포함하여 OpenAI API로 질의
        response = await openai_service.stream_response_with_context(
            websocket_manager.active_connections[user_id], 
            processed_message, 
            chroma_results,
            user_info=stored_user_info,
            location_service=location_service
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
