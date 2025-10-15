from typing import Dict, List
from fastapi import WebSocket
import json
import logging

logger = logging.getLogger(__name__)

class WebSocketManager:
    """WebSocket 연결을 관리하는 클래스"""
    
    def __init__(self):
        # 사용자 ID별 WebSocket 연결을 저장
        self.active_connections: Dict[str, WebSocket] = {}
        # 사용자 ID별 연결 상태를 저장
        self.connection_status: Dict[str, bool] = {}
        # 사용자 ID별 사용자 정보를 저장
        self.user_info: Dict[str, dict] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str, user_info: dict = None):
        """사용자와 WebSocket 연결을 수립"""
        await websocket.accept()
        self.active_connections[user_id] = websocket
        self.connection_status[user_id] = True
        self.user_info[user_id] = user_info or {}
        logger.info(f"사용자 {user_id}와 WebSocket 연결 수립")
        
        # 연결 확인 메시지 전송
        await self.send_personal_message({
            "type": "connection_established",
            "message": "WebSocket 연결이 성공적으로 수립되었습니다.",
            "user_id": user_id
        }, user_id)
    
    def disconnect(self, user_id: str):
        """사용자의 WebSocket 연결을 해제"""
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            self.connection_status[user_id] = False
            if user_id in self.user_info:
                del self.user_info[user_id]
            logger.info(f"사용자 {user_id}의 WebSocket 연결 해제")
    
    async def send_personal_message(self, message: dict, user_id: str):
        """특정 사용자에게 메시지 전송"""
        if user_id in self.active_connections and self.connection_status[user_id]:
            try:
                await self.active_connections[user_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"사용자 {user_id}에게 메시지 전송 실패: {str(e)}")
                self.disconnect(user_id)
    
    async def send_chat_response(self, user_id: str, response_type: str, content: str):
        """채팅 응답 전송"""
        await self.send_personal_message({
            "type": "chat_response",
            "response_type": response_type,
            "content": content,
            "timestamp": self._get_timestamp()
        }, user_id)
    
    async def send_error(self, user_id: str, error_message: str):
        """에러 메시지 전송"""
        await self.send_personal_message({
            "type": "error",
            "message": error_message,
            "timestamp": self._get_timestamp()
        }, user_id)
    
    def is_connected(self, user_id: str) -> bool:
        """사용자의 연결 상태 확인"""
        return user_id in self.connection_status and self.connection_status[user_id]
    
    def get_connected_users(self) -> List[str]:
        """연결된 사용자 목록 반환"""
        return [user_id for user_id, status in self.connection_status.items() if status]
    
    def get_user_info(self, user_id: str) -> dict:
        """사용자 정보 반환"""
        return self.user_info.get(user_id, {})
    
    def _get_timestamp(self) -> str:
        """현재 시간을 문자열로 반환"""
        from datetime import datetime
        return datetime.now().isoformat()

# 전역 WebSocket 매니저 인스턴스
websocket_manager = WebSocketManager()
