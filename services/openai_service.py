"""
OpenAI API 서비스 모듈
OpenAI GPT-4o 모델과의 통신을 담당
"""

import asyncio
import json
import logging
from typing import AsyncGenerator, Dict, Any, Optional
import openai
from fastapi import WebSocket

from models.chat_models import StreamingResponse
from prompts.chat_prompts import ChatPrompts
from prompts.prompt_config import PromptConfig

logger = logging.getLogger(__name__)

class OpenAIService:
    """OpenAI API 서비스 클래스"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-5"):
        self.model = model
        self.client = openai.AsyncOpenAI(api_key=api_key)
        
    
    
    async def stream_response_with_context(self, websocket: WebSocket, message: str, chroma_results: list, user_info: Dict[str, Any] = None, location_service = None) -> str:
        """
        RAG 결과를 포함한 통합 프롬프트로 스트리밍 응답 생성
        """
        try:
            # RAG 프롬프트 생성
            rag_prompt = ChatPrompts.build_rag_prompt(message, chroma_results, user_info)
            
            logger.info(f"OpenAI {self.model} 통합 스트리밍 요청 시작: {message[:50]}...")
            
            full_response = ""  # 전체 응답을 저장할 변수
            
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "당신은 대학교 학사 관련 질의를 처리하는 AI 어시스턴트입니다. 제공된 문서 정보를 바탕으로 정확하고 구체적인 답변을 제공해주세요."},
                    {"role": "user", "content": rag_prompt}
                ],
                temperature=0.7,
                max_tokens=2000,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    
                    # 토큰을 실시간으로 전송
                    await websocket.send_text(json.dumps({
                        "type": "token",
                        "content": content,
                        "metadata": {"model": self.model}
                    }))
            
            logger.info(f"OpenAI {self.model} 통합 스트리밍 완료: {len(full_response)} 문자")
            
            # 위치 관련 액션 처리
            if location_service:
                try:
                    map_action = await location_service.process_location_query(message, full_response)
                    if map_action:
                        await websocket.send_text(json.dumps(map_action))
                except Exception as e:
                    logger.error(f"위치 액션 처리 중 오류: {str(e)}")
            
            return full_response
            
        except Exception as e:
            logger.error(f"OpenAI 통합 스트리밍 중 오류: {str(e)}")
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": f"스트리밍 오류: {str(e)}"
            }))
            return ""  # 오류 시 빈 문자열 반환
    
    
    async def check_model_status(self) -> Dict[str, Any]:
        """
        OpenAI 모델 상태 확인
        """
        try:
            # 간단한 테스트 요청으로 모델 상태 확인
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            
            return {
                "available": True,
                "model": self.model,
                "status": "ready"
            }
            
        except Exception as e:
            logger.error(f"모델 상태 확인 중 오류: {str(e)}")
            return {
                "available": False,
                "model": self.model,
                "status": "error",
                "error": str(e)
            }
    
    async def close(self):
        """클라이언트 연결 종료"""
        # OpenAI 클라이언트는 별도 종료 메서드가 없음
        pass
