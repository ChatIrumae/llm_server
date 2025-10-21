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
        
    async def generate_response(self, message: str, user_info: Dict[str, Any] = None) -> str:
        """
        OpenAI 모델로부터 응답 생성
        """
        try:
            # 프롬프트 생성
            prompt = ChatPrompts.build_initial_prompt(message, user_info)
            
            logger.info(f"OpenAI {self.model}에 요청 전송: {message[:50]}...")
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "당신은 대학교 학사 관련 질의를 처리하는 AI 어시스턴트입니다. 정확하고 도움이 되는 답변을 제공해주세요."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000,
                stream=False
            )
            
            generated_text = response.choices[0].message.content
            
            logger.info(f"OpenAI {self.model} 응답 수신: {len(generated_text)} 문자")
            return generated_text
            
        except Exception as e:
            logger.error(f"OpenAI 응답 생성 중 오류: {str(e)}")
            raise
    
    async def stream_response(self, websocket: WebSocket, message: str, user_info: Dict[str, Any] = None) -> str:
        """
        OpenAI 모델로부터 스트리밍 응답 생성 및 전체 답변 반환
        """
        try:
            # 프롬프트 생성
            prompt = ChatPrompts.build_initial_prompt(message, user_info)
            
            logger.info(f"OpenAI {self.model} 스트리밍 요청 시작: {message[:50]}...")
            
            full_response = ""  # 전체 응답을 저장할 변수
            
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "당신은 대학교 학사 관련 질의를 처리하는 AI 어시스턴트입니다. 정확하고 도움이 되는 답변을 제공해주세요."},
                    {"role": "user", "content": prompt}
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
            
            logger.info(f"OpenAI {self.model} 스트리밍 완료: {len(full_response)} 문자")
            return full_response
            
        except Exception as e:
            logger.error(f"OpenAI 스트리밍 중 오류: {str(e)}")
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": f"스트리밍 오류: {str(e)}"
            }))
            return ""  # 오류 시 빈 문자열 반환
    
    async def stream_response_with_context(self, websocket: WebSocket, message: str, chroma_results: list, user_info: Dict[str, Any] = None) -> str:
        """
        RAG 결과를 포함한 통합 프롬프트로 스트리밍 응답 생성
        """
        try:
            # 통합 프롬프트 생성 (RAG 결과 포함)
            integrated_prompt = ChatPrompts.build_integrated_prompt(message, chroma_results, user_info)
            
            logger.info(f"OpenAI {self.model} 통합 스트리밍 요청 시작: {message[:50]}...")
            
            full_response = ""  # 전체 응답을 저장할 변수
            
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "당신은 대학교 학사 관련 질의를 처리하는 AI 어시스턴트입니다. 제공된 문서 정보를 바탕으로 정확하고 구체적인 답변을 제공해주세요."},
                    {"role": "user", "content": integrated_prompt}
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
            return full_response
            
        except Exception as e:
            logger.error(f"OpenAI 통합 스트리밍 중 오류: {str(e)}")
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": f"스트리밍 오류: {str(e)}"
            }))
            return ""  # 오류 시 빈 문자열 반환
    
    async def map_placeholders(self, websocket: WebSocket, streamed_response: str, chroma_results: list):
        """
        플레이스홀더 매핑 (OpenAI 모델 사용)
        """
        try:
            # 프롬프트 템플릿에서 매핑 프롬프트 생성
            mapping_prompt = ChatPrompts.build_mapping_prompt(streamed_response, chroma_results)
            
            logger.info(f"OpenAI {self.model}로 플레이스홀더 매핑 시작")
            
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "당신은 문서 정보를 바탕으로 플레이스홀더를 실제 값으로 대체하는 AI 어시스턴트입니다."},
                    {"role": "user", "content": mapping_prompt}
                ],
                temperature=0.5,
                max_tokens=2000,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    
                    await websocket.send_text(json.dumps({
                        "type": "mapping",
                        "content": content,
                        "metadata": {"model": self.model}
                    }))
            
            logger.info(f"OpenAI {self.model} 플레이스홀더 매핑 완료")
            
        except Exception as e:
            logger.error(f"플레이스홀더 매핑 중 오류: {str(e)}")
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": f"매핑 오류: {str(e)}"
            }))
    
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
