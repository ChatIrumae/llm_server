"""
Ollama 서비스 모듈
Llama 3.2 3B와 1B 모델과의 통신을 담당
"""

import asyncio
import json
import logging
from typing import AsyncGenerator, Dict, Any, Optional
import httpx
from fastapi import WebSocket

from models.chat_models import OllamaResponse, StreamingResponse
from prompts.chat_prompts import ChatPrompts
from prompts.prompt_config import PromptConfig

logger = logging.getLogger(__name__)

class OllamaService:
    """Ollama API 서비스 클래스"""
    
    def __init__(self, base_url: str = "http://ollama-3b:11434"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def generate_response(self, message: str, model: str = "llama3.2:3b", user_info: Dict[str, Any] = None) -> str:
        """
        Ollama 모델로부터 응답 생성
        """
        try:
            url = f"{self.base_url}/api/generate"
            
            payload = {
                "model": model,
                "prompt": ChatPrompts.build_initial_prompt(message, user_info),
                "stream": False,
                "options": {
                    "temperature": PromptConfig.INITIAL_TEMPERATURE,
                    "top_p": PromptConfig.INITIAL_TOP_P,
                    "max_tokens": PromptConfig.INITIAL_MAX_TOKENS
                }
            }
            
            logger.info(f"Ollama {model}에 요청 전송: {message[:50]}...")
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get("response", "")
            
            logger.info(f"Ollama {model} 응답 수신: {len(generated_text)} 문자")
            return generated_text
            
        except Exception as e:
            logger.error(f"Ollama 응답 생성 중 오류: {str(e)}")
            raise
    
    async def stream_response(self, websocket: WebSocket, message: str, model: str = "llama3.2:3b", user_info: Dict[str, Any] = None) -> str:
        """
        Ollama 모델로부터 스트리밍 응답 생성 및 전체 답변 반환
        """
        try:
            url = f"{self.base_url}/api/generate"
            
            payload = {
                "model": model,
                "prompt": ChatPrompts.build_initial_prompt(message, user_info),
                "stream": True,
                "options": {
                    "temperature": PromptConfig.INITIAL_TEMPERATURE,
                    "top_p": PromptConfig.INITIAL_TOP_P,
                    "max_tokens": PromptConfig.INITIAL_MAX_TOKENS
                }
            }
            
            logger.info(f"Ollama {model} 스트리밍 요청 시작: {message[:50]}...")
            
            full_response = ""  # 전체 응답을 저장할 변수
            
            async with self.client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if data.get("response"):
                                token = data["response"]
                                full_response += token  # 전체 응답에 토큰 추가
                                
                                # 토큰을 실시간으로 전송
                                await websocket.send_text(json.dumps({
                                    "type": "token",
                                    "content": token,
                                    "metadata": {"model": model}
                                }))
                                
                                if data.get("done", False):
                                    break
                        except json.JSONDecodeError:
                            continue
            
            logger.info(f"Ollama {model} 스트리밍 완료: {len(full_response)} 문자")
            return full_response
            
        except Exception as e:
            logger.error(f"Ollama 스트리밍 중 오류: {str(e)}")
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": f"스트리밍 오류: {str(e)}"
            }))
            return ""  # 오류 시 빈 문자열 반환
    
    async def map_placeholders(self, websocket: WebSocket, streamed_response: str, chroma_results: list, model: str = "llama3.2:3b"):
        """
        같은 모델을 재사용하여 플레이스홀더 매핑 (3B 모델 재사용)
        """
        try:
            # 프롬프트 템플릿에서 매핑 프롬프트 생성
            mapping_prompt = ChatPrompts.build_mapping_prompt(streamed_response, chroma_results)
            
            url = f"{self.base_url}/api/generate"  # 같은 3B 모델 사용
            
            payload = {
                "model": model,
                "prompt": mapping_prompt,
                "stream": True,
                "options": {
                    "temperature": PromptConfig.MAPPING_TEMPERATURE,
                    "top_p": PromptConfig.MAPPING_TOP_P,
                    "max_tokens": PromptConfig.MAPPING_MAX_TOKENS
                }
            }
            
            logger.info(f"Ollama {model}로 플레이스홀더 매핑 시작 (모델 재사용)")
            
            async with self.client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if data.get("response"):
                                await websocket.send_text(json.dumps({
                                    "type": "mapping",
                                    "content": data["response"],
                                    "metadata": {"model": model}
                                }))
                                
                                if data.get("done", False):
                                    break
                        except json.JSONDecodeError:
                            continue
            
            logger.info(f"Ollama {model} 플레이스홀더 매핑 완료")
            
        except Exception as e:
            logger.error(f"플레이스홀더 매핑 중 오류: {str(e)}")
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": f"매핑 오류: {str(e)}"
            }))
    
    async def check_model_status(self, model: str) -> Dict[str, Any]:
        """
        Ollama 모델 상태 확인
        """
        try:
            url = f"{self.base_url}/api/tags"
            response = await self.client.get(url)
            response.raise_for_status()
            
            models = response.json().get("models", [])
            model_exists = any(m.get("name") == model for m in models)
            
            return {
                "available": model_exists,
                "model": model,
                "status": "ready" if model_exists else "not_found"
            }
            
        except Exception as e:
            logger.error(f"모델 상태 확인 중 오류: {str(e)}")
            return {
                "available": False,
                "model": model,
                "status": "error",
                "error": str(e)
            }
    
    
    async def close(self):
        """클라이언트 연결 종료"""
        await self.client.aclose()
