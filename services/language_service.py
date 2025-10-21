"""
언어 감지 및 번역 서비스 모듈
다국어 지원을 위한 언어 감지 및 번역 기능
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
import openai

logger = logging.getLogger(__name__)

# TODO 한국어임을 감지하는 방법을 개선해야 함
class LanguageService:
    """
    [ 언어 감지 및 번역 서비스 클래스 ]
    is_korean: 한국어임을 감지하는 함수
    translate_to_korean: 텍스트 => LLM  => 한국어 번역
    """
    
    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        self.model = model
        self.client = openai.AsyncOpenAI(api_key=api_key)
    
    def is_korean(self, text: str, threshold: float = 0.5) -> bool:
        korean_chars = sum('가' <= ch <= '힣' or 'ㄱ' <= ch <= 'ㅎ' or 'ㅏ' <= ch <= 'ㅣ' for ch in text)
        ratio = korean_chars / max(len(text), 1)
        return ratio >= threshold

    # TODO => LLM 연결 필요
    async def translate_to_korean(self, text: str) -> str:
        try:
            logger.info("번역중")
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": """다음 텍스트를 한국어로 번역해주세요. 
                        대학교 학사 관련 용어는 정확하게 번역하고, 
                        사용자가 대학교에 대해 묻는 질문임을 고려하여, 주제에 맞게 자연스러운 한국어로 번역해주세요."""
                    },
                    {"role": "user", "content": text}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            translated_text = response.choices[0].message.content.strip()
            logger.info("번역 완료")
            
            return translated_text
            
        except Exception as e:
            logger.error(f"번역 중 오류: {str(e)}")
            return text
    
    async def close(self):
        """클라이언트 연결 종료"""
        pass
