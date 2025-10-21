"""
언어 감지 및 번역 서비스 모듈
다국어 지원을 위한 언어 감지 및 번역 기능
"""
import httpx
import os
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
    is_korean: 한국어임을 감지하는 함수 (한글 문자 비율 기반)
    detect_language: 텍스트 언어 감지 (한국어/기타)
    translate_to_korean: DeepL API를 사용한 한국어 번역
    """
    
    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        self.model = model
        self.client = openai.AsyncOpenAI(api_key=api_key)
    
    def is_korean(self, text: str, threshold: float = 0.5) -> bool:
        korean_chars = sum('가' <= ch <= '힣' or 'ㄱ' <= ch <= 'ㅎ' or 'ㅏ' <= ch <= 'ㅣ' for ch in text)
        ratio = korean_chars / max(len(text), 1)
        return ratio >= threshold
    
    async def detect_language(self, text: str) -> str:
        """
        텍스트의 언어를 감지합니다.
        한국어가 포함되어 있으면 'ko', 그 외는 'other' 반환
        """
        try:
            # is_korean 함수를 사용하여 한국어 감지
            if self.is_korean(text):
                return 'ko'
            else:
                return 'other'
                
        except Exception as e:
            logger.error(f"언어 감지 중 오류: {str(e)}")
            # 오류 시 기본값으로 'other' 반환
            return 'other'

    async def translate_to_korean(self, text: str) -> str:
        try:
            logger.info("DeepL 번역 시작")
            
            # DeepL API 키 (임시로 AAA 설정)
            deepl_api_key = os.getenv("DEEPL_API_KEY", "AAA")
            deepl_url = "https://api-free.deepl.com/v2/translate"
            
            # DeepL API 요청 파라미터
            params = {
                "auth_key": deepl_api_key,
                "text": text,
                "target_lang": "KO",  # 한국어
                "source_lang": "auto"  # 자동 감지
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(deepl_url, data=params)
                
                if response.status_code == 200:
                    result = response.json()
                    translated_text = result["translations"][0]["text"]
                    logger.info("DeepL 번역 완료")
                    return translated_text
                else:
                    logger.error(f"DeepL API 오류: {response.status_code} - {response.text}")
                    # DeepL API 실패 시 원본 텍스트 반환
                    return text
                    
        except Exception as e:
            logger.error(f"DeepL 번역 중 오류: {str(e)}")
            # 번역 실패 시 원본 텍스트 반환
            return text
    
    async def close(self):
        """클라이언트 연결 종료"""
        pass
