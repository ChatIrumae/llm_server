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
    """언어 감지 및 번역 서비스 클래스"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        self.model = model
        self.client = openai.AsyncOpenAI(api_key=api_key)
        
        # 한국어 키워드 (대학교 관련 용어)
        self.korean_keywords = [
            "학사", "졸업", "수강", "성적", "휴학", "복학", "전과", "복수전공", 
            "장학금", "등록금", "시험", "과제", "출석", "학점", "전공", "교양",
            "교수", "강의", "강의실", "도서관", "기숙사", "학생회", "동아리"
        ]
    
    async def detect_language(self, text: str) -> str:
        """
        텍스트의 언어를 감지합니다.
        한국어가 포함되어 있으면 'ko', 그 외는 'other' 반환
        """
        try:
            # 간단한 한국어 감지 (키워드 기반)
            text_lower = text.lower()
            for keyword in self.korean_keywords:
                if keyword in text_lower:
                    return 'ko'
            
            # OpenAI를 통한 언어 감지
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "다음 텍스트의 언어를 감지해주세요. 한국어면 'ko', 그 외 언어면 해당 언어 코드를 반환해주세요. (예: en, ja, zh, es 등)"
                    },
                    {"role": "user", "content": text}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            detected_lang = response.choices[0].message.content.strip().lower()
            
            # 한국어가 아니면 'other'로 반환
            if detected_lang == 'ko':
                return 'ko'
            else:
                return 'other'
                
        except Exception as e:
            logger.error(f"언어 감지 중 오류: {str(e)}")
            # 오류 시 기본값으로 'other' 반환
            return 'other'
    
    async def translate_to_korean(self, text: str) -> str:
        """
        텍스트를 한국어로 번역합니다.
        """
        try:
            logger.info("번역 시작")
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": """다음 텍스트를 한국어로 번역해주세요. 
                        대학교 학사 관련 용어는 정확하게 번역하고, 
                        사용자가 대학교에 대해 묻는 질문임을 고려하여 자연스러운 한국어로 번역해주세요."""
                    },
                    {"role": "user", "content": text}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            translated_text = response.choices[0].message.content.strip()
            logger.info("번역 완료")
            
            return translated_text
            
        except Exception as e:
            logger.error(f"번역 중 오류: {str(e)}")
            # 번역 실패 시 원본 텍스트 반환
            return text
    
    async def process_multilingual_query(self, query: str) -> Dict[str, Any]:
        """
        다국어 질의를 처리합니다.
        외국어인 경우 한국어로 번역 후 반환
        """
        try:
            # 1. 언어 감지
            detected_language = await self.detect_language(query)
            
            result = {
                "original_query": query,
                "detected_language": detected_language,
                "processed_query": query,
                "was_translated": False
            }
            
            # 2. 한국어가 아닌 경우 번역
            if detected_language != 'ko':
                translated_query = await self.translate_to_korean(query)
                result["processed_query"] = translated_query
                result["was_translated"] = True
                result["translated_query"] = translated_query
            
            return result
            
        except Exception as e:
            logger.error(f"다국어 질의 처리 중 오류: {str(e)}")
            # 오류 시 원본 쿼리 반환
            return {
                "original_query": query,
                "detected_language": "unknown",
                "processed_query": query,
                "was_translated": False,
                "error": str(e)
            }
    
    async def close(self):
        """클라이언트 연결 종료"""
        pass
