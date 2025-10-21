"""
위치 관련 서비스 모듈
위치 정보 감지 및 지도 액션 기능
"""

import asyncio
import json
import logging
import re
from typing import Dict, Any, Optional, List
import openai

logger = logging.getLogger(__name__)

# detect_locaiton_mention 수정
class LocationService:
    """위치 관련 서비스 클래스"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        self.model = model
        self.client = openai.AsyncOpenAI(api_key=api_key)
        
        # 위치 관련 키워드
        self.location_keywords = [
            "위치", "장소", "어디", "길", "가는", "오는", "찾는", "찾아", "방문",
            "건물", "강의실", "교실", "도서관", "기숙사", "학생회관", "체육관",
            "식당", "카페", "편의점", "은행", "우체국", "병원", "약국",
            "교수실", "사무실", "행정실", "학과사무실", "학생지원센터"
        ]
        
        # 대학교 내 주요 건물/시설 (환경변수에서 좌표 가져오기)
        import os
        default_lat = float(os.getenv("UNIVERSITY_LAT", "37.5665"))
        default_lng = float(os.getenv("UNIVERSITY_LNG", "126.9780"))
        
        self.university_buildings = {
            "도서관": {"name": "중앙도서관", "address": "대학교 중앙도서관", "coordinates": {"lat": default_lat, "lng": default_lng}},
            "기숙사": {"name": "학생기숙사", "address": "대학교 학생기숙사", "coordinates": {"lat": default_lat + 0.001, "lng": default_lng + 0.001}},
            "학생회관": {"name": "학생회관", "address": "대학교 학생회관", "coordinates": {"lat": default_lat - 0.001, "lng": default_lng - 0.001}},
            "체육관": {"name": "체육관", "address": "대학교 체육관", "coordinates": {"lat": default_lat - 0.002, "lng": default_lng - 0.002}},
            "식당": {"name": "학생식당", "address": "대학교 학생식당", "coordinates": {"lat": default_lat + 0.002, "lng": default_lng + 0.002}},
            "행정실": {"name": "본관 행정실", "address": "대학교 본관", "coordinates": {"lat": default_lat - 0.0005, "lng": default_lng + 0.0005}},
            "강의실": {"name": "강의동", "address": "대학교 강의동", "coordinates": {"lat": default_lat - 0.0015, "lng": default_lng - 0.0005}}
        }
    
    async def detect_location_mention(self, text: str) -> bool:
        """
        텍스트에 위치 관련 내용이 포함되어 있는지 감지합니다.
        """
        try:
            # 키워드 기반 감지
            text_lower = text.lower()
            for keyword in self.location_keywords:
                if keyword in text_lower:
                    return True
            
            # OpenAI를 통한 위치 관련성 감지
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "다음 텍스트가 위치, 장소, 길찾기, 건물 찾기 등과 관련이 있는지 판단해주세요. 관련이 있으면 'YES', 없으면 'NO'로만 답변해주세요."
                    },
                    {"role": "user", "content": text}
                ],
                temperature=0.1,
                max_tokens=5
            )
            
            result = response.choices[0].message.content.strip().upper()
            return result == "YES"
            
        except Exception as e:
            logger.error(f"위치 관련성 감지 중 오류: {str(e)}")
            return False
    
    async def extract_location_info(self, text: str) -> Optional[Dict[str, Any]]:
        """
        텍스트에서 위치 정보를 추출합니다.
        """
        try:
            # 건물명 매칭
            for building_key, building_info in self.university_buildings.items():
                if building_key in text:
                    return {
                        "building": building_key,
                        "name": building_info["name"],
                        "address": building_info["address"],
                        "coordinates": building_info["coordinates"],
                        "type": "university_building"
                    }
            
            # OpenAI를 통한 위치 정보 추출
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": """다음 텍스트에서 찾고자 하는 장소나 건물명을 추출해주세요. 
                        대학교 관련 건물이나 시설명을 찾아주세요. 
                        JSON 형태로 {"building": "건물명", "description": "설명"}으로 답변해주세요."""
                    },
                    {"role": "user", "content": text}
                ],
                temperature=0.3,
                max_tokens=100
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # JSON 파싱 시도
            try:
                result = json.loads(result_text)
                if result.get("building"):
                    # 추출된 건물명이 대학교 건물 목록에 있는지 확인
                    building_name = result["building"]
                    for building_key, building_info in self.university_buildings.items():
                        if building_key in building_name or building_name in building_key:
                            return {
                                "building": building_key,
                                "name": building_info["name"],
                                "address": building_info["address"],
                                "coordinates": building_info["coordinates"],
                                "type": "university_building",
                                "description": result.get("description", "")
                            }
                    
                    # 일반적인 위치 정보로 반환
                    return {
                        "building": building_name,
                        "name": building_name,
                        "address": f"대학교 {building_name}",
                        "coordinates": {"lat": self.university_buildings["도서관"]["coordinates"]["lat"], "lng": self.university_buildings["도서관"]["coordinates"]["lng"]},  # 기본 좌표
                        "type": "general_location",
                        "description": result.get("description", "")
                    }
            except json.JSONDecodeError:
                pass
            
            return None
            
        except Exception as e:
            logger.error(f"위치 정보 추출 중 오류: {str(e)}")
            return None
    
    async def generate_map_action(self, location_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        위치 정보를 바탕으로 지도 액션을 생성합니다.
        """
        try:
            map_action = {
                "type": "map_action",
                "action": "show_location",
                "location": {
                    "name": location_info["name"],
                    "address": location_info["address"],
                    "coordinates": location_info["coordinates"]
                },
                "message": f"{location_info['name']} 위치를 지도에서 확인할 수 있습니다.",
                "map_url": f"https://www.google.com/maps?q={location_info['coordinates']['lat']},{location_info['coordinates']['lng']}"
            }
            
            return map_action
            
        except Exception as e:
            logger.error(f"지도 액션 생성 중 오류: {str(e)}")
            return None
    
    async def process_location_query(self, query: str, response_text: str) -> Optional[Dict[str, Any]]:
        """
        위치 관련 질의를 처리하고 지도 액션을 반환합니다.
        """
        try:
            # 1. 위치 관련성 감지
            is_location_related = await self.detect_location_mention(query) or await self.detect_location_mention(response_text)
            
            if not is_location_related:
                return None
            
            # 2. 위치 정보 추출
            location_info = await self.extract_location_info(query) or await self.extract_location_info(response_text)
            
            if not location_info:
                return None
            
            # 3. 지도 액션 생성
            map_action = await self.generate_map_action(location_info)
            
            return map_action
            
        except Exception as e:
            logger.error(f"위치 질의 처리 중 오류: {str(e)}")
            return None
    
    async def close(self):
        """클라이언트 연결 종료"""
        pass
