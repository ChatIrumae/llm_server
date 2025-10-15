#!/usr/bin/env python3
"""
LangChain Chroma 통합 서비스 테스트 스크립트
"""

import asyncio
import logging
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.langchain_chroma_service import LangChainChromaService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """LangChain Chroma 서비스 테스트"""
    try:
        logger.info("LangChain Chroma 서비스 테스트 시작")
        
        # 서비스 초기화
        chroma_service = LangChainChromaService()
        
        # 컬렉션 정보 출력
        collection_info = await chroma_service.get_collection_info()
        logger.info(f"컬렉션 정보: {collection_info}")
        
        # 검색 테스트
        test_queries = [
            "학사경고 기준이 뭐야?",
            "졸업 요구사항은?",
            "휴학 신청 방법은?",
            "장학금 신청 조건은?",
            "복수전공 신청은?"
        ]
        
        logger.info("=== 검색 테스트 시작 ===")
        for query in test_queries:
            logger.info(f"\n질문: {query}")
            results = await chroma_service.search_documents(query, n_results=3)
            
            for i, result in enumerate(results, 1):
                logger.info(f"  {i}. 점수: {result['score']:.3f}")
                logger.info(f"     카테고리: {result['metadata'].get('category', 'N/A')}")
                logger.info(f"     내용: {result['content'][:100]}...")
        
        # 필터링 검색 테스트
        logger.info("\n=== 필터링 검색 테스트 ===")
        filter_results = await chroma_service.search_with_filter(
            "학점 관련", 
            filter_dict={"category": "graduation"},
            n_results=2
        )
        
        for i, result in enumerate(filter_results, 1):
            logger.info(f"  {i}. 필터링 결과: {result['content'][:100]}...")
        
        # 새 문서 추가 테스트
        logger.info("\n=== 새 문서 추가 테스트 ===")
        new_doc_id = await chroma_service.add_document(
            "시험 성적 확인: 중간고사와 기말고사 성적은 각각 40%와 60%의 비율로 반영되며, 출석은 별도로 평가됩니다.",
            {"category": "exam_grading", "source": "학사처", "type": "academic_policy"}
        )
        logger.info(f"새 문서 추가됨: {new_doc_id}")
        
        # 추가된 문서로 검색 테스트
        new_search_results = await chroma_service.search_documents("시험 성적", n_results=2)
        logger.info(f"새 문서 검색 결과: {len(new_search_results)}개")
        
        # 서비스 상태 확인
        health_status = await chroma_service.health_check()
        logger.info(f"\n=== 서비스 상태 ===")
        logger.info(f"상태: {health_status['status']}")
        logger.info(f"임베딩 모델: {health_status.get('embedding_model', 'N/A')}")
        logger.info(f"저장 디렉토리: {health_status.get('persist_directory', 'N/A')}")
        
        logger.info("LangChain Chroma 서비스 테스트 완료")
        
    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
