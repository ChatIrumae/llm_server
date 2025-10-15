"""
LangChain Chroma 통합 서비스 모듈
LangChain의 내장 Chroma 벡터 스토어를 사용한 문서 검색 기능
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import uuid

logger = logging.getLogger(__name__)

class LangChainChromaService:
    """LangChain Chroma 통합 서비스 클래스"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.collection_name = "university_documents"
        
        # 디렉토리 생성
        os.makedirs(persist_directory, exist_ok=True)
        
        # Ollama 임베딩 모델 초기화
        self.embeddings = OllamaEmbeddings(
            base_url="http://ollama-3b:11434",
            model="llama3.2:3b"
        )
        
        # 텍스트 분할기 초기화
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Chroma 벡터 스토어 초기화
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        
        # 초기 문서가 없으면 샘플 문서 추가
        if self.vectorstore._collection.count() == 0:
            self._populate_sample_documents()
    
    def _populate_sample_documents(self):
        """샘플 문서 데이터 추가"""
        sample_documents = [
            {
                "content": "학사 경고 기준: 당해 학기 평점 평균이 1.5 미만인 경우 학사 경고를 받습니다. 연속 2학기 학사 경고 시 제적처리됩니다.",
                "metadata": {"category": "academic_warning", "source": "학사규정", "type": "academic_policy"}
            },
            {
                "content": "졸업 요구사항: 총 130학점 이상 취득하고, 전공 필수 과목을 모두 이수해야 합니다. 평점 평균 2.0 이상이어야 졸업이 가능합니다.",
                "metadata": {"category": "graduation", "source": "학사규정", "type": "graduation_requirement"}
            },
            {
                "content": "휴학 신청: 학기 시작 2주 전까지 신청 가능하며, 최대 4학기까지 연속 휴학이 가능합니다. 군입대 휴학은 별도 규정이 적용됩니다.",
                "metadata": {"category": "leave_of_absence", "source": "학사규정", "type": "academic_policy"}
            },
            {
                "content": "성적 정정: 성적 공고일로부터 1주일 이내에 신청 가능하며, 교수와 상담 후 정정 신청서를 제출해야 합니다.",
                "metadata": {"category": "grade_correction", "source": "학사규정", "type": "academic_policy"}
            },
            {
                "content": "복수전공 신청: 2학년 2학기부터 신청 가능하며, 주전공 평점 평균 3.0 이상이어야 합니다. 복수전공 이수학점은 36학점 이상입니다.",
                "metadata": {"category": "double_major", "source": "학사규정", "type": "academic_policy"}
            },
            {
                "content": "교환학생 프로그램: 평점 평균 3.0 이상, 어학성적 기준을 만족해야 하며, 1년간 해외 파트너 대학에서 수학할 수 있습니다.",
                "metadata": {"category": "exchange_student", "source": "국제교류처", "type": "international_program"}
            },
            {
                "content": "장학금 신청: 성적우수 장학금은 평점 평균 3.5 이상, 소득기준 장학금은 가구소득 기준을 만족해야 합니다. 매학기 신청 가능합니다.",
                "metadata": {"category": "scholarship", "source": "학생지원처", "type": "financial_aid"}
            },
            {
                "content": "강의 수강신청: 수강신청 기간은 매학기 말 1주일간이며, 정원 초과 시 무작위 추첨을 통해 결정됩니다. 폐강 기준은 수강생 10명 미만입니다.",
                "metadata": {"category": "course_registration", "source": "학사처", "type": "academic_policy"}
            },
            {
                "content": "재수강 신청: F학점을 받은 과목은 다음 학기에 재수강 신청이 가능하며, 최대 3회까지 재수강할 수 있습니다. 재수강 시 이전 성적은 무효화됩니다.",
                "metadata": {"category": "retake", "source": "학사규정", "type": "academic_policy"}
            },
            {
                "content": "전과 신청: 2학년 2학기부터 3학년 1학기까지 전과 신청이 가능하며, 평점 평균 3.0 이상이어야 합니다. 전과 시 이수학점의 50% 이상을 새 전공으로 인정받을 수 있습니다.",
                "metadata": {"category": "major_change", "source": "학사규정", "type": "academic_policy"}
            }
        ]
        
        try:
            # 문서를 LangChain Document 객체로 변환
            documents = []
            for doc in sample_documents:
                documents.append(Document(
                    page_content=doc["content"],
                    metadata=doc["metadata"]
                ))
            
            # 벡터 스토어에 추가
            self.vectorstore.add_documents(documents)
            
            # 변경사항 저장
            self.vectorstore.persist()
            
            logger.info(f"{len(sample_documents)}개의 샘플 문서가 추가됨")
            
        except Exception as e:
            logger.error(f"샘플 문서 추가 중 오류: {str(e)}")
    
    async def search_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        문서 검색 수행 (유사도 기반)
        """
        try:
            logger.info(f"LangChain Chroma 검색 시작: {query[:50]}...")
            
            # 유사도 검색 실행
            docs = self.vectorstore.similarity_search(
                query, 
                k=n_results
            )
            
            # 유사도 점수 계산을 위한 추가 검색
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                query,
                k=n_results
            )
            
            # 결과 포맷팅
            formatted_results = []
            for i, ((doc, score)) in enumerate(zip(docs, docs_with_scores)):
                # 거리를 유사도 점수로 변환 (Chroma는 거리를 반환하므로)
                similarity_score = 1 / (1 + score)
                
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": similarity_score,
                    "rank": i + 1
                })
            
            logger.info(f"LangChain Chroma 검색 완료: {len(formatted_results)}개 결과")
            return formatted_results
            
        except Exception as e:
            logger.error(f"LangChain Chroma 검색 중 오류: {str(e)}")
            return []
    
    async def search_with_filter(
        self, 
        query: str, 
        filter_dict: Optional[Dict[str, Any]] = None,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        필터를 적용한 문서 검색
        """
        try:
            logger.info(f"필터링 검색 시작: {query[:50]}... (필터: {filter_dict})")
            
            # 필터가 있는 경우
            if filter_dict:
                docs_with_scores = self.vectorstore.similarity_search_with_score(
                    query,
                    k=n_results,
                    filter=filter_dict
                )
            else:
                docs_with_scores = self.vectorstore.similarity_search_with_score(
                    query,
                    k=n_results
                )
            
            # 결과 포맷팅
            formatted_results = []
            for i, (doc, score) in enumerate(docs_with_scores):
                similarity_score = 1 / (1 + score)
                
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": similarity_score,
                    "rank": i + 1
                })
            
            logger.info(f"필터링 검색 완료: {len(formatted_results)}개 결과")
            return formatted_results
            
        except Exception as e:
            logger.error(f"필터링 검색 중 오류: {str(e)}")
            return []
    
    async def add_document(self, content: str, metadata: Dict[str, Any]) -> str:
        """
        새 문서 추가
        """
        try:
            # 텍스트를 청크로 분할
            chunks = self.text_splitter.split_text(content)
            
            # Document 객체 생성
            documents = []
            for chunk in chunks:
                doc_metadata = metadata.copy()
                doc_metadata["chunk_id"] = str(uuid.uuid4())
                documents.append(Document(
                    page_content=chunk,
                    metadata=doc_metadata
                ))
            
            # 벡터 스토어에 추가
            self.vectorstore.add_documents(documents)
            
            # 변경사항 저장
            self.vectorstore.persist()
            
            logger.info(f"새 문서 추가됨: {len(chunks)}개 청크")
            return f"added_{len(chunks)}_chunks"
            
        except Exception as e:
            logger.error(f"문서 추가 중 오류: {str(e)}")
            raise
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """
        컬렉션 정보 조회
        """
        try:
            count = self.vectorstore._collection.count()
            return {
                "name": self.collection_name,
                "document_count": count,
                "status": "active",
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"컬렉션 정보 조회 중 오류: {str(e)}")
            return {
                "name": self.collection_name,
                "document_count": 0,
                "status": "error",
                "error": str(e)
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        서비스 상태 확인
        """
        try:
            # 컬렉션 존재 여부 확인
            collection_info = await self.get_collection_info()
            
            # 간단한 검색 테스트
            test_results = await self.search_documents("테스트", n_results=1)
            
            return {
                "status": "healthy",
                "collection": collection_info,
                "search_test": "passed" if test_results else "failed",
                "embedding_model": "llama3.2:3b",
                "persist_directory": self.persist_directory
            }
            
        except Exception as e:
            logger.error(f"서비스 상태 확인 중 오류: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def close(self):
        """서비스 종료"""
        try:
            # 벡터 스토어 변경사항 저장
            self.vectorstore.persist()
            logger.info("LangChain Chroma 서비스 종료됨")
        except Exception as e:
            logger.error(f"서비스 종료 중 오류: {str(e)}")
