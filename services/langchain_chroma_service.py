"""
LangChain Chroma 통합 서비스 모듈
LangChain의 내장 Chroma 벡터 스토어를 사용한 문서 검색 기능
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
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
        
        # OpenAI 임베딩 모델 초기화
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
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
            logger.info("LangChain Chroma 검색 시작")
            
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
    
    async def mock_search_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        테스트용 Mock 문서 검색 함수 (2초 지연 후 고정 문서 반환)
        """
        try:
            logger.info("Mock Chroma 검색 시작 (2초 지연)")
            
            # 2초 지연 (실제 검색 시간 시뮬레이션)
            await asyncio.sleep(2)
            
            # 고정된 테스트 문서 반환
            mock_document = """
제6장 시험ㆍ성적평가
제31조(시험의 종류)
시험은 정기시험, 수시시험, 추가시험으로 구분한다.<개정 2003.2.7>
제32조(시험의 실시)
① 교과목 담당교수는 정기시험을 지정된 기간에 실시한다.<개정 2003.2.7>
② 교과목 담당교수는 정기시험 이외에 수시시험을 실시할 수 있다.<신설 2003.2.7>
③ 교과목 담당교수는 질병 또는 삼촌 이내의 친족의 사망, 재난, 기타 부득이한 사유로 정기시험 또는 수시시험에 응시하지 못한 학생에 대하여 추가시험을 실시할 수 있다. <신설 2003.2.7., 2022.5.30.>
④ 추가시험에 응시하고자 하는 학생은 해당시험 전후 1주일 이내에 교과목 담당교수에게 관련 증빙서류를 포함하여 사유서를 제출하여야 한다.<신설 2003.2.7., 2022.5.30.>
제33조(시험시간표)
① 교과목 담당교수는 정기시험의 시간표를 수업시간표에 준하여 편성하고 추가시험의 시간표는 수강에 지장이 없도록 편성하여야 한다.<개정 2003.2.7.>
② 전항의 시험시간표는 시험개시 일주일 전에 공고하여야 한다.
제34조(시험감독)
시험감독은 당해 교과목 담당교수로 한다. 그러나 담당교수가 수행하기 어려울 때에는 다른 교수로 대행하게 할 수 있다.
제35조
삭 제
제36조
삭 제
제37조(부정행위자 처리)
① 시험중 부정행위를 한 자를 발견한 감독자는 그 증거물을 첨부하여 교무처장에게 보고하여야 한다.
② 시험중 부정행위를 한 자에 대한 징계는 학생생활지도규정이 정하는 바에 따른다.<개정 2003.2.7, 2009.12.7>
③ 시험중 부정행위를 한 자의 성적은 F로 처리한다.<신설 2003.2.7>
제38조(성적평가원의 다양화)
교과목의 성적은 담당교수가 학생의 시험, 과제물, 출석 및 학업태도(예습, 복습, 협의, 토의 등) 등에 의하여 종합적으로 평가하여야 한다. 그러나 실험ㆍ실습, 실기 및 이에 준하는 특수과목의 성적은 별도의 방법으로 평가할 수 있다.<개정 2003.2.7>
제39조(성적평가방법)
① 교과목의 성적평가는 교수목표의 성취수준에 준거를 두어야 하며 다음과 같은 성적분포비율에 의한 상대평가방법에 의함을 원칙으로 한다. 다만, 전공교과목의 경우 B+이상이 60%(실험ㆍ실습, 실기, 공학교육인증 설계비중이 1학점 이상인 전공교과목의 경우 75%)를 초과하지 않아야 하고, 교양교과목의 경우 A⁰ 이상은 35%(실험ㆍ실습, 실기의 경우는 45%), B+이상은 60%(실험ㆍ실습, 실기의 경우는 75%)를 초과하지 않아야 한다. <개정 2011.12.9., 2014.1.24., 2022.12.13., 2024.01.15. 2024.8.5.>
- 전공 및 교양 교과목 상대평가 분포비율 -
A⁺ 및 A⁰ 등급 30%±5
B⁺ 및 B⁰ 등급 40%±5
C⁺ 및 C⁰ 등급 20%±5
D⁺, D⁰ 및 F등급 10%±5
다만, 다음 각 호의 어느 하나에 해당되는 경우에는 절대평가를 적용할 수 있다.<단서신설 2003.2.7., 개정 2024.01.15.>
1. 수강인원 15명 이하인 교과목 <개정 2014.5.29., 2024.01.15.>
2. 교직과목
3. 총장이 인정하는 과목
4. 신입학, 편입학, 학점교류 등을 통해 본교에 재학중인 외국인 학생<2003.7. 15>
5. 외국어로 강의하는 교과목(다만, 외국어 교과목은 제외) <신설 2014.5.29., 2022.12.13.>
6. R.O.T.C교과목(군사학)<신설 2019.9.6>
7. 성적평가가 이수(S),미이수(U)인 과목<신설 2019.9.6.>
8. 제4호에 해당하는 외국인 학생이 수강인원의 20%이상인 전공 교과목 <신설 2024.8.5.>
② <삭제 2003.2.7>
③ 총장은 매학기초 성적평가의 기본방침을 정하여 시행하게 할 수 있다.
④ 총장은 제1항에도 불구하고 천재지변 또는 그 밖에 교육과정의 운영상 부득이한 사유 발생시 성적평가방법 및 분포비율을 따로 정할 수 있다<개정 2020.9.9.>
제40조(유고결석, 공결)
① 학생이 다음 각 호의 어느 하나에 해당하여 수업에 참석하지 못하는 경우에는 결석으로 처리하지 않을 수 있으며, 학생은 정해진 절차 및 기준에 따라 승인을 받아야한다. 이에 관한 세부사항은 지침으로 따로 정하되, 제3호의 경우 결석으로 처리하지 않는 것으로 한다.〈개정 2003.2.7., 2016. 11.4., 2019.9.6., 2022.4.18. 2024.8.5., 2025.2.3.>
1. 친족사망 : 3일(외·조부모, 자녀), 5일(부모, 배우자, 형제·자매)
2. 법정 전염병 확진 또는 의심으로 격리 조치가 필요한 경우, 응급실 진료, 입원, 기타 질병으로 인하여 출석이 어려운 경우 : 수업시수의 1/3 이내
3. 「예비군법」 및 「병역법」에 따른 병역판정검사, 훈련 등에 참가하는 경우 : 실제 소요기간
4. 교육실습(교직)으로 출석이 어려운 경우 : 실제 소요기간
5. 학과 수업과 연계된 현장 학습(답사, 학회 등)에 참가하는 경우 : 실제 소요기간
6. 총장이 허가한 학교행사 및 이에 준하는 경우, 단과대학 공식행사 : 실제 소요기간
7. 생리공결에 의한 사유 : 학기 중 최대 4일
8. 총학생회 주최 정기학생총회(연2회): 실제 소요기간
9. 본인 및 배우자 출산(유산·사산 포함) : 본인 20일, 배우자 10일
10. 졸업예정자(해당 학기 종료시 졸업이 가능한 자)가 학기중에 취업이 확정된 경우 : 취업기간
11. 학생활동부서 임원의 국제회합 또는 이에 준하는 경우 : 실제 소요기간
12. 정부·지방자치단체의 요청에 의한 특별회담 : 실제 소요기간
13. 기타 사유로 부득이하게 결석하는 경우 담당교수의 재량에 의하여 출석으로 대체인정 할 수 있으나, 해당 증빙서류 제출 및 출석에 상응하는 활동 등을 수행하여야 한다. : 실제 소요기간
② <신설 2003.2.7. 삭제 2025.2.3.>
③ 출석 인정은 사전에 신청하여 승인을 받는 것을 원칙으로 하며, 부득이한 사유로 신청하지 못한 경우 해당 사유발생일로부터 7일 이내에 신청하여야 한다. 다만, 불가피한 경우를 제외하고 신청 가능 시기는 종강일 이전으로 한다. <신설 2025.2.3.>
④ 교과목 담당 교원은 유고결석 및 공결을 승인받은 학생에게 대체과제를 부여 할 수 있다. <신설 2025.2025.2.3.>
제41조(성적평가 입력 및 제출)
① 교과목 담당교수는 기말시험 기간 종료 후 10일 이내에 대학행정정보시스템에 성적(세부항목 포함)을 입력ㆍ제출하고, 전자출결시스템에 출결 및 강의마감 처리를 완료하여야 한다. 다만, 전자출결시스템을 사용하지 않은 교과목은 출석 관련 증빙자료를 교과목이 개설된 학부ㆍ과장에게 제출하여야 한다.<개정 2003.2.7., 2022.4.18.>
② 기말시험 기간 전에 휴학한 자의 성적은 평가하지 아니한다. 다만, 수업일수의 4분의 3 이상을 출석하고 부득이한 사유로 휴학한 학생이 성적 취득원을 제출할 경우에 담당교수는 성적을 평가하여야 한다.<개정 2003.2.7>
③ 삭 제
④ 학부ㆍ과장은 제1항의 성적 및 출결 관련 증빙자료에 미비사항이 있을 경우 보완한 후 즉시 대학장에게 제출하여야 한다.<개정 2022.4.18.>
⑤ 전산 입력된 성적표의 성적란에 성적이 없는 것은 F로 처리한다.<개정 2003.2.7>
제42조(성적의 열람)
① 소속 대학장은 학생에게 매학기 성적표를 교부한다.<개정 2003.2.7>
② 매학기 성적표는 학생이 직접 인터넷으로 성적을 열람할 수 있고 소속 대학 행정실에 문의할 수 있다.<신설 2003.2.7., 2014.4.14>
제43조
삭 제
제44조(성적의 정정)
① 성적에 착오 또는 누락이 있는 경우에는 성적을 열람할 수 있는 날로부터 10일 이내에 담당교수의 사유서 및 이를 증빙할 수 있는 구체적인 자료를 첨부하여 당해 교과목이 개설된 학부ㆍ과장이 대학장에게 성적정정을 신청할 수 있다. 다만, 제39조를 준수하여야 한다.<단서신설 2003.2.7>
② 대학장은 전항의 신청에 대하여 정당한 사유가 있다고 인정되는 경우에는 이를 허가할 수 있다.
③ 성적 정정을 허가하는 경우 해당 대학장은 교무처장에게 성적정정 처리를 의뢰하고 처리결과를 신청 학부ㆍ과장과 해당 학생의 소속 학부ㆍ과장에게 통보하여야 한다.<개정 2003.2.7>
제45조(성적의 취소)
다음 각 호의 1에 해당하는 자의 성적은 이를 취소한다.
1. 시험 중 부정행위를 하여 무효처분을 받은 해당과목의 성적
2. 미등록 응시자의 성적
3. 수강신청하지 아니한 자의 성적
4. 재수강 과목을 재수강 신청하지 아니하고 응시한 자의 성적
5. 수강신청한 과목이 아닌 과목의 성적
6. 착오로 잘못 부여된 자의 성적
7. 기타 부정한 방법으로 얻은 과목의 성적
제46조(평점의 계산)
① 학업성적의 평점평균은 교과목의 학점수와 평점을 곱한 평점 합계를 평점을 부여하는 신청학점의 합계로 나누어 산출하되, 소수점 이하 셋째자리에서 반올림한다. <개정 2016.12. 8>
② 성적 평점 평균이 같은 경우에는 다음에 의하여 성적순위를 정한다.
1. 평점합계
2. 학점합계
3. 성적취득 과목수
제47조(성적확정 정리)
① 대학장은 성적정정기간이 경과한 후 즉시 교무처장에게 성적확정 통보를 하여야 한다.<개정 2003.2.7>
② 교무처장은 재수강 등 무효 처리된 성적이 포함된 최종 누적성적표를 학적부에 첨부하여 관리한다.<개정 2003.2.7>
③ 총장은 제41조제1항의 성적표를 영구 보존ㆍ관리하여야 하며, 전자출석부는 당해 학생의 졸업시까지 보존ㆍ관리하여야 한다. 다만, 전자출결시스템을 사용하지 않은 교과목의 증빙자료는 대학장이 당해 학생의 졸업 시까지 보존ㆍ관리하여야 한다.<신설 2003.2.7., 2022.4.18.>
"""
            
            # Mock 결과 포맷팅
            formatted_results = [{
                "content": mock_document,
                "metadata": {
                    "category": "exam_grade_evaluation",
                    "source": "학사규정",
                    "type": "academic_policy",
                    "chapter": "제6장",
                    "title": "시험ㆍ성적평가"
                },
                "score": 0.95,  # 높은 유사도 점수
                "rank": 1
            }]
            
            logger.info(f"Mock Chroma 검색 완료: {len(formatted_results)}개 결과 반환")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Mock Chroma 검색 중 오류: {str(e)}")
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
            logger.info(f"필터링 검색 시작 (필터: {filter_dict})")
            
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
