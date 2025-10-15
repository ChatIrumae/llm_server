"""
채팅 관련 프롬프트 템플릿 관리
"""

from typing import List, Dict, Any
from .prompt_config import PromptConfig

class ChatPrompts:
    """채팅 프롬프트 템플릿 클래스"""
    
    @staticmethod
    def build_initial_prompt(message: str, user_info: Dict[str, Any] = None) -> str:
        """
        초기 답변 생성을 위한 프롬프트
        """
        user_context = ""
        if user_info:
            user_context = f"""
사용자 정보:
- 학과: {user_info.get('major', '정보 없음')}
- 학년: {user_info.get('grade', '정보 없음')}
- 학생 유형: {user_info.get('student_type', '정보 없음')}
- 기타: {user_info.get('additional_info', '정보 없음')}

"""
        
        return f"""
당신은 대학교 학사 관련 질의를 처리하는 AI 어시스턴트입니다.
사용자의 질문에 대해 구조화된 답변을 제공하되, 구체적인 수치나 조건은 [E1], [E2] 등의 플레이스홀더로 표시하세요.

{user_context}질문: {message}

답변 형식:
- 명확하고 구조화된 답변 제공
- 구체적인 수치나 조건은 [E1], [E2] 등의 플레이스홀더 사용
- 대학교 학사 규정에 맞는 정확한 정보 제공
- 사용자 정보를 고려한 맞춤형 답변 제공
"""
    
    @staticmethod
    def build_mapping_prompt(streamed_response: str, chroma_results: List[Dict[str, Any]]) -> str:
        """
        플레이스홀더 매핑을 위한 프롬프트
        """
        # Chroma DB 결과를 텍스트로 변환
        context_text = ChatPrompts._format_chroma_results(chroma_results)
        
        no_info_msg = PromptConfig.NO_INFO_MESSAGE
        
        return f"""
다음은 당신이 이전에 생성한 답변입니다:
{streamed_response}

다음은 대학교 관련 문서에서 검색된 구체적인 정보입니다:
{context_text}

위 검색된 정보를 바탕으로 이전 답변의 플레이스홀더 [E1], [E2] 등을 실제 값으로 대체하여 완전한 답변을 생성해주세요.
검색된 정보가 없다면 "{no_info_msg}"라고 표시하세요.
"""
    
    @staticmethod
    def build_integrated_prompt(message: str, chroma_results: List[Dict[str, Any]]) -> str:
        """
        통합 답변 생성을 위한 프롬프트 (Chroma 결과 포함)
        """
        # Chroma DB 결과를 텍스트로 변환
        context_text = ChatPrompts._format_chroma_results(chroma_results)
        
        no_info_msg = PromptConfig.NO_INFO_MESSAGE
        
        return f"""
당신은 대학교 학사 관련 질의를 처리하는 AI 어시스턴트입니다.

사용자 질문: {message}

관련 문서 정보:
{context_text}

위 관련 문서 정보를 바탕으로 사용자 질문에 대한 정확하고 구체적인 답변을 제공해주세요.
답변 시 다음 사항을 준수해주세요:
- 관련 문서의 구체적인 수치와 조건을 정확히 포함
- 명확하고 이해하기 쉬운 구조로 답변 제공
- 관련 문서에 정보가 없다면 "{no_info_msg}"라고 표시
- 대학교 학사 규정에 맞는 정확한 정보 제공
"""
    
    @staticmethod
    def _format_chroma_results(chroma_results: List[Dict[str, Any]]) -> str:
        """
        Chroma DB 검색 결과를 텍스트 형식으로 변환
        """
        if not chroma_results:
            return PromptConfig.NO_DOCUMENTS_MESSAGE
        
        formatted_results = []
        max_results = PromptConfig.MAX_CHROMA_RESULTS
        for i, result in enumerate(chroma_results[:max_results], 1):
            formatted_results.append(f"{i}. {result.get('content', '')}")
        
        return "\n".join(formatted_results)
