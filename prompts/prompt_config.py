"""
프롬프트 설정 관리
"""

class PromptConfig:
    """프롬프트 설정 클래스"""
    
    # 플레이스홀더 패턴
    PLACEHOLDER_PATTERN = "[E{}]"  # [E1], [E2], [E3] ...
    
    # 프롬프트 설정
    INITIAL_TEMPERATURE = 0.7
    INITIAL_TOP_P = 0.9
    INITIAL_MAX_TOKENS = 1000
    
    MAPPING_TEMPERATURE = 0.5
    MAPPING_TOP_P = 0.8
    MAPPING_MAX_TOKENS = 500
    
    # Chroma 결과 개수 제한
    MAX_CHROMA_RESULTS = 5
    
    # 기본 응답 메시지
    NO_INFO_MESSAGE = "관련 정보를 찾을 수 없습니다."
    NO_DOCUMENTS_MESSAGE = "관련 문서를 찾을 수 없습니다."
