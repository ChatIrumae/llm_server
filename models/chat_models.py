"""
채팅 관련 데이터 모델 정의
"""

# WebSocket 기반 채팅 시스템에서는 별도의 모델이 필요하지 않음
# JSON 메시지로 직접 통신

# 필요시 추가할 수 있는 모델들:
# - WebSocketMessage: {"type": "token|mapping|complete|error", "content": str, "metadata": dict}
# - HealthCheckResponse: {"status": str, "services": dict, "timestamp": str}
