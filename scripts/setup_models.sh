#!/bin/bash

# Ollama 모델 설치 스크립트
# 이 스크립트는 Docker Compose가 시작된 후 실행되어야 합니다.

echo "=== Ollama 모델 설치 시작 ==="

# Ollama 3B 서버에 Llama 3.2 3B 모델 설치
echo "Llama 3.2 3B 모델 설치 중..."
docker exec ollama-3b ollama pull llama3.2:3b

if [ $? -eq 0 ]; then
    echo "✅ Llama 3.2 3B 모델 설치 완료"
else
    echo "❌ Llama 3.2 3B 모델 설치 실패"
    exit 1
fi

# 모델 상태 확인
echo "=== 모델 상태 확인 ==="
echo "3B 모델 목록:"
docker exec ollama-3b ollama list

echo "=== 3B 모델 설치 완료 ==="
