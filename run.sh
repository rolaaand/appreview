#!/bin/bash
set -e

cd "$(dirname "$0")"

# 가상환경 생성 (없으면)
if [ ! -d ".venv" ]; then
    echo "가상환경 생성 중..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# 패키지 설치
echo "패키지 설치 중..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "✅ 설치 완료. 대시보드를 시작합니다..."
echo "   브라우저에서 http://localhost:8501 로 접속하세요"
echo ""

streamlit run app.py
