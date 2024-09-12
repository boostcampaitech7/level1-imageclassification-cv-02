import os
import sys

# 프로젝트 루트 디렉토리 경로
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# src 디렉토리를 sys.path에 추가
sys.path.insert(0, project_root)
