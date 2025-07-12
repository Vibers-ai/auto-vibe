#!/usr/bin/env python3
"""
VIBE CLI 모듈 진입점
python -m src.cli 명령어로 실행할 수 있게 해줍니다.
"""

if __name__ == "__main__":
    # cli.py가 상위 디렉토리에 있으므로 import 경로 조정
    import sys
    from pathlib import Path
    
    # src 디렉토리를 Python path에 추가
    src_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(src_dir))
    
    # cli.py의 main 함수 실행
    from src.cli import main
    main()