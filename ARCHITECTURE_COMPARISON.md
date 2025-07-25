# VIBE 아키텍처 비교 - 기존 vs 향상된 구조

## 사용자 질문에 대한 답변

### Q: "master가 들고 있는 context를 압축하는 거지? code claude는 code를 짤때마다 매번 새로 프로세스를 여는 것인가?"

**답변: 맞습니다. 하지만 중요한 최적화가 있습니다:**

## 아키텍처 구조

### Master Claude (Gemini) - 지속적 프로세스
```
Master Claude Process (프로젝트 전체 실행)
├─ 작업 1 실행 → 컨텍스트 추가 & 압축
├─ 작업 2 실행 → 컨텍스트 추가 & 압축  
├─ 작업 3 실행 → 컨텍스트 추가 & 압축
├─ ...
└─ 프로젝트 완료까지 계속 메모리 유지
```

**특징:**
- ✅ 전체 프로젝트 메모리 유지
- ✅ 5가지 압축 전략으로 토큰 한계 관리
- ✅ 패턴 학습 및 누적
- ✅ 작업 간 의존성 이해

### Code Claude - 최적화된 Stateless 실행
```
각 작업마다:
작업 1: 새 Claude 프로세스 + 큐레이션된 컨텍스트 → 실행 → 종료
작업 2: 새 Claude 프로세스 + 큐레이션된 컨텍스트 → 실행 → 종료  
작업 3: 새 Claude 프로세스 + 큐레이션된 컨텍스트 → 실행 → 종료
```

**핵심 최적화:**
- 🎯 **큐레이션된 컨텍스트**: Master가 관련있는 정보만 선별해서 제공
- 🎯 **프로젝트 영역별 세션**: 같은 영역(예: auth, api) 작업은 세션 지속
- 🎯 **인사이트 전파**: Code Claude의 학습 내용을 Master에게 전달

## 구체적 작동 방식

### 1. 컨텍스트 큐레이션 예시
```python
# Master Claude가 Code Claude에게 제공하는 컨텍스트
curated_context = {
    "project_summary": "진행률: 3/10 완료, 인증 시스템 구축 중",
    "established_patterns": ["JWT 토큰 사용", "bcrypt 해싱", "REST API 구조"],
    "coding_conventions": {"style": "Python PEP8", "testing": "pytest"},
    "related_tasks": ["auth-001에서 JWT 구현 완료", "auth-002에서 비밀번호 해싱 추가"],
    "current_files": ["src/auth/models.py 존재", "tests/test_auth.py 존재"],
    "architectural_decisions": ["FastAPI 사용", "SQLite 개발 DB"]
}
```

### 2. 프로젝트 영역별 세션 관리
```python
# Code Claude 세션 관리
sessions = {
    "auth": persistent_session_1,    # 인증 관련 작업들이 공유
    "api": persistent_session_2,     # API 관련 작업들이 공유  
    "frontend": persistent_session_3  # 프론트엔드 작업들이 공유
}
```

### 3. 컨텍스트 압축 전략
```python
# Master Claude의 5가지 압축 전략
strategies = {
    "SUMMARIZE": "완료된 작업들을 요약으로 압축",
    "HIERARCHICAL": "중요도에 따라 계층적 관리", 
    "SLIDING_WINDOW": "최근 작업 위주로 유지",
    "SEMANTIC_FILTERING": "현재 작업과 관련성 높은 것만 유지",
    "HYBRID": "위 전략들을 조합해서 최적 압축"
}
```

## 비효율성 해결

### 기존 문제점들
❌ Code Claude가 매번 프로젝트 구조를 다시 파악  
❌ 이전 작업에서 정한 코딩 스타일을 모름  
❌ 같은 설명을 반복해서 제공해야 함  
❌ 작업 간 일관성 부족  

### 향상된 구조의 해결책
✅ **선별적 컨텍스트**: Master가 필요한 정보만 큐레이션해서 제공  
✅ **패턴 학습**: 이전 작업에서 확립된 패턴을 자동 전달  
✅ **세션 지속성**: 같은 프로젝트 영역 내에서는 컨텍스트 유지  
✅ **양방향 학습**: Code Claude의 인사이트를 Master가 흡수  

## 실제 예시

### 작업 실행 시나리오
```
작업: auth-003 (이메일 인증 추가)

1. Master Claude 분석:
   - "auth-001에서 JWT 구현됨"
   - "auth-002에서 bcrypt 해싱 추가됨"  
   - "패턴: FastAPI + SQLAlchemy 사용"
   - "스타일: 타입 힌트 필수, pytest 테스트"

2. 큐레이션된 컨텍스트 생성:
   - 기존 인증 구조 요약 (500토큰)
   - 관련 파일 상태 (200토큰)
   - 코딩 규칙 (100토큰)
   - 총 800토큰 (전체 128K가 아닌!)

3. Code Claude 실행:
   - auth 세션에서 실행 (이전 auth 작업 기억)
   - 큐레이션된 800토큰만 받음
   - 기존 패턴 이해하고 일관성 있게 구현

4. 결과 흡수:
   - Code Claude의 새로운 패턴을 Master가 학습
   - 다음 작업에서 활용
```

## 핵심 이점

1. **Master Claude**: 전체 프로젝트 메모리 + 지능적 압축
2. **Code Claude**: 효율적 실행 + 관련 컨텍스트만 수신
3. **세션 관리**: 프로젝트 영역별 연속성
4. **양방향 학습**: 지속적인 패턴 개선

이 구조로 "매번 새로 시작"의 비효율성을 해결하면서도 각 에이전트의 장점을 극대화했습니다.