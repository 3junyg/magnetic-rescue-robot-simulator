# Magnetic Rescue Robot Simulator — V2

자기장 관측으로 사람의 존재와 위치를 추정하고, 로봇이 환경을 탐색하는 Python 시뮬레이터입니다. V2는 탐지 후보 재검증, 핑 판단 분리, 좁은 통로 이동, 구역별 순찰과 짧은 왕복 방지를 추가했습니다.

이 버전의 자기장은 합성 모델입니다. 시뮬레이션 내 분류 성적을 실제 인체 탐지 성능으로 해석하지 않습니다.

## 버전

- [V1 코드](https://github.com/3junyg/magnetic-rescue-robot-simulator/tree/v1.0.0)
- [V2 코드](https://github.com/3junyg/magnetic-rescue-robot-simulator/tree/v2.0.0)
- [Release 다운로드](https://github.com/3junyg/magnetic-rescue-robot-simulator/releases)

V2는 `v2` 브랜치와 `v2.0.0` 태그로 배포합니다. 기존 `main`과 연결된 사이트는 별도로 관리합니다.

## 시작하기

Python 3.12 환경에서 검증했습니다. 저장소 루트에서 실행합니다. 배포에 필요한 학습 모델이 포함돼 있어 재학습할 필요가 없습니다.

```powershell
git clone --branch v2.0.0 --single-branch https://github.com/3junyg/magnetic-rescue-robot-simulator.git
cd magnetic-rescue-robot-simulator
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python local_apps_v2/open_apps_v2.py
```

개별 앱도 바로 실행할 수 있습니다.

```powershell
python desktop_app_v2.py
python smooth_perception_app_v2.py
streamlit run app_v2.py
```

앱에서 `CNN Agent` 모드를 선택하고 실행합니다. 실행·정지·리셋·랜덤 배치·로봇 인식 화면 전환을 지원합니다. 로컬 앱 선택 메뉴에는 기본, Smooth, 추가 노이즈, 노이즈 대응 모델, 좁은 통로 시각화가 있습니다. 로컬 창에는 GUI와 Tk 지원 환경이 필요합니다.

## 현재 작동 구조

```text
합성 환경 / 센서
  ├─ 자기장 관측 → 변화 예측·잔차 CNN → 후보 선택 → 후보 검증 CNN → 핑 결정
  │                                                        └─ 동일 좌표의 시간 변화 검사
  └─ 거리 관측 → 벽 인식·관측 지도 → 경로 계획·순찰·반복 감시 → 로봇 action
핑 → 시뮬레이터의 구조 판정 → 해당 사람의 이동·자기장 중단
```

탐지기는 `RobotObservation`에 담긴 자기장·로봇 이동·거리 관측을 사용합니다. 사람·금속의 실제 좌표는 환경 생성, 학습 정답, 평가 및 구조 판정에서 사용합니다. 로봇 자신의 위치·방향은 현재 시뮬레이션에서 이상적인 자기 위치 추정값으로 취급합니다.

학습된 탐색 정책을 불러오지만, 현재 경로 계획·구역 순찰·탈출·짧은 왕복 방지는 규칙 기반 제어와 결합돼 있습니다.

## 폴더

| 경로 | 역할 |
|---|---|
| `simulation/` | 환경, 엔티티, 자기장 모델, 센서, 로봇 인식 지도 |
| `integration/` | 탐지·핑·이동 제어·구조 처리 연결 |
| `navigation/` | 학습 정책, 안전 경로, 순찰 스케줄러, 반복 이동 감시 |
| `ml/`, `models/` | 신경망·특징 추출 코드와 배포 모델 |
| `training/` | 합성 데이터 생성과 학습 스크립트 |
| `ui/`, `local_apps_v2/` | Matplotlib 화면과 PC용 실행 메뉴 |
| `evaluation/` | headless 검증, 성능 비교, 위치 평가 |
| `web_static/` | FastAPI 웹 화면의 HTML·JavaScript·CSS |
| `docs/` | 구조 설명, V1 대비 변경 사항, 검증 결과 |

## 상세 문서

- [구조와 작동 방식](docs/ARCHITECTURE_v2.md)
- [V1 대비 변경 사항](docs/CHANGELOG_v2.md)
- [모델 학습·검증 결과와 해석 범위](docs/VALIDATION_v2.md)
- [V2 Release 설명](docs/RELEASE_NOTES_v2.md)

## 검증 실행

```powershell
python -m evaluation.verifier_smoke_v2
python -m evaluation.narrow_navigation_v2 --bend --steps 300 --output evaluation/results/narrow_check_v2.json
python -m evaluation.navigation_cycle_v2 --seeds 73 99 --steps 800 --output evaluation/results/cycle_check_v2.json
python -m evaluation.patrol_coverage_v2 --only-new --seeds 73 --kinds split random --steps 2200 --output evaluation/results/patrol_check_v2.json
```

## 웹 서버

```powershell
uvicorn server_v2:app_v2 --host 127.0.0.1 --port 8000
```

브라우저에서 `http://127.0.0.1:8000`에 접속합니다. Streamlit 실행 파일은 `app_v2.py`입니다. Render 구성 예시는 `render_v2.yaml`이며, 이번 Release는 기존 Render 서비스의 배포 대상을 변경하지 않습니다.

합성 학습 데이터, 중간 체크포인트, 캐시와 대용량 실행 추적은 배포본에서 제외합니다. 학습을 다시 진행하려면 `training/`의 생성기를 먼저 실행해야 합니다. 라이선스는 MIT입니다.
