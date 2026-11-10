# V2 구조와 작동 방식

## 환경과 자기장

`simulation/board_v2.py`의 `Board`가 사람, 금속, 직사각형 장애물, 로봇, 센서와 시뮬레이션 시간을 관리합니다. 기본 장애물은 서로 겹치지 않게 배치합니다.

`simulation/field_model_v2.py`에 `MagneticFieldModel` 인터페이스와 `LegacyMagneticFieldModel`, `EnhancedMagneticFieldModel`을 분리했습니다. `field_at()`과 `scan_at()`을 통해 자기장 생성 방식을 교체할 수 있습니다. 기본 설정은 enhanced 모델입니다.

사람 신호는 거리, heading, 호흡·미세 진동에 해당하는 합성 시간 변화, 이동 속도·방향을 반영합니다. 금속과 배경을 합성한 전체 장에 센서 오차를 추가합니다. 실제 인체 자기장이나 수중 전자기 전달을 실측 보정한 모델은 아닙니다.

## 관측과 정답 상태의 구분

`RobotObservation`에는 시각, 로봇 자신의 위치·방향·이동 여부, 중심 자기장 측정, 범위 내 자기장 스캔, 거리 스캔이 들어갑니다. 사람·금속의 실제 좌표 목록과 개별 신호 성분은 탐지 입력에 넣지 않습니다.

로봇 자신의 위치는 센서 스캔을 동일 공간 좌표에 정렬하기 위해 사용합니다. 현재는 이상적인 위치 추정을 가정하므로 실환경의 위치 오차까지 검증한 결과가 아닙니다.

`HumanDetectorRuntime.process_board()`는 관측을 추출하고 추정 핑을 화면에 반영하는 연결 함수입니다. 핵심 분류는 `process_observation()`에서 수행합니다. 이동 런타임도 로봇 상태·거리 스캔·누적 인식 지도를 사용합니다. `Board` 전체를 전달하는 연결 계층이 있으므로 이 구분은 프로세스 권한 격리가 아니라 입력 구성과 코드 사용 규칙으로 유지됩니다.

실제 사람 좌표는 데이터 생성기의 label, 평가 코드, 실제 환경 디버그 화면, `RescueManager`의 구조 판정에 사용합니다.

## 사람 탐지와 핑

1. `MetalFieldEstimator`가 과거 합성 자기장 관측에서 정적 장을 추정합니다.
2. `MagneticTransitionPredictor`가 이전 스캔과 로봇 이동 정보로 예상 변화를 계산합니다.
3. `ResidualHumanCNN`이 예측 잔차·현재 신호·시간 차이·정적 장 잔차 등의 연속 특징으로 존재 확률, 인원 범주, 위치 점수를 계산합니다.
4. `DetectionCandidateSelector`가 위치 후보를 만듭니다.
5. `CandidateVerifierRuntime`이 후보 주변의 공간·시간 자기장 패치를 검사합니다.
6. `PingDecisionAgent`가 확률, 반복 관측, 위치 안정성, 정적 장 후보, 중복 억제 등으로 핑을 결정합니다. 증거 계산과 확정 판단은 별도 모듈로 나뉩니다.
7. `WorldDeltaPingAgent`는 연속 스캔의 동일한 공간 좌표에서 시간 변화를 검사하는 추가 경로입니다. 공통 변화 제거, 지속·간격·거리 조건과 같은 후보 검증 CNN을 사용합니다.

후보 검증 모델은 시점별 2D CNN과 시간축 1D CNN을 결합한 하나의 모델입니다. 최근 8시점, 후보 주변 5×5칸, 칸별 8채널과 로봇 이동 특징을 사용합니다. CNN 층 수나 모듈 수가 독립적인 성능 검증 횟수를 뜻하지 않습니다.

## 이동과 탐색

`CoverageBoardRuntime`은 학습된 `CoverageActorCritic` 정책과 경로 계획기를 결합합니다. 기본 경로 계획은 0.5m 격자이며, 좁은 통로·실패 경로에는 0.25m 정밀 계획을 사용합니다. 관측된 장애물과 로봇 반경을 고려해 여유 공간을 계산합니다.

`ClearancePlanner`는 미관측 목표와 순찰 목표를 위한 경로를 계산합니다. `ControlledBoardStepper`는 목표 방향에 맞춘 제한 회전, 경로 충돌 확인, 짧은 전진을 적용합니다. 연속 회전 후 전진을 강제로 실행하던 방식을 제거했습니다. 모서리에서는 경로를 대각선으로 이어 붙이지 않습니다.

거리 센서는 광선과 직사각형의 교점을 계산합니다. 인식 지도는 관측된 동일 직선 위의 가까운 점들을 연결하며, 센서로 보지 못한 장애물 전체 형상을 복사하지 않습니다.

## 순찰과 반복 이동 방지

`PatrolScheduler`가 8m 구역별 방문 기록과 순찰 주기를 유지합니다. 현재 주기의 미방문 구역을 우선하고, 그중 가까운 목표를 선택합니다. 도착·실패·보류를 구분하며, 다음 주기에도 전체 대상 구역을 다시 순찰합니다.

`NavigationProgressGuard`는 이동량만 보지 않고 관측 진전, 순찰 구역 방문 진전, 일정 범위 내 반복 이동을 확인합니다. 가까이 가도 계속 미관측으로 남는 목표는 잠시 보류합니다. 정체하면 다른 관측 구역으로 이동한 뒤 탐색을 재개합니다.

짧은 직선 왕복에는 별도 검사를 적용합니다. 한 번의 복귀는 허용하지만 새 관측이나 방문 진전 없이 반전이 연속되면 다음 반전 동작을 실행 전에 취소합니다. 보류·반복 기록은 Reset/Randomize에서 초기화됩니다. 관련 값은 `config_v2.py`에 있습니다.

## 화면과 구조 판정

기본 PC 앱은 `desktop_app_v2.py`, Smooth 앱은 `smooth_perception_app_v2.py`, 로컬 실행 메뉴는 `local_apps_v2/open_apps_v2.py`입니다. Streamlit 앱은 `app_v2.py`, FastAPI 앱 객체는 `server_v2:app_v2`입니다.

로봇 인식 화면은 관측된 자기장·벽, 금속 후보, 추정 사람 핑을 표시합니다. 순찰 중에는 방문/미방문 구역, 목표와 계획 경로도 표시합니다. 실제 환경 화면의 사람 위치는 디버그 정보입니다.

`RescueManager`가 실제 사람과 구조 신호의 거리를 평가합니다. 신호 주변에 도달한 사람은 정지하고 자기장을 더 이상 생성하지 않도록 처리합니다. 구조 신호를 유지해 나중에 접근한 사람도 처리할 수 있습니다. 이 판정은 탐지기가 사람 좌표를 읽어 핑을 생성하는 과정과 분리됩니다.

## 배포 모델

| 파일 | 역할 |
|---|---|
| `models/human_detector/best_human_transition_detector_v2.pth` | 기본 자기장 변화 예측·사람 분류 |
| `models/candidate_verifier/best_candidate_verifier_v2.pth` | 기본 후보 재검증 |
| `models/coverage_agent/best_coverage_agent_v2.pth` | 학습된 탐색 정책 |
| `models/human_detector_noise_robust/best_human_transition_detector_v2.pth` | 별도 노이즈 대응 실험 앱 |
| `models/human_detector_v4/best_temporal_human_models_v2.pth` | 기본 앱에 연결하지 않은 temporal 실험 모델 |

배포 파일 이름이 `_v2`인 것과 모델을 V2 전체 환경에서 새로 학습했다는 것은 다릅니다. 최근 이동 개선은 경로 계획과 제어 코드 수정이며 모델 재학습을 의미하지 않습니다.
