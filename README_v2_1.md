# 인명 탐지 로봇 시뮬레이션 V2.1

## 4,000만 배 조건

금속/사람 기본 신호 계수의 분포 평균 비율을 40,000,000:1로 설정했습니다. 사람 신호는 유지하고 금속 계수만 확대합니다.
거리·방향·크기에 따른 실제 센서 신호 비율은 일정하지 않습니다. 실측 물성비가 아니라 극단 조건의 합성 실험입니다.
로봇 인식 자기장 색은 확대된 세기 기준의 제곱근 스케일로 표시합니다. 센서값·CNN 입력 자체는 색상 때문에 정규화하거나 축소하지 않습니다.
CNN과 핑 조건은 재학습·변경하지 않았으므로 이 조건에서의 정확도를 보장하지 않습니다.
아래의 기존 검증 수치는 별도 표기가 없다면 4,000만 배 전환 이전 결과입니다.


## 실행

```powershell
python desktop_app_v2_1.py
python local_apps_v2_1/open_apps_v2_1.py
streamlit run app_v2_1.py
uvicorn server_v2_1:app_v2_1 --host 127.0.0.1 --port 8000
```

## 설정이 바꾸는 것

V2의 신호 형태와 사람·금속의 상대적인 반응, 시간 변화, 잡음, CNN 가중치와 핑 판단 기준을 유지합니다.
V2.1 설정은 물체 반응의 공통 배율과 실제 센서 범위에만 반영됩니다.
별도 실험 모델 선택은 필요하지 않습니다. 기본 설정은 반응 1배·범위 16이며 V2와 동일합니다.

물체 신호는 배경 [24,-7,39]를 제외한 V2 성분에만 같은 배율을 곱합니다.
사람만 강하게 만들거나 금속만 약하게 만들지 않습니다. 배경 자기장과 기존 잡음은 확대하지 않습니다.
이는 능동 탐지의 송신·대상 반응을 기존 합성 신호 계수로 표현하는 모델을 유지하는 방식입니다.
실제 와전류·수신 파형·물성 계산으로 전체 신호를 교체하는 방식은 기본 앱에 사용하지 않습니다.

## 자기장 설정

**예상값** 또는 **실험값**을 선택합니다.

- 예상값: 송신 주파수·전류, 송수신 코일의 반지름·감은 횟수·거리로 결과 전압을 자동 계산합니다.
- 실험값: 같은 설계 조건과 수신 코일에서 측정한 기본파 RMS 전압(mV)을 입력합니다. 서로 다른 거리 두 곳 이상이 필요합니다. 미측정 전압은 비워 둡니다.

증폭 전·후 선택과 전자 증폭 배율 입력은 없습니다.
외부 증폭기로 측정했다면 그 배율로 나눈 코일 입력 전압을 넣습니다.
예상값 모드에서는 이전 실험값이나 측정 CSV를 사용하지 않습니다.
설정은 잡음의 추가·변경에 사용하지 않으며 기존 환경별 잡음을 유지합니다.

아래 버튼은 두 개입니다.

- **적용**: 설정을 반영하고 같은 seed·환경으로 다시 시작합니다. 센서·CNN·핑·경로 기록을 초기화합니다.
- **예상값으로 처음부터 초기화**: 예상값과 최초 설계 조건으로 돌아갑니다. 주파수 2000 Hz, 전류 50 mA RMS, 송수신 코일 각각 반지름 5 cm·100회, 중계 꺼짐입니다. 반응 1배·범위 16으로 복원됩니다.

입력 중에는 전압 미리보기만 계산하며 **적용** 전에는 시뮬레이션이 바뀌지 않습니다.
설정은 현재 실행 또는 웹 접속 세션에만 적용되며 원본 설정 파일을 덮어쓰지 않습니다.

## 반응 배율과 센서 범위

기준 거리는 30 cm입니다. 기준 설계의 이론 수신 전압은 약 2.650750 mV RMS입니다.
현재 설계의 같은 거리 이론 전압을 이 기준 전압으로 나눠 설계 배율을 구합니다.
주파수·전류·코일 크기·감은 횟수는 코일 상호 인덕턴스와 유도 전압 계산에 반영됩니다.

실험값 모드에서는 각 측정점의 (측정 전압/측정 전류)를 현재 설계의 이론 전달값으로 나눈 보정비를 구합니다.
그 보정비의 중앙값을 하나의 대표값으로 사용합니다. 거리별 실측 곡선 전체를 공간에 재현하는 것은 아닙니다.
반응 배율 S = 설계 배율 × 실험 보정 대표값 × 중계 전압 크기 배율입니다.
코일 간 전달 실험으로 인체의 와전류 반응 계수를 측정한 것은 아닙니다.

센서 범위 R = 16 × S의 세제곱근입니다.
이 관계는 일정 감도 경계를 표현하기 위한 합성 가정이며 실제 탐지 거리로 실측 검증한 공식은 아닙니다.
예를 들어 나머지 조건이 같으면 전류 25/50/100 mA에서 반응은 0.5/1/2배, 범위는 약 12.70/16/20.16입니다.
범위는 격자 측정을 유지할 수 있는 최소 2부터 전체 지도 대각선까지 제한하고 제한 여부를 데이터에 기록합니다.
시뮬레이션 좌표 단위이며 실물 탐지 거리(m)를 의미하지 않습니다.

실제 자기장 스캔, 지형 거리 센서, 관측 지도, 위치 추정의 반경 구간, 화면 센서 원에 같은 범위를 사용합니다.
실제 사람·금속 좌표는 시뮬레이터의 신호 계산 및 평가에만 사용하고 CNN에는 관측 스캔만 전달합니다.
기존 모델이므로 설정을 크게 바꾼 환경에서 탐지 성능이 유지된다는 보장은 없습니다. 정확도는 별도 평가해야 합니다.

## 코일 중계 설정

별도 **코일 중계 설정**에서 사용 여부·개수(1~8)·이웃 장치 중심 간격(cm)·전달 방향에 정렬할 축(X/Y/Z)을 입력합니다.
예상값/실험값 구분 없이 기존 L/R/C 가정과 상호 인덕턴스로 항상 계산합니다.
개수는 3축 중계 장치 수이며 선택 축의 코일 한 쌍이 계산에 사용됩니다.
전체 송신–수신 거리는 (중계 장치 수+1)×간격입니다.
중계 없는 전압과 현재 전압, 크기 배율, 위상을 표시합니다. 신호가 감소할 수도 있습니다.

**적용**하면 전압의 크기 배율만 공통 반응 배율에 포함합니다. 위상 변화는 표시만 하며 기존 신호 파형을 바꾸지 않습니다.
지도에 실제 코일을 배치하거나 물체마다 송신·반사 경로를 새로 계산하지 않습니다.
실측 전 예상 회로값이며 L/R/C는 calibration/coil_profile_v2_1.json의 relay_axes에 있습니다.

## 검증

```powershell
python -m evaluation.signal_scaling_check_v2_1
python -m evaluation.v2_baseline_regression_v2_1
python -m evaluation.mi_settings_check_v2_1
```

기본 설정과 V2의 센서 스캔·이동·확률·핑·구조 결과를 같은 seed로 비교합니다.
설정 변경 검사는 세기 배율, 사람/금속 공통 적용, 실제 측정·화면·위치 추정 범위, 적용·초기화를 확인합니다.
동작 검사는 탐지 정확도 향상을 증명하는 실험과 구분합니다.

자기장 관측과 머신러닝을 이용해 이동하는 사람을 탐지하고, 자율 탐색 로봇이 구조 신호 영역을 순회하는 시뮬레이터입니다.

```powershell
python -m pip install -r requirements_v2_1.txt
python desktop_app_v2_1.py
streamlit run app_v2_1.py
```

```powershell
python -m training.generate_human_dataset_v2_1
python -m training.train_human_detector_v2_1

python -m training.collect_detection_failures_v2_1 --output data/detection_failures_v1_v2_1.npz
python -m training.generate_hard_human_dataset_v2_1 --output data/human_temporal_v4_v2_1.npz --episodes 480 --steps 120 --failure-file data/detection_failures_v1_v2_1.npz
python -m training.train_temporal_human_models_v2_1 --data data/human_temporal_v4_v2_1.npz --output-dir models/human_detector_v4 --device cpu

python -m training.train_coverage_agent_v2_1 --num-envs 6 --max-steps 1000 --expert-steps 6000 --behavior-epochs 8 --dagger-rounds 1 --dagger-steps 3000 --dagger-epochs 3 --updates 0 --device cpu
```

`generate_human_dataset`는 과거 magnetic scan을 누적해 정적인 금속장 후보를 추정하고, 현재 합성장과 금속 예상장의 차이를 `physics_features`로 저장합니다. 사람 위치는 label 생성에만 사용됩니다.

학습 결과는 `models/human_detector/best_human_transition_detector_v2_1.pth`에 저장되며 scan, motion, physics feature의 train episode 정규화 파라미터를 함께 포함합니다.

길찾기 모델은 관측 누적 지도와 거리 센서만 입력으로 사용하며, `models/coverage_agent/best_coverage_agent_v2_1.pth`에 저장됩니다. 학습 환경은 도심 침수 지역, 산업 시설, 주거 지역을 순환합니다.

`train_temporal_human_models`는 실패 사례를 포함한 hard-example 데이터로 사람 존재 detector와 8×16 고해상도 위치 localizer를 각각 학습합니다. `integration.temporal_human_detector_runtime_v2_1.TemporalHumanDetectorRuntime`은 같은 센서 특징을 사용해 실시간 추론을 수행하며, 현재 기본 앱은 검증된 기존 detector를 유지합니다.

Coverage 런타임은 탐색 완료 후 최근 방문 셀과 누적 방문 횟수를 이용해 patrol 목표를 분산하고, 목표 방향 유지와 최소 회전 조건으로 좁은 영역에서의 반복 선회와 급격한 방향 반전을 줄입니다.

센서 최적화는 기존 측정 결과를 유지하면서 거리 센서와 인식 지도 갱신을 배열 연산으로 처리하고, 센서 방향·샘플 좌표를 캐시합니다.

## 웹 실행

Streamlit Community Cloud에서 저장소를 연결하고 실행 파일로 `app_v2_1.py`를 선택합니다.

FastAPI Canvas 웹 앱은 다음 명령으로 실행합니다.

    uvicorn server_v2_1:app_v2_1 --host 0.0.0.0 --port 8000

Render에서는 저장소의 `render_v2_1.yaml`을 Blueprint로 연결합니다.

## 라이선스

MIT License

[4,000만 배 핑 검사 결과](EXTREME_RATIO_RESULTS.md)
