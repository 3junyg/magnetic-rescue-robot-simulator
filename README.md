# 인명 탐지 로봇 시뮬레이션

자기장 관측과 머신러닝을 이용해 이동하는 사람을 탐지하고, 자율 탐색 로봇이 구조 신호 영역을 순회하는 시뮬레이터입니다.

```powershell
python -m pip install -r requirements.txt
python desktop_app.py
streamlit run app.py
```

```powershell
python -m training.generate_human_dataset
python -m training.train_human_detector

python -m training.collect_detection_failures --output data/detection_failures_v1.npz
python -m training.generate_hard_human_dataset --output data/human_temporal_v4.npz --episodes 480 --steps 120 --failure-file data/detection_failures_v1.npz
python -m training.train_temporal_human_models --data data/human_temporal_v4.npz --output-dir models/human_detector_v4 --device cpu

python -m training.train_coverage_agent --num-envs 6 --max-steps 1000 --expert-steps 6000 --behavior-epochs 8 --dagger-rounds 1 --dagger-steps 3000 --dagger-epochs 3 --updates 0 --device cpu
```

`generate_human_dataset`는 과거 magnetic scan을 누적해 정적인 금속장 후보를 추정하고, 현재 합성장과 금속 예상장의 차이를 `physics_features`로 저장합니다. 사람 위치는 label 생성에만 사용됩니다.

학습 결과는 `models/human_detector/best_human_transition_detector.pth`에 저장되며 scan, motion, physics feature의 train episode 정규화 파라미터를 함께 포함합니다.

길찾기 모델은 관측 누적 지도와 거리 센서만 입력으로 사용하며, `models/coverage_agent/best_coverage_agent.pth`에 저장됩니다. 학습 환경은 도심 침수 지역, 산업 시설, 주거 지역을 순환합니다.

`train_temporal_human_models`는 실패 사례를 포함한 hard-example 데이터로 사람 존재 detector와 8×16 고해상도 위치 localizer를 각각 학습합니다. `integration.temporal_human_detector_runtime.TemporalHumanDetectorRuntime`은 같은 센서 특징을 사용해 실시간 추론을 수행하며, 현재 기본 앱은 검증된 기존 detector를 유지합니다.

Coverage 런타임은 탐색 완료 후 최근 방문 셀과 누적 방문 횟수를 이용해 patrol 목표를 분산하고, 목표 방향 유지와 최소 회전 조건으로 좁은 영역에서의 반복 선회와 급격한 방향 반전을 줄입니다.

센서 최적화는 기존 측정 결과를 유지하면서 거리 센서와 인식 지도 갱신을 배열 연산으로 처리하고, 센서 방향·샘플 좌표를 캐시합니다.

## 웹 실행

Streamlit Community Cloud에서 저장소를 연결하고 실행 파일로 `app.py`를 선택합니다.

## 라이선스

MIT License
