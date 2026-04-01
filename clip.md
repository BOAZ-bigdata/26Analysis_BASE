# CLIP

## 배경
OpenAI의 CLIP(Contrastive Language-Image Pre-training)은 이미지→텍스트 방향의 멀티모달 모델이다. 핵심 동기는 NLP처럼 대규모 인터넷 데이터로 범용 시각 모델을 학습하는 것이다.

---

## 선행 연구

**ConVIRT** — CLIP의 아키텍처 기반. 의료 도메인 특화 및 데이터 규모 부족이 한계였다. CLIP은 이를 처음부터 학습(train from scratch), 선형 투영만 사용, 더 적은 augmentation으로 극복했다.

**VirTex** — 이미지 캡셔닝 기반 사전학습. 생성 목적함수와 스케일링 어려움이 한계였다. CLIP은 대조 목적함수(Contrastive Objective)로 학습 효율을 크게 높였다.

---

## 데이터셋
기존 MS-COCO(10만 장)와 YFCC100M(품질 불균일)의 한계를 극복하기 위해 인터넷에서 수집한 WIT(Web Image Text) 4억 쌍을 구축했다.

---

## 학습 방법

**대조 사전학습** — 이미지 인코더(ResNet 또는 ViT)와 텍스트 인코더(CBOW 또는 Transformer)로 임베딩을 추출한 뒤, N×N 유사도 행렬에서 대각선 쌍만 높이는 대칭 손실 함수를 사용한다.

**프롬프트 엔지니어링** — 단순 레이블 대신 `"A photo of a {label}"`로 변환해 ImageNet 정확도를 1.3% 향상시켰다. 여러 프롬프트를 앙상블하면 추가로 3.5% 향상된다.

---

## 실험 결과

**Zero-shot 성능** — StanfordCars +28.9%, Food101 +22.5% 등 다수 데이터셋에서 ResNet50 Linear Probe를 능가한다. 단, MNIST·EuroSAT 등 특수 도메인에서는 성능이 낮다.

**분포 변화 강건성** — ImageNet으로 학습한 ResNet은 분포 변화 시 성능이 크게 떨어지는 반면, CLIP은 다양한 데이터로 학습해 강건성 격차(Robustness Gap)를 최대 75% 줄인다.

**인간 비교** — Oxford IIT Pets 기준 Zero-shot CLIP 93.5% vs 인간 53.7%. 오류 패턴도 인간과 유사하다.

**데이터 중복 분석** — 사전학습 데이터와 평가 데이터 간 중복이 성능에 미치는 영향은 통계적으로 유의미하지 않다(p > 0.05).

---

## 한계

Performance Gap : SOTA 대비 약 1000배 연산 필요 
Task Weakness : 세밀한 분류 및 추상적 추론 취약
OOD Sensitivity : 학습 외 데이터 일반화 부족 
Data Inefficiency : 128억 장 학습 필요 (405년치) 

---

## 사회적 영향

필터링되지 않은 인터넷 데이터의 편향이 그대로 학습된다. 흑인 얼굴 14% 비인간 분류, 남성 16.5% 범죄 카테고리 오분류, 여성→외모/가사 라벨 편중 등의 문제가 확인된다.

향후 과제로 Characterization, Bias Testing, Policy Surfacing, Failure Analysis가 제시된다.