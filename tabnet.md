## Introduction

**정형 데이터에서 딥러닝이 어려운 이유**
- 트리 모델의 지배: 정형 데이터에는 트리 기반 모델이 이미 강력함
- DNN의 inductive bias 부족: 정형 데이터에 맞는 귀납적 편향이 없음
- 과도한 파라미터화: hypothesis space가 너무 넓어 좋은 최적점을 찾기 어려움

**그럼에도 딥러닝이 필요한 이유:** 성능 확장성 / End-to-End 학습 / 표현 학습(Representation Learning)

**TabNet의 3가지 핵심 설계**
- Raw Tabular Data를 전처리 없이 직접 입력, Gradient Descent 기반 End-to-End 학습
- Sequential Attention 기반 피처 선택 + Instance-wise Selection
- Interpretability + Self-supervised Learning 지원

---

## Background

**정형 데이터의 특성**
- 피처 간 독립성 (Independent Features)
- 변수 중요도의 극심한 편차 (Feature Saliency)
- 복잡한 결정 경계 (Decision Manifolds)

**Decision Tree의 장점** — Hyperplane boundaries / Automatic Feature Selection / Interpretability
→ TabNet은 이 장점들을 딥러닝으로 구현하는 것이 목표

**Feature Selection: Global vs Instance-wise**
- Global: 전체 훈련 데이터 기준으로 변수 중요도 단일 평가 (ex. Lasso)
- Instance-wise (TabNet): 샘플마다 중요한 피처를 다르게 선택, Feature selection과 Output mapping을 동시 수행

**Top-down Attention & Sparsemax**
- Top-down Attention: 예측 목표(Task)에 따라 필요한 정보만 선별적으로 탐색
- Sparsemax: 불필요한 피처를 완전히 0으로 만들어 희소 선택 → 해석 가능성 확보 (softmax는 모든 피처에 양수 가중치 부여)

---

## Core Concept

**TabNet = 트리 + 딥러닝의 결합**

| 특성 | 내용 |
|---|---|
| Sequential Processing | 여러 Step에 걸쳐 순차적으로 피처 처리 |
| Instance-wise Selection | 샘플마다 중요 피처를 다르게 선택 |
| Soft Feature Selection | Backpropagation으로 학습 가능한 마스크 M[i] 사용 |

---

## Detailed Architecture

**TabNet Encoder 구조**
- Step-by-step Flow: 이전 단계 정보(a[i-1])로 마스크 생성 → 선택된 피처 처리 → 결과 도출
- 각 Step = Feature Transformer + Attentive Transformer + Mask + Split 블록
- Feature Masking: `f_i = M[i] · f` (곱셈 방식으로 salient한 피처만 통과)

**Attentive Transformer & Feature Masking**
- Prior Scale P[i]: 특정 피처가 이전 단계에서 얼마나 사용됐는지 기록 → 다양한 피처 탐색 유도
- Sparsemax: 중요도 낮은 피처 가중치를 완전히 0으로 → Interpretability 확보

**Feature Transformer**
- Shared layers(2개) + Step-dependent layers(2개) 결합
- FC → BN → GLU 구조: GLU가 어떤 정보를 통과/차단할지 비선형적으로 제어
- Residual Connection에 √0.5 스케일링 적용 → 깊은 네트워크에서도 안정적 학습
- 출력: `[d[i], a[i]]` — d[i]는 현재 단계 특징, a[i]는 다음에 무엇을 볼지 결정 힌트

---

## Self-supervised Learning

- Unsupervised pre-training: 일부 피처를 마스킹 후 TabNet encoder+decoder로 복원 학습
- Supervised fine-tuning: 사전 학습된 encoder를 분류/회귀 태스크에 적용
- 레이블이 부족한 상황에서 성능 극대화 가능

---

## Training

**희소성 정규화 (Sparsity Regularization)**

$$L_{total} = L_{task} + \lambda_{sparse} L_{sparse}$$

- $L_{sparse}$: 각 단계·피처별 마스크 가중치에 음의 엔트로피 적용 → 마스크를 더 희소하게 만들어 Interpretability 확보

**최적화 & 안정성**
- Ghost Batch Normalization (GBN): 작은 가상 배치로 나눠 BN 적용 → 정형 데이터 학습 불안정성 완화
- Learning Rate Scheduling: 초기 높은 lr에서 점진적으로 감소

---

## Experiments

**Instance-wise feature selection 검증**
- 전역 피처가 중요한 케이스(Syn2)와 인스턴스별로 피처가 달라지는 케이스(Syn4) 모두에서 강한 성능
- 파라미터 수 26~31K로 101K짜리 INVASE와 대등하거나 우세

**Real-world 데이터셋 성능**

| 데이터셋 | TabNet 성능 | 비고 |
|---|---|---|
| Forest Cover Type | 96.99% | AutoML Tables(94.95%) 능가 |
| Poker Hand | 99.2% | 높은 비선형성, XGBoost(71.1%) 대비 압도 |
| Sarcos (회귀) | MSE 0.14 (TabNet-L) | 같은 크기 모델 대비 최고 성능 |
| Rossmann Store | MSE 485.12 | 시간 피처 인스턴스별 처리 강점 |
| Higgs Boson | 78.84% (TabNet-M) | Sparse MLP(81K)와 동일 성능을 0.66M으로 달성 |

**Interpretability 검증**
- Syn2: X3-X6에만 집중, 무관한 피처 마스크 ≈ 0으로 수렴
- Syn4: X11 값에 따라 X1-X2 또는 X3-X6으로 인스턴스별 피처 전환
- 버섯 식용성 예측: "Odor" 피처에 43% 중요도 부여 (LIME, DeepLift 등은 30% 미만)
- 소득 예측: "Age"를 가장 중요한 변수로 선택, T-SNE에서 학습된 표현과 일치 확인

**Self-supervised Learning**
실험 설계: TabNet-M / Higgs Boson 데이터셋 / 15번 반복 평균

| 학습 데이터 크기 | Supervised | With pre-training | 향상폭 |
|---|---|---|---|
| 1k | 57.47% | 61.37% | +3.9% |
| 10k | 66.66% | 68.06% | +1.4% |
| 100k | 72.92% | 73.19% | +0.27% |

- 레이블 데이터가 적을수록 사전학습 효과가 더 크게 나타남
- 사전학습으로 수렴 속도도 빨라짐
- 활용 예: Continual learning (매일 새로운 거래 데이터가 쌓이는 금융 사기 탐지), Domain adaptation (의료 데이터로 사전학습 후 특정 병원 데이터에 빠르게 적용)

**Performance on KDD Datasets**
CRM 분류 + 소득 예측 (Appetency / Churn / Upselling / Census)

| 모델 | Appetency | Churn | Upselling | Census |
|---|---|---|---|---|
| XGBoost | 98.2 | 92.7 | 95.1 | 95.8 |
| CatBoost | 98.2 | 92.8 | 95.1 | 95.7 |
| TabNet | 98.2 | 92.7 | 95.0 | 95.5 |

- 이미 단순 모델로도 높은 정확도가 나오는 **성능 포화 상태**이기 때문에 TabNet도 XGBoost, CatBoost와 거의 동일한 성능
- Adult Census Income 피처 중요도 순위는 SHAP, XGBoost 등 다른 방법들과 유사한 경향 (Age, Education 등이 상위권)

---

## Limitations & Future Directions

**TabNet의 주요 한계**

1. 하이퍼파라미터 민감성 — 최적 성능을 내려면 많은 튜닝 필요, 데이터셋마다 최적값이 달라 범용적 사용 어려움
2. 높은 계산 비용 — 순차적 어텐션 구조 특성상 학습 시간이 오래 걸리고 대규모 배치 크기 필요
3. 노이즈에 민감 — sparsemax로 변수를 강하게 선택하는 구조라 노이즈 많은 데이터에서 잘못된 변수를 선택할 위험 존재

**후속 연구: FT-Transformer**
- 수치형·범주형 모든 피처를 동일 차원의 토큰으로 변환하여 Transformer로 피처 간 관계 학습
- TabNet: 순차적으로 피처 선택, 중요한 피처만 집중 처리
- FT-Transformer: 모든 피처를 동시에 처리하여 피처 간 고차원 상호작용 포착
- 해석 가능성: FT-Transformer는 어텐션 맵 평균화로, TabNet은 마스크 M[i] 시각화로 각각 구현