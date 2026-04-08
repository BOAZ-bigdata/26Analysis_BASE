## Introduction

**표 형식 데이터의 문제점**
- 데이터 유형이 다양하고 (불리언, 범주형, 수치형 등) Imbalanced / Missing / Outlier 존재
- 딥러닝: 이질성 때문에 적용 어려움
- 머신러닝: Transfer 어렵고, 기울기 전파 불가로 신경망과 결합 힘듦

**TabPFN 성능**
- 적용 범위: 최대 10,000 샘플 × 500 특징
- 분류 5,140×, 회귀 30,000× 빠름 (vs Gradient-boosted 트리)

---

## Principled In-Context Learning (ICL)

대형 언어모델의 ICL과 동일한 메커니즘을 표 데이터에 적용

| 단계 | 내용 |
|------|------|
| 데이터 생성 | 수백만 개의 합성 표 데이터셋 생성 |
| 사전 학습 | 마스킹 목표로 PFN 학습 (개발 중 단 한 번) |
| 실제 예측 | 학습 샘플을 컨텍스트로 제공 → 미지 데이터셋 예측 |

- 추론 시 전체 데이터셋을 입력으로 받아 단일 순방향 전달로 훈련 + 예측 동시 수행
- 파인튜닝 없이 임의의 미지 데이터셋에 바로 적용 가능

---

## Architecture

**2D TabPFN Layer (×12)**
- 1D feature attention (열 방향) → 1D sample attention (행 방향) → MLP → 확률 분포 출력
- 양방향 어텐션으로 샘플과 특징 순서에 무관 (순서 불변 설계)

**적합-예측 설정 (Fit-Predict)**
- 훈련 세트에 대한 ICL을 한 번 수행 후 결과 캐싱 → 여러 테스트 세트 재사용 가능
- CPU 약 300배, GPU 6배 속도 향상
- 특징 10배 증가 시 CPU 800배, GPU 30배 향상

**하드웨어 최적화**
- Flash Attention2, 활성화 체크포인트, half precision 적용
- 메모리 요구 사항 1/4로 감소 → H100 기준 최대 50,000셀 데이터셋 처리 가능

---

## Synthetic Data (인과 모델 기반)

실제 공개 데이터 대신 합성 데이터를 사용하는 이유
- 개인정보·저작권 침해 방지
- 테스트 데이터로 훈련 데이터 오염 방지

**생성 파이프라인**
1. 데이터셋 크기·특징 수·난이도 등 하이퍼파라미터 샘플링
2. 방향성 비순환 그래프(DAG) 구성
3. 가우시안 노이즈 추가 + Kumaraswamy 분포 워핑 등 후처리
4. 훈련당 약 1억 개의 합성 데이터셋 코퍼스 생성

---

## Experiment

**정성적 분석 (Toy Problem)**
- 선형·비선형·step 함수 등 데이터 구조에 따라 유연하게 모델링
- 단일 값이 아닌 전체 확률 분포 예측 (불확실성 표현 가능)
- CatBoost 대비 매끄러운 곡선 근사, MLP 대비 step 함수도 잘 처리

**정량적 분석**
- Gradient-boosted 결정 트리 대비 전반적으로 우수한 성능
- 소규모~중규모 데이터셋에 최적화

---

## Quantitative Analysis

**실험 설정**
- 데이터셋: AutoML Benchmark, OpenML-CTR23, Kaggle 5개 등 (분류 29개 + 회귀 28개, 최대 10,000샘플 × 500특징 × 10클래스)
- 비교 모델: Random Forest, XGBoost, CatBoost, LightGBM, Linear, SVM, MLP
- 평가 지표: 분류 → ROC AUC / 회귀 → R², negative RMSE
- 실험 방식: 10개 random seed, 90/10 train-test split, random search + 5-fold CV로 튜닝 (30초~4시간), 추론 시 8번 앙상블

---

## Comparison with State-of-the-art Baselines

**vs CatBoost (가장 강한 baseline)**

| | TabPFN | CatBoost |
|---|---|---|
| 분류 (기본) | 0.939 | 0.752 |
| 분류 (튜닝) | 0.952 | 0.822 |
| 회귀 (기본) | 0.923 | 0.872 |
| 회귀 (튜닝) | 0.968 | 0.875 |

**튜닝 시간 vs 성능**
- 하이퍼파라미터 탐색 시간을 늘릴수록 baseline 성능은 올라가지만, default TabPFN이 여전히 강함
- Kaggle, Tabular Playground 등 추가 벤치마크에서도 동일한 경향

---

## Evaluating Diverse Data Attributes

**강건성 실험** (데이터를 일부러 망가뜨림)
- 불필요한 feature 추가, 극단적 outlier 추가, sample/feature 수 감소
- → 지저분한 환경에서도 비교적 안정적으로 동작

**속성별 subgroup 비교**
- 결측치 유무, 범주형 변수 유무, sample 수, feature 수별로 분리 평가
- → 특정 상황에서도 안정적으로 동작

---

## Comparison with Tuned Ensemble Methods

**AutoGluon vs TabPFN(PHE)**

| | AutoGluon | TabPFN(PHE) |
|---|---|---|
| 방식 | 여러 모델 stacked ensemble + 튜닝 | TabPFN 여러 개 + PHE 앙상블 |
| 분류 소요 시간 | 최대 4시간 | 2.8초 |
| 분류 성능 | 낮음 | 우세 |
| 회귀 성능 | 최대 4시간 필요 | 최소 300초에서도 우세 |

- 분류에서는 default TabPFN(2.8초)만으로도 AutoGluon(4시간) 대비 우세
- TabPFN도 단일 모델에서 끝나는 게 아니라 앙상블로 더 강화 가능

---

## Foundation Model with Interpretability

TabPFN은 예측 외에도 다양한 확장 기능을 지원한다.

**Density Estimation** — 데이터 분포 추정 (사기 탐지, 이상치 탐지 등에 활용)

**Synthetic Data Generation** — 실제 데이터 특성을 흉내 낸 샘플 생성 (data augmentation, privacy-preserving sharing)

**Embedding** — 내부 표현이 단순 압축 벡터가 아니라 분류에 유리한 구조를 담음 → imputation(결측값 채우기), clustering(label 없이 군집화)에 활용 가능

**Fine-tuning** — neural architecture이기 때문에 특정 데이터셋/태스크에 맞춰 fine-tuning 가능, 관련 태스크 간 지식 transfer 가능 → "TabPFN을 foundation model로 보겠다"는 주장

---

## Conclusion

- ICL 기반으로 효율적인 알고리즘을 자동 탐색, tabular data modeling의 큰 전환점
- 최대 10,000 샘플 × 500 특징 범위에서 기존 방법들을 능가

Future Work: 더 큰 데이터셋 확장 / data drift 처리 / 관련 tabular task 간 fine-tuning 연구

---
