## 1. Introduction

**기존 Diffusion Model**
- 정의하기 쉽고 학습이 효율적이지만 고품질 샘플 생성이 어렵다
- DDPM은 parameterized Markov chain을 추가해서 원하는 데이터에 맞는 고품질 샘플을 만드는 모델이다

**Forward Process**
- Markov chain이 점진적으로 가우시안 노이즈를 추가해서 최종적으로 완전한 노이즈 상태를 만든다
- 선명한 원본 이미지(X0)에 노이즈를 섞어서 X1, X2... Xt를 만들어간다

**Reverse Process**
- forward process를 역행하도록 학습한다
- 노이즈 상태인 Xt를 보고 직전 단계 Xt-1을 예측한다

**Markov Chain이 중요한 이유**
- 특정 state의 조건부 확률 분포가 오직 현재 상태에만 의존한다
- 즉 다음 단계를 생성하거나 맞추는 데 바로 직전 단계만 보면 된다

**가우시안 노이즈를 쓰는 이유**
- 노이즈 양이 적으면 조건부 가우시안으로 근사가 충분하다
- 선형 결합이 가능해서 평균과 분산만으로 계산이 편리하다
- 노이즈 제거가 노이즈 추가의 대칭적 구조를 가진다
- 결론적으로 DDPM은 GAN보다 우수한 성능을 보이며 효율적인 신경망 학습이 가능하다

---

## 2. Background

### 2.1 Reverse Process
- **pθ (최종 목표)**: 완전한 노이즈 상태 xT에서 출발해서 t → t-1 단계를 거쳐 가우시안 분포를 따라 이미지를 복원하는 과정 (XT → X0)이다
- 노이즈가 섞인 데이터에서 노이즈를 조금씩 제거하는 과정을 p라고 한다
- p는 학습 가능한 파라미터로 이루어져 있어서 이 신경망을 학습하는 게 핵심이다
- 학습 대상은 **평균(μθ)과 분산(Σθ) 함수**다

### 2.2 Forward Process
$$q(X_t | X_{t-1}) := \mathcal{N}(X_t;\, \sqrt{1-\beta_t}\, X_{t-1},\, \beta_t \cdot I)$$

- **q**: 원본 사진(X0)에서 variance schedule(βt)을 따라 가우시안 노이즈를 추가하는 과정 (X0 → Xt)이다
- **βt**: 매 스텝 t마다 조금씩 증가하는 상수이고, 노이즈를 얼마나 섞을지 결정한다
- 학습시키거나 고정하나 성능이 비슷해서 **고정된 값**을 쓴다
- q는 정해진 βt 비율로 노이즈를 추가하기 때문에 **학습하지 않는다**

### 2.3 Negative Log Likelihood (NLL)
- 모델이 이미지를 잘 복원하도록 학습하기 위한 최종 Loss function이다
- 모델이 training data의 **likelihood(AI가 원본 이미지 X0를 만들어내는 확률)를 최대화**하는 방향으로 학습한다
- 모든 latent 변수를 적분해야 해서 직접 계산이 매우 어렵다
- 그래서 계산이 안 되는 **NLL 대신 ELBO를 활용해서 ELBO를 최대화하도록 학습**한다
- **ELBO**: 완벽하게 특징을 반영하는 데이터 분포를 구하는 건 불가능하니까 하한선을 정해서 근사값을 찾는 방식이다
- **KL Divergence**: p(예측 분포)와 q(정답 분포) 두 분포 간의 차이를 최소화하기 위한 Loss Function의 핵심 요소다. 분포 간 차이가 0에 가까울수록 p가 q를 완벽하게 재현하고 있다는 뜻이다

### 2.4 Closed Form
$$q(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t;\, \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,\, (1-\bar{\alpha}_t)\mathbf{I})$$

- 기존 방식은 Xt를 만들기 위해 노이즈를 t번 반복해서 섞어야 한다 (X100이면 100번)
- Closed Form 덕분에 원본 X0에서 t단계 Xt로 **한 번에 건너뛰어서 학습 속도를 절약**한다
- 즉 **임의의 시간 t에서 바로 샘플링이 가능**하다
- q(xt-1|xt)는 계산하기 어렵지만, **X0을 조건으로 준 forward process posterior q(xt-1|xt,x0)는 쉽게 구할 수 있다**
- 이를 이용해서 KL divergence로 pθ(xt-1|xt)를 학습시킨다

---

## 3. Diffusion Models and Denoising Autoencoders

### 3.1 Forward Process, Lt
- 원본 사진에 노이즈를 추가해서 의도적으로 망가뜨리는 과정(q)이다
- forward process의 분산 βt를 상수로 고정하기 때문에 q에는 학습되는 파라미터가 없어서 **학습 중 Lt는 무시 가능하다**

### 3.2 Reverse Process, Lt-1
- 노이즈에서 사진을 복구하는 과정(p)이다 — ELBO 사용
- 모델이 예측한 이미지의 **평균(μθ)이 수학적으로 계산된 원본의 평균값(μ̃t)과 얼마나 가까운지** 비교한다
- 분산은 고정되어 있어서 학습하지 않는다
- 이미지 전체를 한 번에 생성하는 게 아니라 **노이즈 값(ε)을 찾는 방식으로 모델을 학습**한다
- εθ(xt, t): 특정 시점 t와 이미지 Xt가 주어졌을 때 해당 시점의 노이즈(εt)를 예측하는 네트워크다
- 수식을 이용해서 평균 예측 문제를 **노이즈 예측 문제로 바꿀 수 있어서** loss가 훨씬 간단해진다
- 저자들은 ε을 예측하도록 loss term을 simplification하는 게 성능이 좋다고 말한다 (ablation study)

### 3.3 Algorithms

**Algorithm 1 - Training**
1. 데이터셋에서 원본(X0)을 가져와서 노이즈 섞을 정도 t를 랜덤하게 선택한다
2. 노이즈를 더해나가는 과정에서 네트워크(εθ, pθ)가 t 스텝에서 노이즈(ε)가 얼마만큼 더해졌는지를 학습한다
3. Closed Form으로 원본(X0)만 가지고 효율적으로 학습 가능하다

**Algorithm 2 - Sampling**
1. 네트워크를 학습한 후 가우시안 노이즈에서 시작해서 순차적으로 denoising하는 과정이다
2. 학습 알고리즘과 반대 과정으로 이미지를 복구한다
3. 매 단계마다 미세한 랜덤 노이즈(z)를 더해서 더 풍부한 이미지를 생성하도록 한다

### 3.4 Data Scaling
- {0, 1, ..., 255} 범위의 이미지 데이터를 **[-1, 1]로 스케일링**해서 모델이 계산하기 편한 상태(선형적)로 만든다
- 모델이 생성한 이미지를 다시 정수 픽셀로 변환해서 저장하는 최종 출력 필터 역할을 한다

### 3.5 Simplified Training Objective
$$L_{simple}(\theta) := \mathbb{E}_{t,x_0,\epsilon}\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon,\, t)\|^2\right]$$

- 저자들은 Lt-1 loss term을 위와 같이 **simplification**했다
- **특정 시점의 이미지에 추가된 노이즈를 잘 예측하는 것을 목표**로 학습한다
- 복잡한 가중치를 제거하고 예측 오차만 계산하는 단순한 형태다
- 기존 ELBO 식 앞에 붙은 복잡한 가중치항을 생략해서 큰 t에서도 학습이 잘 되도록 했다
- t가 작을 때(노이즈가 거의 없는 상태) → 학습 강도 낮음
- t가 클 때(노이즈가 많은 상태) → 학습 강도 높음
- **결과적으로 t가 이미지 형태를 잘 다루게 되어서 전체적인 이미지 구도가 더 선명하게 잡힌다**

--- 

## 4. Experiments

### Setting
- **T=1000**: 노이즈를 추가하고 복원하는 과정을 총 1000단계에 걸쳐 수행한다
- **β1=10⁻⁴ → βt=0.02**: 선형적으로 증가하는 값이고, [-1,1] 스케일링에 비해 충분히 작은 값이다. reverse와 forward가 같은 형태를 가진다
- **Forward process 마지막 단계 SNR 최소화**: 거의 순수한 가우시안 상태(~N(0,I))를 만들기 위해서다
- **U-Net backbone** 사용

### 4.1 Sample Quality

**평가 지표 정의**
- **IS (Inception Score)**: 생성 샘플을 분류기로 분류했을 때, 각 샘플은 특정 클래스에 명확하게 속하고 전체적으로는 다양한 클래스 분포를 가질수록 높게 나오는 지표다
- **FID (Fréchet Inception Distance)**: 생성된 샘플 분포와 실제 데이터 분포의 차이를 측정하는 이미지 품질 지표다. 낮을수록 좋다
- **Conditional**: 조건(y)을 반영한 특정 데이터 분포 p(x|y)에서 이미지를 생성하는 방식이다
- **Unconditional**: 전체 데이터 분포 p(x)에서 랜덤하게 이미지를 생성하는 방식이다

**결과**: Unconditional 모델임에도 불구하고 FID가 **3.17**로, 기존 모델과 Conditional 모델보다도 뛰어난 성능을 보인다

### 4.2 Reverse Process Parameterization and Training Objective Ablation

- **μ̃ 예측**: simplified objective(unweighted)에서는 성능이 악화되고, fixed variance일 때 성능이 안정된다
- **ε 예측**: simplified objective에서 더 좋은 성능을 낸다
- 결론적으로 **Lsimple + ε 예측 조합**이 IS 9.46, FID 3.17로 가장 우수하다

### 4.3 Progressive Coding

**NLL과 압축 관점**
- **Codelength**: 데이터 x를 pθ(x)의 확률분포로 압축할 때 필요한 비트 수다
- **NLL**: lossless codelength에 해당하고, 값이 낮을수록 압축 효율이 좋다
- DDPM(Ours)의 NLL은 3.72 bits/dim으로, Sparse Transformer의 2.80 bits/dim보다 높다
- **High quality 이미지를 만들지만, 다른 likelihood-based 생성 모델보다 압축 성능은 부족하다**

**Lossy Compression 관점**
- **Lossy Compression**: 데이터 일부를 제거해서 압축하는 방식이고 원본과 완전히 동일하지 않다
- **Rate**: 데이터 압축 효율이다 (L1+···+LT = 1.78 bits/dim)
- **Distortion**: 최종 생성 결과(X0)와 실제 원본 데이터와의 차이다 (L0 = 1.97 bits/dim, RMSE ≈ 0.95)
- **Lossy compressor 관점에서는 rate가 1.78로 매우 효율적이고, distortion의 절반 이상은 사람 눈에 보이지 않는 부분이다**
- Rate가 조금만 증가해도 Distortion이 빠르게 감소하는데, 이는 대부분의 비트가 실제로는 사람 눈에 보이지 않는 imperceptible distortion을 줄이는 데 쓰인다는 걸 의미한다

**Progressive Generation**
- 이미지 생성 시 **전반적인 형태와 구조(large scale features)가 먼저 나타나고**, 이후 **세부적인 특징(detail)이 나중에 나타난다**
- 이 과정이 **Conceptual Compression**과 유사하다. 무작위 노이즈에서 시작해서 중요한 정보를 먼저 복원하고 덜 중요한 정보를 나중에 복원하는 방식이 압축 해제 과정과 닮아있다

### 4.4 Interpolation

- **Interpolation**: 이미지 A와 이미지 B를 섞어서 중간 이미지를 만드는 작업이다
- 새로운 latent xt 입력에도 artifact 없이 자연스럽게 복원되는데, 이는 모델이 이미지의 구조와 의미적 특징을 학습했다는 걸 의미한다
- 단순 복원이 아니라 **새로운 이미지도 생성 가능하다**는 걸 보여준다 (**Generalization through the reverse process**)

---

## 5. Related Work

- Flows, VAE 등과 비슷해 보이지만, diffusion은 **latent xt가 원본 x0와 독립적이도록 설계**되어 있다. 이 덕분에 모델이 점진적으로 [정보 제거 → 복원] 과정을 안정적으로 수행할 수 있다
- **ε-prediction 방식**은 diffusion을 score matching과 Langevin dynamics sampling에 연결한다. 단순히 노이즈 제거 모델이 아니라, 변분 추론을 통해 학습된 Langevin-like sampler다
- Infusion training, variational walkback, generative stochastic networks 등 비슷하게 Markov chain의 전이 규칙을 학습하는 연구들이 존재하고, diffusion은 이들과 같은 맥락에서 발전한 방식이다
- **Energy-based model**과도 연결되며, rate-distortion 분석이 annealed importance sampling 방식과 유사하다. Progressive decoding 아이디어는 DRAW, autoregressive 모델의 확장 가능성도 보여준다

---

## 6. Conclusion

- **High quality image**: 기존의 생성 모델들과 비교하여 우수한 샘플 품질을 보여준다
- **Connections**: Diffusion Model과 Denoising Score Matching, Annealed Langevin Dynamics 등과의 이론적 연결고리를 확인했다
- **Expansion**: Diffusion 모델은 이미지에 강한 inductive bias를 가지며, 향후 다양한 데이터와 다른 생성 모델과의 결합 가능성이 있다

---

## 7. Future Work — Stable Diffusion

**High-Resolution Image Synthesis with Latent Diffusion Models (CVPR 2022)**

- **Text-to-image 모델**이다
- 픽셀 공간 대신 **latent space에서 Diffusion을 수행**해서 계산 효율이 크게 올라간다
- CLIP을 이용해서 텍스트 프롬프트를 조건으로 이미지를 생성한다
- 기존 방식보다 빠르고 가볍게 고품질 이미지 생성이 가능하다