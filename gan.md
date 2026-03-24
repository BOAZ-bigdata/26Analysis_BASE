## 1. Introduction

**Discriminative Models vs Generative Models**

- **Discriminative Model**: 입력 데이터를 클래스 라벨로 매핑하는 모델 → p(y|x) 학습
  - Backpropagation, Dropout, piecewise linear units 등 알고리즘이 잘 발전되어 있고 안정적이며 효율적인 학습이 가능하다
- **Generative Model**: 데이터 분포 자체를 학습해야 하는 모델 → p(x) 학습
  - 확률 계산이 intractable하고 복잡한 방법이 필요하다는 문제가 있어서 학습 비용이 크다

**GAN의 구조**
- 노이즈 입력 → **Generator(샘플 생성, MLP)** → **Discriminator(판별, MLP)** → 다시 Generator로 피드백
- 실제 데이터 분포도 Discriminator에 함께 입력된다

**Advantages of GAN**
- Backpropagation + Dropout으로 학습 가능하다
- Forward propagation만으로 샘플링이 가능하다
- Approximate inference나 Markov chain이 필요 없다

---

## 2. Related Work

**Discriminative Models의 목표와 한계**
- 목표: 실제 데이터 분포를 학습하고 실제 데이터와 유사한 샘플을 생성하는 것
- 한계: 학습에 어려운 확률 계산이 필요하다

**Boltzmann-based Models: RBM, DBM, DBN**
- **RBM**: visible unit과 hidden unit으로 구성된 undirected graphical model
- **DBM**: RBM을 여러 층 쌓은 구조
- **DBN**: undirected layer 1개 + directed layer 여러 개로 이루어진 hybrid model
- **학습 방식**: Energy-based model로 energy function을 통해 확률을 정의하고 sampling이 필요하다
- **한계**: partition function이 intractable하고, MCMC sampling이 필요하며 학습 비용이 크다

**다른 접근법들의 장단점**

| | Score Matching / NCE | Generative Stochastic Networks (GSN) |
|---|---|---|
| 장점 | direct MLE 회피, discriminative objective 사용 가능 | backpropagation으로 학습 가능, 매 스텝마다 확률 계산 불필요 |
| 단점 | tractable density 여전히 필요, 복잡한 모델에 적용 어려움 | parameterized Markov chain 의존, 샘플링이 느리고 반복적 |

**Previous Approaches vs GAN 비교**

| Previous | GAN |
|---|---|
| likelihood 계산 또는 tractable density 필요 | explicit likelihood 불필요 |
| MCMC 또는 Markov chain 의존 | Markov chain 불필요 |
| 샘플링이 느리고 반복적 | forward pass로 바로 샘플링 가능 |
| 학습이 계산적으로 복잡 | 표준 backpropagation으로 학습 가능 |
| 모델 설계가 제한적 | 더 유연한 implicit generative model |

---

## 3. Adversarial Nets

$$\min_G \max_D V(D,G) = \mathbb{E}_{\boldsymbol{x} \sim p_{\text{data}}(\boldsymbol{x})}[\log D(\boldsymbol{x})] + \mathbb{E}_{\boldsymbol{z} \sim p_z(\boldsymbol{z})}[\log(1 - D(G(\boldsymbol{z})))]$$

- **p_data(x)**: 실제 데이터 샘플 m개를 생성하는 PDF
- **p_g(z)**: 노이즈 샘플 m개를 생성하는 PDF
- **Discriminator D**: 입력이 얼마나 실제 같은지 (0,1) 사이의 값으로 출력. D(x) = 1/2를 threshold로 사용
- **Generator G**: 랜덤 노이즈 입력으로부터 가짜 데이터를 생성하는 함수 (latent 노이즈 벡터 → 데이터 공간으로 매핑)

**목적 함수 해석**
- 첫 번째 항 E[log D(x)]: 실제 데이터가 들어왔을 때 실제 데이터일 확률 → **D는 최대화** (~1에 가깝게)
- 두 번째 항 E[log(1-D(G(z)))]: 가짜 샘플이 실제인지 아닌지를 판별하는 값 → **D는 최대화, G는 최소화**

**학습 과정 시각화**
- **(a)**: 생성 모델 분포가 실제 분포와 유사해지기 시작하고, D가 부분적으로 정확한 분류기를 형성한다
- **(b)**: D가 훈련 데이터와 생성 데이터를 구별하도록 학습된다
- **(c)**: D의 기울기에 따라 G(z)가 실제 데이터로 분류될 가능성이 높은 영역으로 이동하도록 학습된다
- **(d)**: G와 D가 충분한 용량을 가지고 있다면, 학습이 진행될수록 생성 모델의 분포는 실제 데이터 분포에 수렴한다

**Problem 1 — Training Imbalance**
- D를 완전히 최적화하려 하면 계산 비용이 매우 커지고, 유한한 데이터에서는 overfitting이 발생할 수 있다
- 그러면 Generator는 매우 약한 gradient만 받게 되어 학습이 불안정해진다
- **해결**: D를 k번 업데이트할 때 G를 1번만 업데이트한다. G가 천천히 학습되는 동안 D를 거의 최적 상태에 가깝게 유지하는 방식이다

**Problem 2 — Gradient Saturation**
- log(1-D(G(z))) 항이 쉽게 saturation 상태가 되면서 vanishing gradient 문제가 발생한다. 특히 학습 초기에 심각하다
- **해결**: log(1-D(G(z)))를 최소화하는 대신 log D(G(z))를 최대화하도록 학습한다

---

## 4. Theoretical Results

### GAN's Objective Function

**문제 1 — Expectation 계산의 불안정성**
- Expectation 계산은 PDF 기반인데 실제 분포를 알기 어렵다
- **해결**: Monte-Carlo Method 사용
$$\mathbb{E}[f(x)] \approx \frac{1}{m} \sum_{i=1}^{m} f(x^{(i)})$$

**문제 2 — MinMax 최적화의 불안정성**
- MinMax를 한 번에 찾는 문제는 파라미터가 동시에 update되기 때문에 안장점(Saddle Point)에 도달하기 어렵다
- **해결**: 최적화를 두 단계로 나눈다
  - Step 1: Discriminator를 먼저 최적화 (stochastic gradient ascent)
  - Step 2: Generator를 이후에 최적화 (stochastic gradient descent)

### 4.1 Global Optimality of p_g = p_data

**최적 Discriminator 도출**
$$D_G^*(\boldsymbol{x}) = \frac{p_{\text{data}}(\boldsymbol{x})}{p_{\text{data}}(\boldsymbol{x}) + p_g(\boldsymbol{x})}$$

- f(y) = a log(y) + b log(1-y)를 미분해서 0으로 놓으면 y = a/(a+b)가 나오고, 이를 적용하면 위 식이 도출된다

**C(G) 변환 → JSD로 표현**
$$C(G) = -\log(4) + 2 \cdot JSD(p_{\text{data}} \| p_g)$$

- **KL Divergence**: D_KL(P||Q) ≠ D_KL(Q||P) → 비대칭
- **Jensen-Shannon Divergence (JSD)**: KL Divergence를 대칭적으로 만든 버전으로 M = (P+Q)/2를 중간 분포로 사용한다
- JSD는 항상 0 이상이고, p_g = p_data일 때 0이 된다 → 이때 C(G)의 최솟값은 -log(4)
- 즉 **Generator가 실제 데이터 분포를 완벽하게 학습하면 global optimum에 도달한다**

### 4.2 Convergence of Algorithm 1

- G와 D가 충분한 capacity를 가지고, 매 step마다 D가 주어진 G에 대해 최적점에 도달할 수 있다면, p_g는 p_data에 수렴한다
- 실제로는 G를 p_g 자체가 아닌 θ_g로 parameterize하기 때문에 parameter space에 multiple critical point가 생긴다
- 그러나 multilayer perceptron의 실용적 성능이 우수하기 때문에 이론적 보장이 없더라도 사용하기에 합리적인 모델이다

---

## 5. Experiments

**실험 세팅**
- 데이터셋: MNIST, Toronto Face Database (TFD), CIFAR-10
- Generator: ReLU + Sigmoid
- Discriminator: Maxout, Dropout, Sigmoid
- Noise: input layer에만 적용

**평가 지표 — Gaussian Parzen Window**
- 테스트 데이터들이 공간에 얼마나 높은 확률로 존재하는지 측정하여 분포에 근사하는 방법이다
- 여전히 PDF를 직접 알 수 없고, 분산이 크고 고차원 이미지에 적합하지 않다는 한계가 있다

**결과**
- MNIST에서 Adversarial nets가 225 ± 2로 가장 높은 성능을 보인다
- TFD에서는 Stacked CAE가 2110 ± 50으로 가장 높고, Adversarial nets는 2057 ± 26이다


### Mode Collapse

- 생성자가 다양한 형태를 학습하는 것을 포기하고, **가장 자신 있는 극소수의 정답만을 반복해서 만들어내는 현상**이다
- G의 목적이 오직 Discriminator를 속이는 것뿐이기 때문에 발생한다
- 다양성보다 안전한 답 하나에 집중하게 되어 이미지의 다양성이 사라진다

### Advanced GANs

#### Wasserstein GAN (WGAN)
Vanishing Gradient와 Mode Collapse 문제를 해결하기 위해 KL Divergence 대신 **Wasserstein Distance**(= Earth-Mover Distance)를 사용한다

- P_r에서 P_g로 질량을 옮기는 데 드는 **최소 비용**이다
- 가능한 이동 방법(γ) 중 비용이 가장 작은 값을 선택한다 → 예시에서 18, 22, 24 중 **18이 Wasserstein Distance**
- KL Divergence와 달리 두 분포가 겹치지 않아도 의미 있는 거리를 계산할 수 있어서 학습이 더 안정적이다

#### Deep Convolution GAN (DCGAN)
GAN의 학습 불안정성을 극복하기 위해 **Convolution layer로 feature map을 추출해서 GAN에 적용**한 모델이다

- Discriminator: Leaky ReLU + Batch Normalization + Sigmoid
- Generator: ReLU + Batch Normalization + tanh, 노이즈 벡터에서 점점 upsampling하는 구조
- latent space에서 벡터 연산이 가능하다 → (안경 쓴 남자) - (안경 없는 남자) + (안경 없는 여자) = **안경 쓴 여자**
- 이미지를 외우는 게 아니라 **개념 자체를 학습**했음을 보여준다