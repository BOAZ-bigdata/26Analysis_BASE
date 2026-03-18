# LoRA: Low-Rank Adaptation of Large Language Models 논문 요약

## Introduction

대규모 사전학습 모델(pretrained model)을 특정 downstream task에 활용하려면 fine-tuning이 필요하다. 기존 full fine-tuning 방식은 모델의 **모든 파라미터를 업데이트**해야 하므로 저장 비용과 계산 비용이 매우 크다.

이를 해결하기 위한 기존 방법으로 두 가지가 있다.

**Adapter layer 추가**는 기존 레이어 사이에 작은 학습 가능한 레이어를 삽입하는 방식이다. 파라미터 수는 줄일 수 있지만 GPU 병렬 처리가 불가능해 inference latency가 증가하는 문제가 있다.

**Prefix tuning**은 입력 앞뒤에 학습 가능한 토큰을 추가하는 방식이다. 하지만 모든 Transformer layer에 적용하면 파라미터 수가 L × d_model × (l_p + l_i)로 늘어나 너무 많은 파라미터를 학습하게 된다.

두 방법 모두 **효율 vs 성능 trade-off**가 존재한다.

이에 LoRA는 새로운 가설을 제시한다. **모델 적응 과정에서 weight 변화가 low-rank 구조를 가진다**는 것이다. 이 가설을 기반으로 전체 파라미터를 업데이트하지 않고도 효율적인 fine-tuning이 가능하다고 주장한다.

---

## Problem Statement

기존 Full Fine-tuning 방식은 사전학습된 언어 모델 PΦ(y|x)를 downstream task에 맞게 **모든 파라미터 Φ를 업데이트**하는 방식이다. GPT-3 같은 Transformer 기반 모델이 대표적인 예시이며, Text Summarization, MRC, NL2SQL 등 다양한 task에 적용된다.

모델 변화는 Φ₀ → Φ₀ + ΔΦ로 표현된다.

문제는 **task마다 새로운 파라미터 ΔΦ가 필요**하고, 그 크기가 |ΔΦ| = |Φ₀|이라는 점이다. 즉, GPT-3 기준으로 task마다 175B짜리 모델을 통째로 따로 저장해야 한다. 이는 현실적으로 매우 비효율적이다.

---

## 기존 해결책은 충분한가?

Adapter layer를 추가하는 방식은 batch size가 줄어들수록 latency가 급격히 증가한다. 실제로 Fine-Tune/LoRA 대비 Adapter^H는 batch size=1일 때 **+30.3%의 latency**가 발생한다. 이는 GPU 병렬 처리가 불가능하기 때문이다. 배치가 충분히 크면 latency 차이가 줄어들지만, 실제 서비스 환경에서는 작은 배치를 사용하는 경우가 많아 문제가 된다.

Prompt/Prefix tuning은 입력 시퀀스의 일부를 학습 가능한 파라미터로 대체하는 방식이다. 하지만 학습 가능한 토큰 수가 늘어날수록 성능이 오히려 불안정해지는 경향이 있다. 또한 시퀀스 길이를 차지하기 때문에 실제 입력 처리에 사용할 수 있는 context 길이가 줄어드는 부작용도 있다.

결국 두 방법 모두 **효율성과 성능을 동시에 만족시키지 못한다**.

---

## LoRA: Low-Rank Weight Update

LoRA의 핵심 아이디어는 weight update ΔW를 **두 개의 작은 행렬 A, B의 곱으로 분해**하는 것이다.

기존 fine-tuning에서 weight는 다음과 같이 업데이트된다.

$$W = W_0 + \Delta W$$

LoRA는 이를 다음과 같이 표현한다.

$$W = W_0 + BA$$

- W₀ ∈ R^(d×k) : 사전학습된 weight, **freeze**
- B ∈ R^(d×r), A ∈ R^(r×k) : **학습 대상**
- r ≪ d, k : rank가 매우 작음

따라서 forward pass는 다음과 같이 표현된다.

$$h = W_0x + \Delta Wx = W_0x + BAx$$

초기화 방식도 중요하다. A는 랜덤 가우시안(N(0, σ²))으로 초기화하고, B는 0으로 초기화한다. 학습 초기에 ΔW = BA = 0이 되어 사전학습 모델의 출력을 그대로 유지하며 안정적으로 학습을 시작할 수 있다.

### LoRA 적용 위치

Transformer의 Self-Attention에 있는 네 가지 weight matrix에 적용할 수 있다.

- Wq (Query)
- Wk (Key)
- Wv (Value)
- Wo (Output)

실험 설정에서는 **Wq와 Wv에만 LoRA를 적용**하고 MLP layer는 freeze한다.

### Practical Benefits of LoRA

| Feature | Description | Result |
|---------|-------------|--------|
| Memory Usage | 사전학습 weight freeze | 1.2TB → 350GB |
| Checkpoint Size | A, B 파라미터만 저장 | 350GB → 35MB |
| Training Speed | gradient 계산 감소 | ~25% 빠름 |
| Inference Latency | 추론 시 W₀와 BA 합산 | 추가 latency 없음 |

GPT-3 175B 기준으로 전체 파라미터 중 **18M개(0.01%)만 학습**하며 용량은 35MB에 불과하다.

---

## Empirical Experiments

### 실험 설계

세 가지 모델을 사용해 평가한다.

**RoBERTa** (NLU 평가): BERT 구조 기반, Base(125M)와 Large(355M) 두 가지 크기로 실험한다.

**DeBERTa** (NLU 평가): BERT 변형 모델로 XXL(1.5B) 크기를 사용한다.

**GPT-2 / GPT-3** (NLG 평가): GPT-2는 medium(355M)과 Large(774M), GPT-3는 175B 파라미터 모델을 사용한다.

평가 기준으로는 **GLUE benchmark**를 활용하며 감정 분석, 문장 유사도, 자연어 추론 등을 평가한다.

Fine-tuning 방식으로는 pretrained 모델에서 특정 task에 맞게 일부 layer만 학습하는 방식(FTTop2 등)도 비교한다.

비교 대상으로는 Fine-Tune, PrefixEmbed, PrefixLayer, Adapter(H), LoRA를 사용한다.

### 실험 결과

WikiSQL과 MultiNLI-matched 태스크에서 LoRA는 **훨씬 적은 파라미터로 Fine-tune과 동등하거나 더 나은 성능**을 달성한다.

특히 GPT-3 기준으로 LoRA는 **4.7M 파라미터**만으로 175B full fine-tuning에 필적하는 성능을 낸다.

---

## Related Works

LoRA가 기반으로 하는 관련 연구 흐름은 네 가지로 정리된다.

**Transformer Model**: 모델이 커질수록 성능이 좋아진다는 것을 BERT, GPT-2, GPT-3를 통해 확인한다.

**Adaptation**: 기존 layer 사이에 adapter layer를 삽입하는 방식으로 파라미터 효율성을 높이려는 시도이다.

**Prompt Engineering / Fine-Tuning**: 모델의 입력 또는 일부를 학습시켜 task 성능을 개선하는 방향이다.

**Low-Rank Structures**: 많은 머신러닝 문제가 내재적으로 low-rank 구조를 가진다는 것이 LoRA의 핵심 이론적 배경이다.

---

## Understanding the Low-Rank Updates

### Q/K/V weight matrix 업데이트 분석

GPT-3 175B에서 18M 파라미터(0.01%)만 사용해 실험한 결과, **Wq와 Wv에 함께 LoRA를 적용하고 rank=4**로 설정했을 때 WikiSQL 73.7%, MultiNLI 91.3%의 성능을 달성한다. 이는 모든 weight에 적용하거나 더 높은 rank를 사용하는 것과 거의 차이가 없다.

### 실제로 low-rank 구조를 가지는가?

**작은 r에서도 좋은 성능이 나온다.** r=1, r=2일 때도 r=64와 성능 차이가 크지 않다. Wq와 Wv를 함께 적용할 때는 오히려 더 높은 성능을 보이기도 한다.

단, 작은 r 값이 모든 task에서 통하는 것은 아니다. pre-training 언어와 fine-tuning 언어가 다른 상황처럼 task 간 차이가 큰 경우에는 더 높은 rank가 필요할 수 있다.

**rank 8과 rank 64의 학습 방향 비교**: subspace similarity φ를 측정하면, i=1(rank 8 모델의 상위 1번째 방향)일 때 φ=0.5 이상으로 rank 64 모델의 상위 subspace와 겹친다. 즉, **rank 8 모델과 rank 64 모델의 학습 방향은 거의 비슷하다.** 작은 rank의 요약본도 큰 rank의 본문 내용을 충분히 담고 있다는 의미이다.

**랜덤 seed가 달라도 같은 방향으로 학습하는가?** ΔWq는 φ값이 높고, ΔWv는 φ값이 비교적 낮다. 이는 Wq(어디를 볼 것인가)는 랜덤 seed와 무관하게 비슷한 방향으로 수렴하지만, Wv(어떻게 업데이트할 것인가)는 상대적으로 더 다양한 방향이 존재한다는 것을 의미한다. 결론적으로 **모델이 업데이트해야 할 중요한 방향은 미리 정해져 있다.**

### 기존 weight와의 관계 (ΔW-W)

Frobenius norm 분석 결과 세 가지 사실을 확인한다.

**LoRA 업데이트는 기존 weight과 연관성이 더 크다.** ΔWq와 Wq 사이의 projection 값(‖U⊤WqV⊤‖_F)이 랜덤 행렬보다 훨씬 크다.

**기존 모델 방향과 다른 특징을 강화시킨다.** ΔWq는 Wq 자체보다 훨씬 작은 norm을 가지며, Wq가 이미 강조하고 있지 않은 방향을 보완적으로 학습한다.

**그 다른 특징을 엄청나게 크게 강화시킨다.** ΔWq의 방향에서 Wq가 가진 성분(projection)이 랜덤 대비 매우 크게 나타난다. 즉 LoRA는 사전학습 모델이 놓치고 있는 task-specific한 방향을 집중적으로 증폭시키는 방식으로 동작한다.

---

## Conclusion & Future Work

### Conclusion

LoRA는 low-rank 기반 parameter-efficient adaptation으로 **추가 latency 없이** 대형 모델을 특정 task에 맞게 적응시킬 수 있다. shared parameter 구조 덕분에 빠른 task switching도 가능하다. 전체 파라미터의 극히 일부만 학습하면서도 full fine-tuning에 필적하는 성능을 달성한다.

### Future Work

앞으로의 연구 방향으로는 다음 네 가지를 제시한다.

- 다른 PEFT 방법과의 결합
- adaptation mechanism에 대한 더 깊은 이해
- LoRA를 어느 layer에 적용할지 선택 문제 자동화
- model weight의 low-rank 구조에 대한 이론적 분석