# GPT-1 논문 요약

## 1. Introduction

**등장 배경**

순차 데이터를 시간 흐름대로 처리하는 RNN은 기울기 소실 문제로 긴 문맥 학습이 어렵다. 이를 해결하기 위해 게이트 구조를 도입한 LSTM이 등장했지만, 순차적 구조로 인해 표현력이 제한된다. 이후 Encoder-Decoder 구조의 Seq2Seq가 번역 등 가변 길이 문제를 해결했고, 최종적으로 RNN 없이 Self-Attention만으로 문맥을 병렬 처리하는 **Transformer**가 등장하여 BERT, GPT 같은 LLM의 기반이 된다.

**Transformer 구조**

Self-Attention 기반 시퀀스 모델로, 각 토큰마다 Query(Q), Key(K), Value(V)를 만들고 Q와 K의 유사도로 중요도를 계산한 뒤 그 가중치로 V를 합산한다. 이를 통해 이 토큰이 다른 토큰들을 얼마나 참고해야 하는가를 계산한다.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**문제 의식**

기존 NLP는 대부분 라벨이 달린 데이터로 지도 학습을 했지만, 라벨 데이터는 비싸고 도메인 확장에 취약하다. 라벨 없는 텍스트에서 언어 지식을 뽑아낼 수 있다면 NLP 전체가 개선된다. word2vec, GloVe 등 사전학습된 word embedding이 이미 성능 향상에 도움이 된다는 선례가 있다.

**기존 방식의 한계**

"뭘 목표로 학습해야 좋은 표현이 생기는지" 합의가 없고, "학습한 표현을 다른 태스크로 옮기는 방법"이 표준화되어 있지 않다. 라벨 없는 데이터로 무언가를 배워도 downstream task에 연결하는 방법이 정리되지 않았다.

**핵심 제안**

목표는 조금만 바꿔도 많은 태스크로 잘 옮겨지는 범용 표현을 학습하는 것이다. 이를 위해 **2-stage 학습(Pre-train → Fine-tune)** 을 도입하며, Decoder-only Transformer를 모델로 선택한다. LM(다음 토큰 예측)에 자연스럽게 맞고 긴 문맥을 잘 다루며 구조 변경 없이도 fine-tuning이 쉽기 때문이다.

---

## 2. Related Work

**학습 방식 분류**

지도학습은 입력과 정답(label)이 쌍으로 주어진 데이터로 학습한다. 비지도 학습은 정답 없이 입력 데이터만으로 학습하고, 준지도 학습은 라벨 데이터와 라벨 없는 데이터를 같이 활용한다.

**NLP에서의 준지도 학습 흐름**

초창기에는 라벨 없는 대규모 텍스트로 단어/구 빈도 등의 통계를 뽑아 지도학습 모델의 입력 특징으로 활용했다. 이후 word2vec, GloVe 등 word embedding으로 발전했지만, 이 접근은 주로 단어 수준 정보(word-level)만 옮긴다는 한계가 있다.

**Unsupervised Pre-training**

지도학습을 바로 학습하는 대신 초기 파라미터를 비지도 목표로 찾는 것으로, 준지도 학습의 특수 케이스다. 사전학습은 성능을 올리는 게 아닌 일반화가 좋아지며, 이미지, 음성, 번역 등 여러 분야로 확장된다. GPT-1과 가장 가까운 기존 연구는 Language modeling으로 pretrain 후 supervised fine-tuning을 수행하는 방식이다.

**LSTM의 한계와 Transformer의 장점**

LSTM은 이론적으로 긴 문맥을 다룰 수 있지만 실제로는 long-range 의존성 학습이 어렵고 순차적 구조로 표현력이 제한된다. Transformer는 Self-attention으로 모든 토큰이 서로 직접 연결되어 긴 거리 의존성 문제를 완화하고 병렬 학습이 가능하다.

**Auxiliary Training Objectives**

메인 태스크를 학습하며 다른 목표를 같이 학습하는 방식으로, $\text{Loss} = L_\text{task} + \lambda L_\text{aux}$ 형태를 사용한다. Rei(2017)는 sequence labeling 태스크를 하면서 auxiliary language modeling loss를 추가하여 성능이 향상됐다. GPT-1에서는 보조 목적을 사용하지만 사전학습 자체가 이미 많은 언어적 정보를 배운다.

---

## 3. Framework

전체 구조는 **Unsupervised Pre-training + Supervised Fine-tuning** 의 2단계로 구성된다.

### 3.1 Unsupervised Pre-training

라벨 없는 토큰 시퀀스 $U = \{u_1, ..., u_n\}$ 에 대해 다음 목적함수를 최대화한다.

$$L_1(U) = \sum_i \log P(u_i | u_{i-k}, ..., u_{i-1}; \Theta)$$

"앞 문맥을 보고 다음 단어 확률의 최대값을 구하라"는 것이 핵심 아이디어다.

**모델 구조**

입력 임베딩은 $h_0 = UW_e + W_p$ 로, token embedding matrix(We)와 position embedding matrix(Wp)를 더해 단어 의미 + 위치 정보를 초기 벡터로 표현한다. 같은 구조의 Transformer 블록을 n번 쌓아 통과시키고, 마지막으로 $P(u) = \text{softmax}(h_n W_e^T)$ 로 모든 단어의 점수를 출력한다.

### 3.2 Supervised Fine-tuning

입력 토큰 시퀀스 $(x^1, ..., x^m, y)$ 를 사전학습된 Transformer에 그대로 넣어 마지막 레이어의 마지막 토큰 hidden state $h_l^m$ 을 얻는다. 이는 문장 전체를 반영한 최종 표현 벡터다. 여기에 새 선형층 $W_y$ 를 붙여 softmax로 클래스 확률을 출력하고, cross-entropy loss $L_2(C)$ 를 최대화한다.

**다운스트림 태스크 입력 변환**

GPT-1 사전학습은 연속된 텍스트(한 줄 토큰 시퀀스)로 진행하지만, 다운스트림 태스크는 입력 구조가 다양하다. 구조별로 다음과 같이 변환하여 입력한다. Textual Entailment(자연어 추론)는 [p $ h] 형태로 premise와 hypothesis를 구분자 토큰 $로 이어붙여 하나의 시퀀스로 만든다. Similarity(문장 유사도)는 순서가 없으므로 두 가지 순서 $[s_1 \$ s_2]$, $[s_2 \$ s_1]$ 를 모두 처리하여 벡터를 더한 뒤 선형층에 입력한다. QA/Commonsense Reasoning은 각 후보마다 $[z \$ q \$ a_k]$ 시퀀스를 생성하여 후보 수만큼 모델을 실행하고 가장 확률 높은 답을 선택한다.

**Auxiliary LM Objective**

Fine-tuning 중에도 LM objective를 조금 유지하면 더 좋은 성능을 낸다. 최종 loss는 다음과 같다.

$$L_3(C) = L_2(C) + \lambda L_1(C)$$

LM loss를 유지하는 이유는 두 가지다. 모델이 분류 데이터에 과적합되는 것을 방지하고 언어 구조를 유지하는 일반화 개선과, 학습 안정화 및 더 빠른 최적화를 위한 수렴 속도 향상이다.

---

## 4. Experiment

### 4.1 사전학습 설정

BooksCorpus(모험, 판타지, 로맨스 등 7,000권 이상의 미출판 도서)를 데이터셋으로 사용한다. 책 데이터라 long-term dependency를 학습할 수 있다. 모델은 Transformer Decoder만 사용하며, ReLU 대신 GELU를 활성화 함수로, sinusoidal 대신 학습 가능한 positional embedding을 사용한다.

### 4.2 Fine-tuning 설정

Pre-training의 하이퍼파라미터를 대부분 유지하고, 3 epoch 동안 fine-tuning을 진행하며 λ=0.5로 설정한다. 특별한 task-specific 설계 없이 단일 language model을 재사용한다.

### 4.3 평가 태스크

4개의 태스크에 대해 실험을 진행한다. Natural Language Inference(자연어 추론), Question Answering & Commonsense Reasoning(질의 응답), Semantic Similarity(문장 유사도), Classification(분류)이며, 각각의 데이터셋은 다음과 같다.

| Task | Datasets |
|---|---|
| Natural Language Inference | SNLI, MultiNLI, QNLI, RTE, SciTail |
| Question Answering | RACE, Story Cloze |
| Sentence Similarity | MSR Paraphrase Corpus, QQP, STS Benchmark |
| Classification | SST-2, CoLA |

**Natural Language Inference**

두 문장 간의 관계를 Contradiction / Neutral / Entailment 3가지로 분류하는 태스크다. 크기가 작은 데이터셋(RTE)보다 큰 경우에서 더 좋은 성능을 보이며, 사용한 데이터셋에서 거의 SOTA를 달성한다.

**Question Answering & Commonsense Reasoning**

RACE dataset과 Story Cloze Test를 사용하여 사용한 모든 데이터셋에서 SOTA를 달성한다. Story Cloze Test와 RACE 모두에서 좋은 성능을 보여 모델이 긴 맥락을 잘 다룬다는 것을 확인한다.

**Semantic Similarity & Classification**

Semantic Similarity의 세 데이터셋 중 두 개에서 SOTA를 달성한다. Classification의 CoLA, GLUE는 SOTA를 달성하며 SST-2도 좋은 성능을 보인다.

---

## 5. Analysis

### 5.1 전이된 레이어 수의 영향

pre-training의 layer를 많이 가져올수록 성능이 올라간다. 각각의 layer가 target task를 해결하는 데 도움이 되며, pretraining이 유용하다는 직접적인 증거가 된다.

### 5.2 Zero-shot Behaviors

LM을 오래 학습할수록 zero-shot 능력이 꾸준히 향상된다. Fine-tuning 없이도 downstream task 성능이 계속 올라가며, LM을 오래 학습할수록 task-specific 학습 없이도 잘 동작한다. Transformer가 LSTM보다 훨씬 빠르게 성능이 오르며, 이는 Transformer 구조가 pre-training에 더 적합하다는 근거가 된다. 감정 분석 / 문법성 판단 / QA / 추론 등 완전히 다른 태스크들이 전부 같은 LM을 오래 학습할수록 함께 올라가므로, **언어 모델 학습이 범용 NLP 능력을 만든다**는 결론을 도출한다.

### 5.3 Ablation Studies

Fine-tuning 단계에서 보조 LM 목적함수는 NLI task와 QQP같은 큰 데이터셋에서는 도움이 되지만 작은 데이터셋에서는 그렇지 않다. Transformer를 2048 unit의 LSTM으로 대체하면 MRPC를 제외하고 평균 점수가 5.6점 감소한다. Pre-training 없이 task를 진행할 경우 14.8% 성능이 감소하여 pre-training의 중요성을 입증한다.

---

## 6. Conclusion

12개 task 중 9개에서 SOTA 또는 SOTA 수준의 성능을 달성한다. 작은 데이터셋(STS-B)부터 큰 데이터셋(SNLI)까지 다양한 규모에서 안정적인 성능을 보인다. Unsupervised pre-training이 downstream task 성능 향상에 크게 기여함을 실험적으로 입증하며, 하나의 pre-trained language model을 다양한 NLP task에 재사용할 수 있음을 보여준다. Task-specific architecture 없이 단일 language model을 다양한 NLP 문제에 적용할 수 있는 가능성을 제시한다.

---

## 7. Future Work

### GPT-2: Scaling

Language model 자체가 multitask learner가 될 수 있는가를 탐구한다. Reddit 링크 기반 웹페이지 45M개인 WebText를 데이터로 사용하며, 파라미터를 117M에서 1.5B로 대폭 확장한다(레이어 12→48, 차원 768→1600). Task-specific fine-tuning 없이도 language model이 다양한 task를 zero-shot setting에서 수행 가능함을 보인다.

### GPT-3: In-Context Learning

모델을 175B 파라미터로 극단적으로 확장하여 "모델을 엄청 크게 만들면 task learning 없이도 문제를 풀 수 있을까?"라는 질문에 답한다. 어떤 task를 수행하도록 모델에게 prompt example만 주면, 그 의미를 스스로 이해해서 fine-tuning이나 구조 변화 없이 작업을 수행한다. Zero-shot, One-shot, Few-shot 설정 모두를 지원한다.

**GPT-1 → GPT-3의 발전은 task-specific model에서 general language model로의 전환**을 의미하며, 이후 ChatGPT, Claude, Gemini, DeepSeek 등 현재 LLM 시대의 기반이 된다.