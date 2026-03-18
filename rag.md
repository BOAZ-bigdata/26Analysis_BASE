# RAG: Retrieval-Augmented Generation 논문 요약

## Introduction

기존 LLM은 세 가지 근본적인 한계를 가진다.

첫째, **메모리를 쉽게 확장하거나 수정할 수 없다.** 모델 파라미터에 지식이 고정되어 있어 새로운 정보를 반영하려면 재학습이 필요하다.

둘째, **모든 지식을 파라미터에 저장하기 어렵다.** 세상의 모든 지식을 유한한 파라미터 안에 압축해서 담는 것은 현실적으로 불가능하다.

셋째, **할루시네이션을 생성할 수 있다.** 학습 데이터에 없는 정보를 묻는 경우 그럴듯하지만 틀린 답변을 생성하는 문제가 있다.

이를 해결하기 위해 RAG를 제안한다. **외부 문서 데이터베이스에서 관련 정보를 검색한 뒤 LLM과 결합하여 답변을 생성**하는 방식이다.

---

## Related Works

RAG는 기존 연구 흐름들을 통합하는 방향으로 설계된다.

**Single-task Retrieval**은 retrieval을 활용해 특정 NLP task의 성능을 높이는 방식이다. Open-domain QA, Fact Checking, Dialogue 등이 대표적이다.

**General-purpose NLP Architectures**는 하나의 pre-trained language model로 다양한 task를 수행하는 방향이다. BERT, GPT-2, BART, T5가 해당된다.

**Learned Retrieval**은 neural retriever를 학습시켜 task에 필요한 문서를 검색하는 방식이다.

**Memory-based Architectures**는 외부 메모리를 활용하여 지식을 저장하고 활용한다.

**Retrieve-and-Edit**은 유사한 example을 검색한 후 수정하여 결과를 생성하는 방식이다.

RAG는 이 모든 흐름을 통합하여 **Retrieval + Generative LLM을 결합한 하나의 architecture**로 여러 knowledge-intensive NLP task를 해결한다.

---

## RAG의 2가지 모듈

RAG는 검색기(Retriever)와 생성기(Generator) 두 모듈로 구성된다.

**검색기 pη**: 쿼리 x가 주어졌을 때 텍스트 구절에 대한 top-k 분포를 반환하는 파라미터 η를 가진다. 즉, 입력 질문과 관련된 문서를 찾아서 상위 k개를 골라 반환한다.

**생성기 pθ**: 이전 i-1개의 토큰 y1:i-1, 원본 입력 x, 그리고 검색된 구절 z의 컨텍스트를 기반으로 현재 토큰을 생성하는 파라미터 θ를 가진다. 즉, 검색된 문서를 참고하여 자연스러운 답변 텍스트를 생성한다.

---

## Hybrid Memory 구조

RAG는 두 종류의 메모리를 결합하는 **Hybrid Knowledge System**을 사용한다.

**모수적 메모리(Parametric Memory)**: 일반적인 사전 학습된 seq2seq 모델처럼 이미 학습된 파라미터 안에 지식을 저장하는 방식이다. RAG에서는 BART 모델이 이 역할을 담당한다.

**비모수적 메모리(Non-parametric Memory)**: 검색 가능한 외부 데이터베이스를 벡터 데이터베이스로 구성하여 필요한 지식을 검색하고 활용하는 방식이다. RAG에서는 Wikipedia 대규모 데이터베이스가 이 역할을 담당한다.

두 메모리를 결합함으로써 모델이 파라미터에 없는 지식도 외부에서 가져와 활용할 수 있게 된다.

---

## Retriever: DPR

검색기로는 **DPR(Dense Passage Retrieval)**을 사용한다.

**Bi-encoder 구조**로 동작한다. 쿼리 문장과 문서 문장을 각각 별도의 인코더에 입력하여 독립적인 임베딩 벡터를 생성하고, 두 벡터 간 내적으로 유사도를 계산한다.

**초기화 및 인덱스 구축** 방식은 다음과 같다. 사전 학습된 bi-encoder로 검색기를 초기화하고, 전체 문서를 임베딩하여 문서 인덱스를 구축한다. 이 문서 인덱스가 비모수적 메모리에 해당한다.

**학습 방식**으로는 TriviaQA, Natural Questions의 답변을 포함하는 문서를 positive sample로 학습시키며, 검색기가 관련 문서를 상위 K개 안에 포함시키도록 훈련한다.

---

## Generator: BART

생성기로는 **BART-large**를 사용한다.

4억 파라미터의 사전 학습된 seq2seq 트랜스포머 모델이다. 입력은 쿼리 x와 검색된 문서 z를 단순 연결하여 구성한다. 다양한 노이즈 함수를 활용한 디노이징으로 사전 학습되어 있으며, 인코더-디코더 구조 덕분에 검색된 문서를 조건으로 자연스러운 텍스트 생성이 가능하다. 동일 크기의 T5 모델 대비 다양한 생성 태스크에서 우수한 성능을 보인다.

---

## Training

학습 대상은 **쿼리 인코더 BERTq와 BART 생성기를 end-to-end로 fine-tuning**하는 것이다.

문서 인코더(BERTd)와 인덱스는 고정한다. 학습 중 업데이트 시 문서 인덱스를 매번 재구축해야 하는데, 비용 대비 성능 향상이 미미하여 고정으로 결정한다. 즉, 쿼리 인코더와 생성기만 학습하고 문서 인덱스는 그대로 유지한다.

---

## Decoding: RAG-Sequence vs RAG-Token

RAG 모델이 답변을 생성하는 방식은 두 가지가 있다.

**RAG-Sequence Model**은 전체 시퀀스 단위로 생성하는 방식이다. 검색된 문서 하나를 시퀀스 전체에 동일하게 사용한다. 과정은 상위 K개 문서를 검색하고, 각 문서로 전체 출력 시퀀스 확률을 계산한 뒤, 확률을 marginalize하여 최종 답변을 생성한다.

$$P_{RAG-Sequence}(y|x) = \sum_{z \in top\text{-}k(p(\cdot|x))} p_\eta(z|x) \prod_i p_\theta(y_i|x, z, y_{1:i-1})$$

**RAG-Token Model**은 토큰 단위의 유연한 생성 방식이다. 토큰을 생성할 때마다 다른 문서를 참조할 수 있다. 과정은 상위 K개 문서를 검색하고, 각 토큰마다 문서별 분포를 생성한 뒤, 분포를 marginalize하여 다음 토큰을 생성하는 것을 반복한다.

$$p_{RAG-Token}(y|x) \approx \prod_i \sum_{z \in top\text{-}k(p(\cdot|x))} p_\eta(z|x) p_\theta(y_i|x, z, y_{1:i-1})$$

두 방식의 차이는 **문서를 시퀀스 전체에 고정하느냐, 토큰마다 다르게 참조하느냐**에 있다. RAG-Token이 더 유연하지만 계산 비용도 더 크다.

---

## Experiment Setting

실험에서는 다음 세 단계를 거쳐 retrieval을 수행한다.

**Document**: Wikipedia dump(2018년 12월 기준)를 외부 지식 소스로 사용한다.

**Embedding**: Facebook에서 만든 vector 라이브러리 **FAISS**로 문서를 임베딩하여 벡터 데이터베이스를 구축한다.

**Vector Search**: **MIPS(Maximum Inner Product Search)**로 쿼리와 가장 유사한 문서를 빠르게 검색한다.

---

## Experiments & Results

네 가지 태스크에서 실험을 진행한다.

**Open-domain QA**는 광범위한 주제에 걸친 질문에 대답하는 태스크이다. Retrieval과 Generation을 결합한 RAG의 강점이 가장 잘 드러나는 영역이다.

**Abstractive QA**는 문서에서 답을 직접 추출하는 것이 아니라 새롭게 생성하는 태스크이다. Gold passage 없이도 답변을 새롭게 생성할 수 있는지를 평가한다.

**Jeopardy Question Generation**은 주어진 정답을 기반으로 자연스러운 질문 문장을 만들 수 있는지를 평가하는 태스크이다. 사람이 자연스럽게 인식할 수 있는 수준의 질문을 생성하는지도 함께 평가한다.

**Fact Verification**은 주어진 주장이 참인지 거짓인지를 판단하는 태스크이다. 주장에 대한 evidence를 명확히 찾아낼 수 있는지를 평가한다.

네 가지 태스크 전반에 걸쳐 RAG는 기존 retrieval-only 방식이나 generation-only 방식보다 우수한 성능을 보이며, **하나의 architecture로 다양한 knowledge-intensive NLP task를 효과적으로 처리할 수 있음**을 보여준다.

---

### 1. Open-domain QA

평가 지표는 Exact Match(EM) Score를 사용한다. RAG는 NQ, WQ, CT에서 SOTA를 달성한다. DPR보다 구조가 단순함에도 동등하거나 더 높은 성능을 낸다. TQA에서만 동일 dataset split 기준 DPR에 비해 낮다. Closed-Book 방식 대비 외부 검색 없이도 QA가 가능함을 보인다.

---

### 2. Abstractive QA

문서에서 정답을 추출하는 게 아니라 새롭게 생성하는 능력을 평가한다. MSMARCO NLG dataset을 사용하며, RAG는 gold passage 없이 스스로 retrieval하여 답변을 생성한다. BART 단독 대비 RAG-Seq.가 R-L 기준 +2.6 향상되며, BART가 hallucination을 생성하는 반면 RAG는 사실에 가까운 자연스러운 답변을 생성한다.

---

### 3. Jeopardy Question Generation

정답이 주어지면 그에 맞는 질문을 생성하는 task이다. RAG-Token이 RAG-Sequence보다 높은 성능을 보인다. 여러 문서를 조합해야 하는 task 특성상 토큰마다 다른 문서를 참조할 수 있는 RAG-Token이 유리하기 때문이다. 사람 평가에서 Factuality 기준 RAG 42.7% vs BART 7.1%, Specificity 기준 RAG 37.4% vs BART 16.8%로 RAG가 압도적으로 우수하다.

---

### 4. Fact Verification

주어진 주장이 Wikipedia에 의해 참/반박/정보 부족인지 판단하는 FEVER 태스크를 사용한다. supervision 없이 RAG가 스스로 evidence를 찾도록 설정한다. SotA(92.2%) 대비 RAG-Seq.는 89.5%로 약간 낮지만, Top1 evidence retrieval 71%, Top10 기준 90%로 높은 검색 성능을 보인다. 복잡한 파이프라인 없이도 스스로 evidence를 탐색한다는 점에서 의의가 있다.

---

## Retrieval Ablations

세 가지 retrieval 방식을 비교한다.

**학습하는 Retrieval(RAG)**이 open-domain QA에서 성능을 크게 향상시킨다. Frozen Retrieval은 RAG 대비 NQ에서 약 2~3점 낮고, BM25는 keyword matching 방식으로 FEVER에서 강점을 보이지만 전반적으로 낮은 성능을 보인다.

결론적으로 **end-to-end로 학습된 retrieval이 가장 효과적**이며, task에 따라 BM25 같은 keyword 기반 방식이 유리한 경우도 있다.

---

## Test: k Retrieved Documents

검색하는 문서 수 k에 따른 성능 변화를 분석한다.

**Open-domain QA 성능**: RAG-Sequence는 k가 커질수록 성능이 꾸준히 향상된다. RAG-Token은 k=10에서 최대 성능을 보이고 이후 점차 감소한다.

**Recall**: k가 커질수록 Recall이 향상된다. 문서 개수가 많아질수록 정답 문서를 포함할 확률이 증가하기 때문이다.

**생성 문장의 평가 점수**: Rouge-L(정답 문장과 의미적으로 겹치는지)과 BLEU-1(같은 단어가 중복 사용되는지) 모두 k가 증가해도 크게 변하지 않고 안정적으로 유지된다.

---

## Conclusion

**의의**: RAG의 텍스트 생성은 모델 파라미터에 저장된 지식(parametric memory)과 외부 문서 검색(non-parametric memory)을 결합하는 방식이다. 단일 architecture로 다양한 knowledge-intensive NLP task를 효과적으로 처리할 수 있음을 보인다.

**향후 연구 1**: parametric memory와 non-parametric memory가 어떻게 상호작용하는지, 두 가지를 어떻게 더 효율적으로 연결할 수 있는지를 연구할 필요가 있다.

**향후 연구 2**: retriever와 generator를 BART의 denoising function이나 다른 목적 함수로 처음부터 pretraining할 수 있는지 탐색할 필요가 있다.

---

## Follow-up Work

RAG 논문 이후로 다양한 방향의 후속 연구가 진행된다.

**다단계 RAG**는 쿼리를 반복 재구성해 이전 검색 결과를 바탕으로 점진적으로 답변을 생성한다.

**대화형 RAG**는 과거 대화 맥락과 새로 검색된 정보를 결합해 일관된 멀티턴 응답을 생성한다.

**하이브리드 및 그래프 기반 RAG**는 어휘 검색과 벡터 검색을 결합하고, 지식 그래프로 데이터 간 관계를 파악한다.

**에이전트 RAG**는 가설 수립 → 증거 검색 → 결과 평가를 반복하며 자율적으로 답변을 도출한다.