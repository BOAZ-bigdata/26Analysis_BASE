# DeepSeek-R1 논문 정리

## 1. Introduction

**문제 인식**

Reasoning은 인간 지능의 핵심으로, 기존 LLM은 모델 규모가 커질수록 추론 능력이 자연스럽게 등장하는 Emergent Ability를 보인다. 그러나 단순히 모델을 크게 만드는 것은 막대한 계산 비용을 요구한다.

기존 추론 방법(Chain-of-Thought Prompting, Few-Shot Examples)은 복잡한 문제에서 성능을 향상시키지만, 다음과 같은 근본적 한계를 가진다.
- 인간이 작성한 추론 경로에 의존 → 확장성 문제
- 인간의 인지 편향이 모델에 그대로 주입된다
- 인간을 넘어서는 사고 탐색이 불가능하다

**핵심 아이디어**

모델이 RL을 통해 reasoning 능력을 스스로 향상시키며, 인간이 reasoning 과정을 직접 가르치지 않는다. DeepSeek-V3-Base에 GRPO 기반 RL을 적용하여, 정답이면 Reward, 오답이면 No Reward를 부여하는 방식으로 추론 과정에 제약을 두지 않는다.

---

## 2. DeepSeek-R1-Zero

### 2.1 GRPO (Group Relative Policy Optimization)

RL 학습 비용을 절감하기 위해, 동일한 크기의 value 모델을 생략하고 그룹 점수에서 베이스라인을 추정하는 방법을 사용한다.

**핵심 아이디어:** 하나의 질문에 대해 여러 개의 답을 생성하고 서로 비교한다.

**훈련 과정**
1. 질문 q를 입력한다
2. old policy πθold가 여러 개의 output을 생성한다 {o1, o2, ..., oG}
3. 각 output에 대해 reward를 계산한다 r1, r2, ..., rG
4. group 기반 advantage를 계산한다

$$A_i = \frac{r_i - \text{mean}(\{r_1, r_2, \cdots, r_G\})}{\text{std}(\{r_1, r_2, \cdots, r_G\})}$$

→ group 평균 대비 상대적인 성능을 평가한다. 프롬프트 템플릿은 `<think>` 태그로 reasoning을 먼저 생성하도록 구조를 정의한다.

### 2.2 보상 설계 (Reward Design)

신경망 기반 reward model은 보상 해킹(Reward Hacking)에 취약하므로, **두 가지 규칙 기반 보상 시스템**을 채택한다.

**정확도 보상 (Accuracy Rewards)**
- 수학 문제: 지정된 형식(박스 안 최종 정답)으로 응답하며, 규칙 기반으로 채점한다
- LeetCode 문제: 컴파일러로 사전 정의된 테스트케이스를 기반으로 평가한다

**형식 보상 (Format Rewards)**
- `<think>` 와 `</think>` 태그 사이에 사고 과정을 포함하도록 강제한다
- 체계적인 사고 과정을 명시하여 논리적 추론 구조를 형성한다

$$\text{Reward}_{\text{rule}} = \text{Reward}_{\text{acc}} + \text{Reward}_{\text{format}}$$

### 2.3 추론 능력 향상 결과

**성능 향상**
- RL 훈련 동안 AIME accuracy가 **15.6% → 77.9%** 로 크게 증가한다
- Self-consistency decoding 적용 시 **86.7%** 로 Human 평균 성능을 초과한다
- 훈련이 진행될수록 response length가 증가하며 Long Chain-of-Thought가 형성된다

**Aha Moment**

강화학습 학습 과정에서 모델이 자신의 추론 과정을 스스로 다시 검토하는 행동이 나타난다. 특정 시점에서 *"Wait, wait… 다시 생각해보자."* 와 같은 표현을 사용하며 이전 추론 과정을 reflection하기 시작한다. 이는 RL만으로 reasoning 능력이 자연스럽게 발전할 수 있음을 보여준다.

---

## 3. DeepSeek-R1

R1-Zero는 강력한 추론 능력을 보이지만, 추론 과정이 길고 구조가 불명확하며 영어·중국어 혼용 문제가 존재한다. 이를 해결하기 위해 4단계 학습 프레임워크를 도입한다.

| 단계 | 방법 | 목적 |
|---|---|---|
| Cold Start SFT | Human-aligned reasoning 데이터로 SFT | 추론 과정 가이드라인 제공 |
| First RL Stage | Rule-based Reward + Language Consistency Reward | 추론 성능 + 언어 혼합 문제 개선 |
| Rejection Sampling & SFT | RL 결과 중 좋은 데이터만 선택 후 재학습 | 강화학습 결과를 정제하여 다시 학습 |
| Second RL Stage | Rule-based + Language + Reward Model | 인간 선호도 반영 |

### 3.1 모델 기반 보상 (Model-based Rewards)

정답이 명확한 데이터는 규칙 기반 검증이 가능하지만, 일반적인 대화·창의적 글쓰기 등은 기준이 주관적이다. → 인간의 선호도를 모사하는 Reward Model을 도입한다.

**유익성 보상 모델 (Helpful Reward Model)**
- 사용자 질문과의 관련성 및 실질적 도움 여부를 평가한다
- 추론 과정에 간섭하지 않기 위해 최종 요약본만 평가한다
- 약 66,000개의 선호도 데이터 쌍으로 학습한다

**안전성 보상 모델 (Safety Reward Model)**
- 잠재적 위험·유해 콘텐츠 식별, 사회적 편향성 제거, 안전 가이드라인 준수 여부를 평가한다
- 모델의 전체 응답(추론 과정 + 최종 결과물)을 평가한다
- 106,000개의 프롬프트와 Safe/Unsafe 라벨링 데이터로 학습한다

### 3.2 훈련 세부 사항

**1차 RL 단계:** 언어 일관성 보상(Language Consistency Reward)을 도입하여 CoT에서 목표 언어 단어의 비율로 계산한다. 성능은 아주 미세하게 하락할 수 있으나, 가독성 확보 및 언어 혼용 문제가 개선된다.

$$\text{Reward}_{\text{language}} = \frac{\text{Num}(\text{Words}_{\text{target}})}{\text{Num}(\text{Words})}$$

**2차 RL 단계:** 추론 데이터에는 규칙 기반 보상, 일반 데이터에는 보상 모델을 적용하여 추론 성능 유지 + 인간 선호도(Helpful/Harmlessness)를 반영한다. 총 1,700 training steps로 구성되며, 마지막 400 steps에서만 preference-based reward를 적용한다. 모델 기반 preference reward를 너무 오래 사용하면 reward hacking이 발생할 수 있기 때문이다.

$$\text{Reward} = \text{Reward}_{\text{reasoning}} + \text{Reward}_{\text{general}} + \text{Reward}_{\text{language}}$$

---

## 4. 실험 결과

각 개발 단계별 특징적 성능 변화가 나타난다.

- **R1-Dev1:** Instruction-following 능력이 향상되나, cold-start 데이터 규모가 작아 추론 성능이 일부 감소한다
- **R1-Dev2:** Reasoning 중심 RL 학습으로 추론 성능이 향상되나, 사용자 선호 기반 벤치마크 개선이 제한적이다
- **R1-Dev3:** Reasoning + Non-reasoning 데이터 SFT로 추론 능력과 일반 언어 생성 능력을 동시에 향상시킨다
- **R1 (최종):** 코드·수학 성능은 소폭 향상되고, Instruction-following 및 사용자 선호 성능이 크게 개선된다 (AlpacaEval2.0, ArenaHard)

---

## 5. 윤리 및 안전성

추론 능력이 고도화됨에 따라 탈옥(Jailbreak) 공격 시 모델이 생성하는 유해 정보의 실행 가능성도 함께 높아진다. DeepSeek-R1은 단독으로 GPT-4o(24년 5월)와 유사한 중간 수준의 안전성을 확보하며, Risk Control System과 결합 시 최고 수준의 안전성에 도달한다. → 모델의 지능뿐 아니라, 이를 제어하는 안전 시스템과의 조화가 필수적이다.

---

## 6. 한계 및 향후 연구

**기술적 한계**
- 정해진 형식의 출력 능력이 부족하며 외부 도구(계산기, 검색 엔진) 활용 능력이 아직 미흡하다
- 쉬운 문제에도 너무 많은 토큰을 소모하는 과잉 사고(Overthinking) 비효율이 존재한다
- 추론 과정 중 언어가 섞이는 현상이 여전히 관찰된다

**도전 과제**
- 글쓰기처럼 정답이 없는 영역은 규칙 기반(Rule-based) 보상을 만들기 어렵다
- 모델 기반 보상을 쓸 경우 리워드 해킹 문제가 발생한다