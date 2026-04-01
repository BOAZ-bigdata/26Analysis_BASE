# SAM

## 배경
GPT처럼 프롬프트만으로 zero-shot 수행이 가능한 foundation model을 Image Segmentation에 적용하는 것이 목표다. CLIP/ALIGN이 이미지+텍스트 정렬로 zero-shot을 가능하게 했지만, segmentation 데이터가 부족해 동일한 접근이 어렵다. 이를 해결하기 위해 Task, Model, Data 세 가지를 함께 설계했다.

---

## Task — Promptable Segmentation
Zero-shot 일반화를 가능하게 하는 태스크로 promptable segmentation을 정의한다. 점, 박스, 마스크, 텍스트 등 다양한 프롬프트를 입력받아 segmentation mask를 출력하며, 모호한 입력에도 그럴듯한 결과를 반환한다. 다양한 프롬프트 조합으로 시뮬레이션해 일반화 성능을 높인다.

---

## Model — SAM 아키텍처
세 개의 모듈로 구성된다.

**Image Encoder** — MAE로 사전학습된 ViT를 사용한다. 확장성과 강력한 사전학습을 위해 선택했으며, 이미지당 한 번만 실행해 프롬프트 입력 전에 미리 계산해둔다.

**Prompt Encoder** — 모든 프롬프트를 동일한 벡터 공간으로 변환한다. 점·박스(sparse)는 위치 정보로, 텍스트는 CLIP으로, 마스크(dense)는 Conv로 처리한다.

**Mask Decoder** — Self-Attention으로 프롬프트끼리 이해 → Cross-Attention으로 프롬프트-이미지 관계 이해 → 업샘플링 → 픽셀별 확률 계산 → mask 출력. 모호성 해결을 위해 하나의 프롬프트에 3가지 스케일의 마스크를 동시 출력하고 신뢰도 점수로 순위를 매긴다.

**효율성** — 이미지 임베딩은 미리 계산하고, 이후 프롬프트 인코더+마스크 디코더만 실행해 CPU에서도 약 50ms 내 결과 출력이 가능하다. Loss = focal loss + dice loss.

---

## Data Engine — SA-1B 구축
Segmentation 데이터가 거의 없어 모델이 스스로 데이터를 생성하는 Data Engine을 구축했다. 3단계로 진행된다.

**1단계 Assisted Manual** (4.3M) — 사람 주도, AI 보조. 6번 재학습. 라벨링 시간 34초→14초, 이미지당 객체 수 20→44개.

**2단계 Semi-automatic** (10.2M) — AI 주도, 사람이 어려운 것만 보완. 5번 재학습. 이미지당 객체 수 44→72개, 누적 1020만 마스크.

**3단계 Fully automatic** (1.1B) — AI 완전 자동. 그리드로 점 뿌려 객체 유추, IoU+NMS로 필터링. 최종 **SA-1B 11억 개 마스크** 완성.

데이터 품질 평가 결과 94% IoU 90% 이상으로 대부분 고품질이다. 성별·피부톤 간 성능 차이는 거의 없으나, 유럽·아시아 등 고중간 소득 국가 비중이 높고 아프리카·저소득 국가 데이터는 적다.

---

## 실험 결과
23개의 다양한 데이터셋으로 zero-shot transfer 능력을 평가했다.

**Single Point** — 23개 중 16개 데이터셋에서 SOTA인 RITM보다 높은 성능. human study 평균 7~9점.

**Edge Detection** — 학습 없이 합리적인 엣지 맵 생성. 최신 SOTA(EDETR)보다는 낮고 초기 딥러닝 방법(HED)과 유사한 수준.

**Object Proposals** — SOTA(ViTDet-H)보다 낮지만 경쟁력 있는 수준.

**Instance Segmentation** — COCO/LVIS에서 Mask AP는 SOTA 대비 약간 낮지만, 경계 품질은 정성평가에서 더 높은 평가를 받는다.

**Text-to-Mask** — 텍스트만 입력 시 불안정하며, 추가 프롬프트(점 등)를 함께 줄 때 어느 정도 동작한다.

---

## 한계
세밀한 구조 감지 부족, 고화질 경계에서 전용 방법 대비 열세, 포인트가 많아질수록 전용 interactive 모델이 더 우수, ViT-H 사용 시 실시간 처리 어려움, text-to-mask 불안정이 있다. SAM은 segmentation foundation model이지만 CV 전체의 foundation model은 아니다.