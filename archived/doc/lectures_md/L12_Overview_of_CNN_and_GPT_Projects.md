# ECE 408 / CS 483 / CSE 408
## Lecture 12: Overview of CNN and GPT-2 Projects

**Instructor:** Hrishi Shah  
**학기:** Spring 2026

> Source: `Lectures_pdf/ece408-lecture12-CNN&GPT-Project-hs-SP26.pdf`

---

## 1. Lecture Briefing

이번 12강은 전형적인 "알고리즘 강의"라기보다, **ECE 408의 두 가지 최종 프로젝트 트랙(CNN, GPT-2)** 을 큰 그림에서 소개하는 오리엔테이션 강의다.

전체 흐름은 다음과 같다.

1. **CNN 프로젝트 개요**
   - 어떤 네트워크를 대상으로 무엇을 최적화하는지 설명
2. **왜 CNN만으로는 NLP를 다루기 어려운가**
   - 문장 문맥, 장거리 의존성 문제를 통해 한계를 설명
3. **왜 Transformer/GPT 프로젝트가 등장했는가**
   - self-attention이 문맥을 더 유연하게 처리한다는 점을 연결
4. **Transformer의 큰 구조 이해**
   - encoder, decoder, BERT, GPT의 차이 소개
5. **GPT-2 추론의 핵심 계산 블록 정리**
   - tokenization, embedding, layer norm, matmul, self-attention, softmax, GeLU, residual
6. **GPT 프로젝트 마일스톤 소개**
   - 어떤 구현과 최적화를 해야 하는지 안내

### 이 강의의 핵심 요지

- **CNN 프로젝트는 convolution forward pass 최적화가 중심이다.**
- **GPT 프로젝트는 decoder-only Transformer의 inference 전체를 CUDA로 가속하는 것이 중심이다.**
- **Transformer는 CNN보다 시퀀스 전체 문맥을 유연하게 볼 수 있지만, 계산량이 크다.**
- **결국 두 프로젝트 모두 본질은 GPU 커널 최적화와 end-to-end 성능 향상이다.**

---

## 2. Course Logistics and Release Notes

### Slide 2: course logistics

슬라이드에는 다음 공지가 나온다.

- Project Milestone 1이 금요일 마감
- 프로젝트를 pull해서 바로 시작하라고 강조
- 3일 late deadline 정책 적용
- 시험은 다음 주 화요일
- 리뷰 세션 공지

### 날짜 주의

슬라이드에는 다음 문장이 보인다.

- `Project Milestone 1 is due Friday, March 6th, 2025.`

하지만 이 강의는 **Spring 2026** 강의이고, PDF 표지도 **2026년 2월 26일**로 표시되어 있다.  
또한 **2026년 3월 6일은 실제로 금요일**이므로, 이 슬라이드의 `2025`는 이전 학기 자료에서 남은 **오타일 가능성이 높다**고 보는 것이 자연스럽다.

따라서 이 노트에서는 혼동을 피하기 위해 다음처럼 이해하면 된다.

- 시험: **2026년 3월 3일 화요일**
- 프로젝트 milestone 1 마감: 슬라이드 맥락상 **2026년 3월 6일 금요일**로 읽는 것이 타당

### Slide 3: project release

- milestone 1은 이미 release됨
- 자세한 내용은 `README.md`와 supporting documents 참고
- 향후 milestone 내용도 GitHub repo에 계속 추가됨
- future milestones에는 PrairieLearn / Gradescope 요소가 붙을 수 있음

즉, 이 강의는 "슬라이드만 보고 끝"이 아니라 **레포 문서를 함께 읽어야 실제 작업을 이해할 수 있다**는 점을 분명히 한다.

---

## 3. CNN Project Overview

### Slide 4: 프로젝트의 핵심 목표

CNN 프로젝트는 다음 작업을 중심으로 한다.

- **modified LeNet-5 CNN**의 convolution layer forward pass를 CUDA로 최적화
- 구현은 **Mini-DNN**이라는 C++ framework 위에서 이루어짐
- 분류 대상은 **Fashion MNIST**

강의에서 언급한 중요한 네트워크 파라미터:

- 첫 번째 convolution 호출:
  - input size: `86 x 86`
  - input channels: `1`
  - output features: `4`
- 두 번째 convolution 호출:
  - input size: `40 x 40`
  - input channels: `4`
  - output features: `16`
- convolution kernel size: `7 x 7`
- batch size: `100 ~ 10k images`

### 이 숫자들이 왜 중요한가

이 파라미터들은 곧바로 CUDA 커널의 workload shape가 된다.

예를 들어 첫 번째 호출의 경우:

- 입력 채널은 적지만
- 배치가 크고
- 출력 feature map이 여러 개이며
- kernel이 `7x7`로 작지 않다

즉, 단순한 2D convolution 예제를 넘어 **실제 프로젝트용 shape를 고려한 최적화**가 필요하다.

### 레포 README와 연결

레포의 [README_CNN.md](/u/ylee21/ece408git/Project/README_CNN.md)도 같은 방향을 설명한다.

- modified LeNet-5
- forward-pass of convolutional layers
- Fashion MNIST
- CUDA 최적화와 profiling이 프로젝트 목표

즉 11강이 convolution 계산 자체를 설명했다면, 12강은 그 내용을 **프로젝트 업무 정의**로 바꾸는 역할을 한다.

---

## 4. CNN Project Timeline

### Slide 5

강의 슬라이드 기준 CNN 프로젝트 타임라인:

- 모든 milestone은 **금요일 오후 8시 Central Time**
- 모든 학생이 **개별 제출**
- **코드 공유 금지**

마일스톤별 내용:

1. **Milestone 1**
   - CPU convolution 구현
   - GPU convolution 구현

2. **Milestone 2**
   - profiling
   - matrix unrolling
   - matrix unrolling용 kernel fusion

3. **Milestone 3**
   - convolution 및 unrolling에 대한 추가 커널 최적화

### README와의 연결

[README_CNN.md](/u/ylee21/ece408git/Project/README_CNN.md)에서는 milestone 설명이 조금 더 정식 문서 형태로 나온다.

- M1: basic CPU/GPU convolution
- M2: unrolling + fusion + profiling report
- M3: 다양한 GPU optimizations + final report

### 강의 포인트

CNN 프로젝트는 사실상 다음 식으로 볼 수 있다.

```text
Convolution 이해
-> 기본 GPU 구현
-> profiling으로 병목 찾기
-> unrolling/GEMM/fusion/tiling 최적화
-> end-to-end 성능 향상
```

---

## 5. Natural Language Processing and Why CNNs Struggle

### Slide 6: NLP란 무엇인가

강의는 language를 "정보를 전달하는 symbol arrangement"로 소개한다.

NLP 예시:

- speech recognition
- machine translation
- sentiment analysis

그리고 문장 이해에는 두 층위가 있음을 강조한다.

- **syntax**: 문장의 구조
- **semantics**: 문장의 의미

인간 언어는 애매하고 맥락 의존적이기 때문에 본질적으로 어렵다.

### Slide 7: CNN의 문맥 이해 한계

강의의 질문:

> “Hrishi was not able to visit Charles as his car broke down.”

여기서 질문은:

- `"his"`가 누구를 가리키는가?

슬라이드의 설명 요점:

- 작은 convolution window만 보면 `"Charles"` 쪽으로 잘못 연결할 수 있음
- CNN은 멀리 떨어진 단어 관계를 유연하게 다루기 어렵다
- 마스크 크기를 키우는 방식은 가능하지만, 문장마다 동적으로 조절하기 어렵다

### 왜 중요한 예시인가

이 문장은 **local pattern만으로는 부족하고 global context가 필요한 문제**를 보여 준다.

CNN은 잘하는 것:

- n-gram 비슷한 지역 패턴
- 짧은 구문

CNN이 약한 것:

- 멀리 떨어진 단어 사이의 관계
- 문장마다 달라지는 참조 구조
- 가변적인 장거리 문맥

### 짧은 직관 예시

```text
문장:
The trophy would not fit in the suitcase because it was too big.

질문:
"it"은 trophy인가 suitcase인가?
```

이런 문제도 주변 몇 단어만 보면 헷갈릴 수 있다.  
즉 NLP에서는 **local filter만으로는 부족한 상황이 자주 나온다.**

---

## 6. Why Transformers

### Slide 8: self-attention의 동기

self-attention은 텍스트 전체 시퀀스에서 dependency를 한 번에 포착할 수 있다.

강의 포인트:

- convolution mask와 달리 attention은 **입력 원소마다 다른 weight**를 줄 수 있다
- fixed kernel size 대신 문맥에 따라 중요한 토큰이 달라질 수 있다
- 그래서 `Attention is All You Need`는 convolution을 제거하고 attention 중심 구조를 제안했다

하지만 단점도 있다.

- **runtime이 sequence length에 대해 quadratic scaling**

즉 Transformer는 더 유연하지만, 계산 비용이 크다.

### 핵심 비교

#### CNN

- local receptive field
- 고정된 커널
- 상대적으로 구조가 단순
- 멀리 떨어진 토큰 관계 처리 약함

#### Transformer

- 전체 시퀀스 문맥을 볼 수 있음
- 입력별로 attention weight가 달라짐
- 문맥 이해가 더 강함
- 계산량과 메모리 사용량이 큼

### 프로젝트 동기

강의는 분명히 말한다.

- Transformer가 더 현대적이고
- 산업적으로도 더 관련성이 높아서
- GPT 프로젝트를 새로 도입했다

---

## 7. Encoder and Decoder Blocks

### Slide 9: encoder block

encoder의 역할:

- 텍스트 전체를 한 번에 읽고
- 각 단어가 다른 단어와 어떻게 연결되는지 이해

핵심 특징:

- 모든 token이 **과거와 미래 token 모두**를 볼 수 있음
- residual connection + layer norm으로 학습 안정화
- attention 이후 token별 FFN(two-layer fully connected network) 적용

### encoder의 직관

encoder는 "문장을 이해하는 기계"에 가깝다.

예:

```text
The bank of the river was steep.
```

여기서 `bank`가 금융기관이 아니라 강둑이라는 의미라는 것을  
문장 전체를 보고 해석하는 쪽이 encoder적 사고다.

### Slide 10: decoder block

decoder의 역할:

- 텍스트를 **autoregressive** 하게 한 단어씩 생성

핵심 특징:

- 미래 토큰을 보면 안 됨
- 그래서 **mask**가 필요
- seq2seq Transformer에서는 encoder output을 참고하는 cross-attention도 있음
- 하지만 **GPT는 그 encoder reference layer를 버린다**

### causal masking 예시

예를 들어 4번째 단어를 예측할 때:

- 볼 수 있는 것은 1, 2, 3번째 단어
- 5번째 이후 단어는 볼 수 없음

즉 attention matrix에서 미래 방향은 막혀 있어야 한다.

```text
token 4 예측 시:
can attend -> 1, 2, 3, 4
cannot attend -> 5, 6, 7, ...
```

### Slide 11: original Transformer

슬라이드에는 원래 Transformer 구조 그림이 나오며,

- encoder stack
- decoder stack
- attention
- FFN
- residual / normalization

의 전체 그림을 보여 준다.

---

## 8. BERT vs GPT

### Slide 12: BERT

강의 정리:

- **Architecture:** encoder blocks only
- **Context:** bidirectional
- **Learning:** masked language modeling
- **Best for:** NLU

즉 BERT는 문장 전체를 이해하는 작업에 강하다.

예:

- sentiment analysis
- document classification
- reading comprehension

### Slide 13: GPT

강의 정리:

- **Architecture:** decoder blocks only
- **Context:** unidirectional / causal
- **Learning:** next-token prediction
- **Best for:** NLG

예:

- chatbot
- essay writing
- summarization
- code generation

또한 프로젝트에서는:

- pretrained model이 주어지고
- training은 하지 않으며
- **inference로 text generation만 수행**

### 핵심 비교표

| 모델 | 구조 | 문맥 방향 | 주된 학습 목표 | 대표 용도 |
| --- | --- | --- | --- | --- |
| BERT | Encoder-only | 양방향 | 빈칸 맞추기 | 이해(NLU) |
| GPT | Decoder-only | 단방향(인과적) | 다음 토큰 예측 | 생성(NLG) |

### 시험 관점 포인트

- BERT는 "양방향 문맥 이해"
- GPT는 "과거만 보고 생성"

이 차이는 거의 반드시 구분할 수 있어야 한다.

---

## 9. A Transformer for Language Generation: GPT

### Slide 14

원래 Transformer는 번역 같은 seq2seq task에 잘 맞지만,  
텍스트 생성에는 그대로 쓰기엔 맞지 않는다.

GPT 쪽으로 가면서 달라지는 핵심:

- encoder stack 제거
- masked attention 사용
- future token을 보면 안 됨
- learnable positional encodings 사용
- transformer block을 여러 번 깊게 쌓음

### Slide 15: GPT architecture overview

슬라이드는 다음 특징을 강조한다.

- **decoder-only**
- **autoregressive**
- token을 한 번에 하나씩 생성
- 생성된 token이 다시 다음 입력으로 들어감
- transformer block을 여러 층 쌓을 수 있음
- attention head 수와 hidden size도 확장 가능

### 파라미터 수는 어떻게 커지는가?

슬라이드가 던지는 질문 중 하나:

> transformer block을 여러 층 쌓으면 파라미터 수는 어떻게 스케일하나?

대략 직관적으로:

- hidden size가 커질수록 각 matmul 비용이 크게 증가
- layer 수가 늘수록 전체 파라미터와 FLOP가 거의 선형적으로 증가
- attention head 수를 늘리면 representation capacity는 증가하지만 계산과 메모리 비용도 따라 증가

즉 GPT 프로젝트는 단순한 "문자열 생성" 프로젝트가 아니라,  
**큰 선형대수 계산을 반복 수행하는 추론 엔진을 CUDA로 만드는 프로젝트**다.

---

## 10. GPT Inference Pipeline

### Slide 16: inference란 무엇인가

inference는 **이미 학습된 모델을 사용해 실제 작업을 수행하는 것**이다.

중요 포인트:

- training이 아님
- dropout은 training용이므로 inference에서는 신경 쓰지 않음

이 강의는 GPT-2 inference를 여러 계산 블록으로 쪼개어 설명한다.

---

## 11. Tokenization and Embeddings

### Slide 17: tokenization

tokenization의 역할:

- 입력 문자열을 잘게 쪼개
- 숫자 token sequence로 바꾸는 것

목적:

- 사람이 쓰는 언어를 모델이 처리할 수 있는 숫자로 변환

성능 관점:

- 보통 전체 추론 비용에서 큰 비중은 아님

강의 메모:

- GPT-2는 **byte-pair encoding(BPE)** 사용
- 자주 등장하는 기호/문자 조합을 token으로 묶는다

### 예시

```text
문장:
"parallel programming is fun"

가능한 tokenization 예:
["parallel", " programming", " is", " fun"]
-> [token ids]
```

실제 GPT-2 토크나이저는 공백, 부분 문자열, 희귀 문자 처리가 더 정교하다.

### Slide 18: token embedding

NLP에서는 token과 의미를 보통 벡터로 표현한다.

- 비슷한 의미의 단어는 비슷한 방향의 벡터를 가질 수 있음
- 이 관계는 training 중 형성됨

강의에서 언급한 수치:

- **GPT-2 Small의 embedding dimension = 768**

즉 token 하나가 정수 ID에서 길이 768의 벡터로 바뀐다.

### Slide 19: encoding tokens

encoding의 역할:

- token ID를 embedding vector로 변환
- 의미뿐 아니라 위치 정보도 함께 반영

성능 관점:

- 보통 forward pass에서 한 번 수행되므로 상대적으로 부담이 적음

간단히 쓰면:

```text
token id -> lookup -> embedding vector
embedding + positional encoding -> transformer 입력
```

---

## 12. LayerNorm, MatMul, and the Core Compute Blocks

### Slide 20: layer normalization

operation:

- 각 row의 평균과 표준편차를 사용해 normalize
- 이후 scale/shift 적용

purpose:

- vanishing / exploding gradient 완화
- activation shift 완화
- 깊은 층에서도 입력 분포를 안정화

강의는 inference 설명 중이지만, layer norm의 학습 안정화 역할도 함께 상기시킨다.

### LayerNorm의 직관

토큰 벡터 하나가 너무 큰 값이나 너무 작은 값으로 치우치지 않게 정리해서  
뒤쪽 레이어가 더 안정적으로 처리하게 만든다.

### Slide 21: linear layers = matrix multiplication

강의는 아주 명확하게 말한다.

- linear transformation은 사실상 **matrix multiplication + bias**
- feature transformation, dimensionality mapping, reinterpretation에 쓰임
- **performance impact가 크다**

즉 GPT 프로젝트에서 matmul은 보조 연산이 아니라 **가장 핵심적인 계산 블록 중 하나**다.

### 간단한 예시

```text
입력 X: [T x C]
가중치 W: [C x D]
출력 Y: [T x D]

Y = XW + b
```

여기서:

- `T`: sequence length
- `C`: hidden size
- `D`: 변환 후 차원

Q, K, V 생성도 본질적으로 이 matmul 여러 개다.

---

## 13. Multi-Head Self-Attention

### Slide 22: self-attention은 무엇을 하나

강의 정리:

- 입력으로부터 `Q`, `K`, `V`를 만든다
- 두 번의 matrix multiplication과 softmax가 핵심
- dropout은 inference에서 고려하지 않음
- 성능 영향이 매우 큼

즉 attention은 개념적으로는 "문맥을 보는 메커니즘"이지만,  
구현적으로는 **매우 비싼 선형대수 파이프라인**이다.

### 기본 수식

\[
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
\]

\[
Attention(Q,K,V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

multi-head attention은 이 과정을 여러 head에 대해 수행한 뒤 합친다.

### Slide 23: Q/K/V 직관

강의 비유:

- query: 검색창에 넣는 질의
- keys: 검색 대상의 제목/설명
- values: 실제로 가져올 결과

토큰 관점에서 보면:

- 어떤 토큰은 "무엇을 찾고 싶은지"를 나타내는 query 역할
- 다른 토큰들은 "내가 이런 정보다"라는 key 역할
- 그리고 실제 전달할 내용은 value 역할

### 예시

```text
문장:
"The animal didn't cross the street because it was tired."
```

여기서 `it`을 해석할 때, attention은 `animal`, `cross`, `tired` 같은 토큰에
가중치를 다르게 줄 수 있다.

이 점이 고정 커널 CNN과 크게 다르다.

---

## 14. Softmax, GeLU, and Residual Connections

### Slide 24: softmax

softmax의 역할:

- raw score를 probability 비슷한 값으로 변환
- self-attention에서 어떤 토큰에 얼마나 집중할지 결정
- causal masking과 결합되어 미래 토큰을 못 보게 함

강의 주의:

- 슬라이드의 "formal definition"이 곧바로 프로젝트 구현과 완전히 동일한 것은 아님
- 실제 프로젝트에서는 numerical stability나 masking 구현 방식이 다를 수 있음

기본 softmax:

\[
softmax(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
\]

### Slide 25: GeLU

GeLU의 역할:

- linear transformation 뒤에 non-linearity 부여
- fully connected layers에서 더 복잡한 패턴 학습 가능하게 함

ReLU보다 부드러운 활성화로 이해하면 된다.

직관적으로:

- 작은 음수는 완전히 잘라 버리기보다 부드럽게 줄이고
- 큰 양수는 비교적 통과시킨다

### Slide 26: residual connection

residual의 역할:

- 이전 입력을 현재 출력에 더함
- 깊은 네트워크에서 정보 보존
- 학습/추론 안정성 향상

간단한 형태:

\[
Y = F(X) + X
\]

### 왜 중요한가

attention과 FFN이 아무리 강력해도,

- 각 블록이 입력을 완전히 덮어써 버리면 정보가 쉽게 손실될 수 있다
- residual은 "원래 의미를 유지하면서 새 정보를 추가"하는 역할을 한다

---

## 15. Why Multi-Head Attention Helps

### Slide 27

강의 요점:

- 여러 head가 입력의 서로 다른 측면에 집중할 수 있음
- richer context-aware representation 형성
- 더 복잡한 패턴과 관계를 포착
- 정확도 향상에 기여

### 직관

head마다 관심사가 다르다고 생각하면 쉽다.

- 어떤 head는 문법적 관계
- 어떤 head는 주어-동사 관계
- 어떤 head는 장거리 참조
- 어떤 head는 문장 내 위치 패턴

를 상대적으로 더 잘 볼 수 있다.

물론 실제로 head마다 역할이 깔끔히 분리되는 것은 아니지만,  
강의 차원에서는 **"여러 관점으로 본다"** 정도로 이해하면 충분하다.

---

## 16. GPT Project Timeline

### Slide 28

GPT 프로젝트 마감 정책:

- 모든 milestone은 **금요일 오후 11:59 Central Time**

마일스톤 구성:

1. **Milestone 1**
   - 기본 GPU kernels 구현
   - Transformer architecture 이해

2. **Milestone 2**
   - kernel profiling + system-level profiling
   - 최적화 제안
   - 여러 required optimization 구현

3. **Milestone 3**
   - 추가 required optimization 구현
   - M2에서 제안한 최적화 반영
   - end-to-end performance analysis 및 final report

메인 목표:

- **end-to-end performance 최대화**

### README와의 연결

[README_GPT.md](/u/ylee21/ece408git/Project/README_GPT.md)에서는 이 내용을 좀 더 확장해 설명한다.

특히 프로젝트에서 다루는 핵심 커널:

- `encoder_forward`
- `layernorm_forward`
- `matmul_forward`
- `attention_forward`
- `residual_forward`
- `gelu_forward`

즉 GPT 프로젝트는 attention 하나만 구현하는 것이 아니라,  
**GPT-2 inference 전체를 이루는 주요 연산 블록 전부를 CUDA로 다루는 프로젝트**다.

---

## 17. Lecture-Wide Summary

이번 강의는 두 프로젝트를 다음처럼 대비시킨다.

### CNN project

- 공간적 구조를 가진 데이터 처리
- convolution forward pass 최적화
- unrolling, fusion, convolution kernel optimization이 핵심

### GPT project

- 시퀀스 데이터와 문맥 처리
- decoder-only Transformer inference 가속
- matmul, attention, layernorm, softmax, residual, GeLU 등 여러 커널 최적화가 핵심

### 가장 중요한 한 줄

둘 다 AI 프로젝트처럼 보이지만, ECE 408 관점에서 핵심은 결국:

- **계산 구조를 이해하고**
- **CUDA 커널로 옮기고**
- **profiling으로 병목을 찾고**
- **end-to-end 성능을 끌어올리는 것**

---

## 18. 꼭 기억할 질문들

강의 중 직접 나오거나, 강의 내용을 이해했는지 점검하기 좋은 질문들:

1. 왜 CNN은 NLP에서 장거리 의존성을 다루기 어렵나?
2. self-attention은 convolution과 달리 무엇을 동적으로 바꿀 수 있나?
3. encoder와 decoder의 가장 중요한 차이는 무엇인가?
4. 왜 GPT는 decoder-only 구조를 쓰는가?
5. BERT와 GPT는 각각 어떤 종류의 작업에 더 적합한가?
6. GPT inference에서 가장 성능 영향이 큰 계산은 무엇인가?
7. residual connection과 layer normalization은 왜 필요한가?
8. GPT 프로젝트의 목표는 개별 커널 속도 향상인가, 아니면 end-to-end 성능 향상인가?

---

## 19. 빠른 복습용 한 페이지 요약

```text
Lecture 12
= CNN 프로젝트 소개 + GPT 프로젝트 소개

CNN 프로젝트:
- modified LeNet-5
- Fashion MNIST
- convolution forward pass CUDA 최적화
- M1: CPU/GPU conv
- M2: profiling + unrolling + fusion
- M3: 추가 최적화

CNN의 NLP 한계:
- local window만 보기 쉬움
- 장거리 문맥 처리 약함

Transformer:
- self-attention으로 전체 문맥 반영
- 계산량은 큼

BERT:
- encoder-only
- bidirectional
- understanding

GPT:
- decoder-only
- causal
- next-token prediction
- generation

GPT inference 주요 블록:
- tokenization
- embedding
- layer norm
- matmul
- self-attention
- softmax
- GeLU
- residual

GPT 프로젝트 목표:
- 여러 커널을 최적화해 end-to-end 성능 최대화
```

