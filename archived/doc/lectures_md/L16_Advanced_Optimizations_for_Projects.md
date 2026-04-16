# ECE 408 / CS 483 / CSE 408
## Lecture 16: Advanced Optimizations for Projects

**Instructor:** Hrishi Shah  
**학기:** Spring 2026

> Source: `Lectures_pdf/ece408-lecture16-Advanced-Optimizations-SP26.pdf`

---

## 1. Lecture Briefing

이번 16강은 지금까지 배운 CUDA 최적화 기법을 프로젝트 수준으로 확장하는 강의다.

앞 강의들에서는:

- shared memory tiling
- reduction
- profiling
- histogram privatization

같은 개별 기법을 배웠다면, 이번 강의는 그 기법들을 **실제 큰 연산에 어떻게 조합해서 더 높은 성능을 내는가**에 초점을 둔다.

특히 중심이 되는 예시는 **matrix multiplication**이다.

강의가 던지는 핵심 질문은 다음과 같다.

- basic shared memory tiling만으로 현대 GPU 성능을 다 쓸 수 있는가?
- shared memory보다 더 빠른 자원은 무엇인가?
- matmul에서 register, tensor core, compiler optimization을 어떻게 활용할 수 있는가?
- 입력 shape가 비정상적일 때, 예를 들어 skinny matrix에서는 어떤 매핑이 더 좋은가?
- FLOP 수보다 **memory traffic**가 더 중요한 경우는 언제인가?

### 이 강의의 핵심 요지

- **기본 shared-memory tiling만으로는 현대 GPU의 compute throughput을 충분히 활용하기 어렵다.**
- **register tiling + thread coarsening**은 matmul reuse를 더 끌어올리는 핵심 기법이다.
- **Tensor Core / TF32 / WMMA**는 NVIDIA GPU에서 matrix-heavy workload를 크게 가속하는 실전 도구다.
- **Split-K, restrict, loop unrolling**은 알고리즘/컴파일러/shape 차원의 추가 최적화 포인트다.
- **FlashAttention**은 FLOP 자체보다 **IO bottleneck**을 줄이는 것이 더 중요할 수 있음을 보여 주는 대표 사례다.

---

## 2. Course Reminders and Context

### Slides 2-3: course logistics

강의 초반 공지:

- Project milestone 2 진행 시작
- Lab 6은 이번 주 마감
- profiling job은 Delta에서 오래 걸릴 수 있으니 미루지 말 것
- README와 milestone 안내 문서 업데이트됨

즉 이 강의는 이론 강의이면서 동시에:

- milestone 2
- Lab 6 직전
- profiling 결과를 바탕으로 성능 개선이 필요한 시점

에 맞춘 **실전 최적화 가이드** 성격이 강하다.

---

## 3. Why Matrix Multiplication Still Matters

### Slide 4

강의는 먼저 matmul이 왜 중요한지 다시 강조한다.

- scientific computing
- graphics
- data analytics
- deep learning

에서 매우 자주 등장하고, 특히 ML에서는 전체 FLOP의 대부분을 차지할 수 있다.

핵심 메시지:

- dense matmul은 compute도 크고 memory traffic도 크다
- 그래서 하드웨어와 라이브러리 모두 matmul에 맞춰 매우 강하게 최적화되어 있다

즉 프로젝트에서 CNN이든 GPT든, **큰 matrix operation을 빠르게 만드는 것**이 전체 성능의 중심이 된다.

---

## 4. Review: Basic Shared Memory Tiling

### Slides 5-8

강의는 먼저 Lab 3 스타일의 기본 shared-memory tiled matmul을 복습한다.

구조:

- `T x U` thread block이 `T x U` output tile 계산
- 입력 `M`에서 `T x S` tile, 입력 `N`에서 `S x U` tile을 로드
- 이전 강의/랩에서는 보통 `T = U = S`

### 중요한 관찰

- tile reuse는 분명히 있다
- 하지만 block 크기에는 한계가 있다
- A40 같은 GPU에서 thread block 최대 크기는 `1024`
- reuse만으로는 GPU peak compute throughput까지 못 올라간다

강의가 제시하는 수치 요점:

- A40 peak FP32 throughput: `37.4 TFLOPs`
- peak memory throughput: `696 GB/s`
- basic tiling으로 input reuse가 최대 `32x` 정도여도 충분하지 않다

즉 결론은:

**shared memory tiling은 필요하지만, 그것만으로는 부족하다.**

---

## 5. Parameter Tuning Beyond Square Tiles

### Slides 6-8

강의는 기본 tiling에서 자주 숨어 있는 가정을 흔든다.

- thread block은 꼭 square일 필요가 있는가?
- `T = U`여야 하는가?
- shared dimension `S`는 `T`, `U`와 같아야 하는가?

핵심 아이디어:

- output tile은 non-square일 수 있다
- shared dimension `S`도 조정 가능하다
- `T`, `U`를 키우면 input reuse가 증가
- `S`를 키우면 `__syncthreads()` 횟수를 줄일 수 있다

즉 matmul tiling은 단순히 “16x16 쓰면 된다”가 아니라,

- reuse
- synchronization overhead
- thread block size
- shared memory usage

를 함께 보는 **parameter tuning 문제**다.

---

## 6. Registers: Faster Than Shared Memory

### Slide 9

강의가 던지는 질문:

> shared memory보다 더 빠른 것은?

답:

- **register**

장점:

- access latency가 매우 낮다
- throughput이 매우 높다

단점:

- thread local이다
- 다른 thread와 직접 재사용할 수 없다

즉 register는 매우 빠르지만,
shared memory처럼 thread 간 협업 reuse를 제공하지는 않는다.

그래서 강의는 다음 단계로:

- register
- shared memory

를 **함께 쓰는 구조**를 소개한다.

---

## 7. Joint Register and Shared Memory Tiling

### Slides 10-17

이번 강의의 가장 중요한 부분 중 하나다.

핵심 아이디어:

- 각 thread가 output 하나만 계산하지 말고
- **여러 output 값을 계산하게 coarsen**
- 그 과정에서 한 입력 tile 일부를 register에 들고 reuse
- 다른 입력 tile은 shared memory에 저장

### 구조

강의는 다음과 같이 설명한다.

- block은 `T` threads를 가진 1D block
- 각 thread는 `U`개의 output 값을 계산
- `M`의 한 tile은 register에
- `N`의 한 tile은 shared memory에

각 iteration에서:

1. block 전체가 `N`의 `S x U` tile을 shared memory로 로드
2. 각 thread는 `M`의 `S`개 값을 register에 로드
3. 각 thread는 자기 담당 `U` outputs에 대해 multiply-accumulate 수행
4. 다음 tile로 이동

### 왜 좋은가

- shared memory reuse는 유지
- register reuse까지 추가
- thread 하나가 더 많은 일을 수행
- output reuse가 늘어난다

### 파라미터 관계

Slides 11-15 핵심:

- thread block 크기 = `T`
- 각 thread output 개수 = `U`
- `N` tile 크기 = `S x U`
- thread 수가 `T`개이고, 각 thread가 `N` tile 원소 1개씩 로드하게 하면:

\[
S \cdot U = T \Rightarrow S = T/U
\]

즉 `T`, `U`, `S`는 독립적이지 않고 서로 trade-off가 있다.

### 실전 주의점

- `U`를 너무 크게 잡으면 register pressure 증가
- output tile이 커질수록 reuse는 좋지만 resource pressure도 증가
- `__syncthreads()`는 여전히 필요

강의는 `U = 16` 정도를 reasonable 예로 언급하지만,
최적 값은 하드웨어와 shape에 따라 달라진다.

---

## 8. Coalescing Concerns in Joint Tiling

### Slide 17

강의는 중요한 질문을 던진다.

- `N`을 shared memory로 로드할 때는 coalescing 맞추기 쉽다
- 그런데 `M`은 각 thread가 한 row의 연속값을 register로 읽는다
- consecutive thread는 다른 row를 읽게 된다

즉:

- thread 내부에서는 연속
- thread 간에는 row가 바뀜

이어서 access pattern이 완전히 coalesced되지 않을 수 있다.

핵심 메시지:

**register tiling은 reuse를 늘려주지만, memory access pattern이 같이 좋아지는 것은 아니다.**

따라서 최적화는 항상:

- reuse
- coalescing
- occupancy
- register pressure

를 같이 봐야 한다.

---

## 9. Tensor Cores and TF32

### Slides 18-21

강의는 A40의 또 다른 숫자를 꺼낸다.

- FP32 peak: `37.4 TFLOPs`
- TF32 peak: `74.8 TFLOPs`

왜 더 높나?

- **Tensor Cores**

### Tensor Cores란

- NVIDIA GPU의 specialized matrix-matrix hardware unit
- Volta 이후 세대에서 도입
- matmul efficiency를 크게 높임

### TF32란

- Tensor Float 32-bit 계열 format
- exponent는 FP32와 동일
- mantissa precision은 낮춤
- deep learning에는 충분한 정밀도인 경우가 많음

또 중요한 실전 포인트:

- TF32 값은 `float` 변수에 저장 가능
- 필요하면 `float_to_tf32` intrinsic으로 cast 가능

즉 TF32는 “새로운 완전히 별도 타입”이라기보다,
Tensor Core 친화적 계산을 위한 **reduced-precision FP32 계열**로 이해하면 된다.

---

## 10. WMMA API

### Slides 22-28

Tensor Core를 프로그래머가 직접 쓰는 대표 API가:

- `nvcuda::wmma`

강의는 WMMA를 warp-level matrix multiply-and-accumulate interface로 소개한다.

### 개념

Tensor Core는 기본적으로:

\[
D = A \cdot B + C
\]

를 warp-synchronous하게 수행한다.

### WMMA 핵심 개념

- `fragment`
  - `matrix_a`
  - `matrix_b`
  - `accumulator`
- fragment shape는 `m`, `n`, `k`로 정의
- multiplicand datatype은:
  - `double`
  - `float`
  - `half`
  - `nv_bfloat16`
  - `char`
  - `unsigned char`
  - `wmma::precision::tf32`
- accumulator datatype은:
  - `double`
  - `float`
  - `int`
  - `half`

### 사용 흐름

1. fragment 생성
2. `load_matrix_sync`로 메모리에서 fragment 로드
3. tensor core matmul-accumulate 수행
4. 결과 fragment를 메모리로 저장

주의점:

- fragment dimension restriction이 있다
- `load_matrix_sync`의 leading dimension 의미를 정확히 이해해야 한다
- row-major/column-major layout 차이를 주의해야 한다

강의 메시지:

**Tensor Core는 매우 강력하지만, 사용 규칙과 shape restriction이 있다.**

---

## 11. Skinny Matrices and Split-K

### Slides 31-33

강의는 matmul shape가 일반적이지 않을 때를 다룬다.

특히:

- inner dimension `K`가 매우 큰 skinny matrix

문제:

- 기존 output-tile mapping으로는 launched block 수가 적어짐
- 한 block이 엄청 긴 `K` loop를 오래 수행
- 병렬성이 부족해질 수 있다

### Split-K 아이디어

- shared dimension `K`를 여러 조각으로 split
- 여러 block이 **같은 output tile**을 담당
- 각 block은 partial result만 계산
- 마지막에 partial result를 합쳐 최종 output 생성

### 장점

- block 수를 늘려 병렬성 증가
- long-K case에서 활용 가능

### 주의점

- split 개수는 tuning variable
- partial result accumulation이 추가로 필요
- global atomic accumulation은 비쌀 수 있음
- reduction kernel을 추가로 쓰는 것이 더 나을 수 있음
- floating-point는 associative하지 않아서 accumulation order가 달라지면 수치 결과가 조금 바뀔 수 있음

즉 Split-K는 “무조건 빠른 기법”이 아니라:

- shape가 skinny한가
- atomic vs reduction 중 무엇이 더 나은가
- numerical variation이 허용되는가

를 보고 선택하는 방법이다.

---

## 12. Compiler-Aware Optimizations

### Slides 34-39

이번 강의는 알고리즘뿐 아니라 compiler가 보는 관점도 강조한다.

### Pointer aliasing

문제:

- compiler는 일반적으로 포인터들이 alias할 수 있다고 가정
- 즉 `a`, `b`, `c`가 같은 주소를 가리킬 가능성을 보수적으로 고려
- 그 결과 불필요한 reload나 redundancy가 생길 수 있다

예:

- 원래는 register에 담아 재사용할 수 있는 값을
- compiler가 안전을 위해 다시 global memory에서 읽게 만들 수 있다

### `restrict` keyword

`restrict`는:

- 이 포인터를 통해 쓴 데이터는
- 다른 restrict 포인터를 통해 읽히지 않는다고
- compiler에게 약속하는 것

효과:

- registerization
- reordering
- 불필요한 reload 제거

즉 특정 경우 significant speedup 가능

### Loop unrolling

강의는 loop overhead도 무시할 수 없다고 설명한다.

- exit condition 계산
- induction variable increment
- branch overhead

가 큰 loop에서는 성능에 영향을 준다.

해결:

- manual unrolling
- compiler directive

하지만 trade-off도 있다.

- code size 증가
- register pressure 증가 가능
- 이미 nvcc가 aggressive unrolling을 하므로 수동 unrolling이 항상 이득은 아님

핵심 메시지:

**compiler optimization은 공짜 성능 향상이 될 수 있지만, 무조건 믿거나 무조건 수동으로 덮어쓰면 안 된다.**

---

## 13. IO-Awareness and FlashAttention

### Slides 40-41

강의 마지막은 attention으로 넘어간다.

표준 self-attention 구현은:

- `S`
- `P`

같은 `N x N` intermediate matrix를 global memory에 읽고 쓰며,
각 단계가 separate kernel인 경우가 많다.

즉 문제의 핵심은 FLOP만이 아니다.

- **global memory traffic**
- **intermediate materialization**

가 큰 병목이 된다.

### FlashAttention 핵심 아이디어

- 수학적으로 exact attention 결과는 유지
- 하지만 access pattern을 shared memory/cache에 맞게 재구성
- Key/Value tile을 global memory에서 shared memory로 load
- Query를 tile-by-tile sweep
- intermediate를 거대한 `N x N` 형태로 global memory에 계속 쓰지 않음

강의가 강조하는 요점:

**modern optimization에서는 FLOP reduction보다 IO reduction이 더 중요할 때가 많다.**

이 메시지는 matmul, attention, project milestone 모두에 직접 연결된다.

---

## 14. Lab / Project Connection

이번 강의는 Lab 6보다 프로젝트 최적화와 더 강하게 연결된다. 그래도 Lab 6 전에 읽어둘 가치가 있다.

### Lab 6와 연결되는 부분

- thread coarsening 사고방식
- reduction 결과 accumulation 구조
- compiler overhead / unrolling 감각

### CNN / GPT project와 연결되는 부분

- matmul bottleneck 이해
- register + shared memory tiling
- Tensor Core / TF32 / WMMA
- shape-aware optimization (skinny matrices, Split-K)
- IO-aware design (FlashAttention)

즉 16강은 “커널 하나를 맞게 짜는 법”보다,
**프로젝트에서 어디를 어떻게 더 최적화할 것인가**를 보는 강의다.

---

## 15. Practical Takeaways

프로젝트/실습 전에 챙길 체크리스트:

1. basic tiling만으로 충분한지 의심하라
2. register reuse를 늘릴 수 있는지 보라
3. coalescing이 깨지지 않는지 확인하라
4. Tensor Core를 쓸 수 있는 datatype / tile shape인지 확인하라
5. 입력 shape가 skinny하면 Split-K를 고려하라
6. pointer aliasing 때문에 compiler가 보수적으로 행동하는지 보라
7. loop unrolling이 실제로 이득인지 profile로 확인하라
8. FLOP보다 memory traffic가 병목인지 항상 확인하라

---

## 16. One-Line Summary

Lecture 16 = **matmul-centered advanced optimization toolbox**

- shared memory 이후엔 register tiling
- 더 나아가 Tensor Core / WMMA
- shape가 다르면 Split-K
- compiler도 최적화의 일부
- attention에서는 IO-aware design이 핵심

