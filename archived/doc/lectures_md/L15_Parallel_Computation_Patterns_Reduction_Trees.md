# ECE 408 / CS 483 / CSE 408
## Lecture 15: Parallel Computation Patterns - Reduction Trees

**Instructor:** Volodymyr Kindratenko  
**학기:** Spring 2026

> Source: `Lectures_pdf/ece408-lecture15-reduction-tree-vk-SP26.pdf`

---

## 1. Lecture Briefing

이번 15강은 병렬 알고리즘에서 가장 자주 나오는 패턴 중 하나인 **reduction**을 다룬다.

reduction은 아주 단순한 문제처럼 보인다.

- 배열 원소들을 모두 더하기
- 최솟값 찾기
- 최댓값 찾기
- 곱하기

하지만 GPU에서는 다음 문제가 같이 따라온다.

- thread 수가 단계마다 줄어드는 **diminishing parallelism**
- 단계마다 필요한 **synchronization**
- naive 구현에서 생기는 **control divergence**
- global memory를 계속 읽고 쓰는 비효율

그래서 강의는 reduction을 점점 개선해 간다.

1. sequential reduction
2. atomic 기반 parallel reduction
3. reduction tree
4. segmented reduction
5. coalesced + reduced divergence version
6. shared-memory version
7. warp shuffle version
8. cooperative groups version
9. two-stage warp reduction
10. thread coarsening

### 이 강의의 핵심 요지

- **reduction은 associative + commutative 연산에 매우 잘 맞는 병렬 패턴이다.**
- **atomic으로도 구현 가능하지만 성능이 매우 나쁘다.**
- **좋은 reduction kernel은 coalescing, synchronization overhead, divergence를 함께 줄여야 한다.**
- **shared memory와 warp shuffle이 큰 성능 향상을 준다.**
- **warp-level primitive와 thread coarsening은 최신 GPU reduction 최적화의 핵심 도구다.**

---

## 2. Course Reminders and Objectives

### Slide 2: course reminders

강의 슬라이드 공지:

- Lab 5 이번 주 금요일 마감
- Lab 6도 release됨, 2주 후 마감
- Labs 1-4 채점 확인
- Project milestone 2 곧 release
- Midterm 1 채점 완료, 재채점 요청 마감은 금요일 자정

### Slide 3: today’s objectives

- reduction의 기본 개념 학습
- reduction parallelization 전략 학습
- GPU reduction의 성능 이슈 이해
- warp-level primitives 학습
- cooperative groups를 이용한 warp-level programming 학습

즉 15강은 reduction 자체뿐 아니라,
**GPU 위에서 reduction을 빠르게 구현하는 현대적 방법**을 다루는 강의다.

---

## 3. Reduction as a Parallel Primitive

### Slides 4-5: theoretical importance

강의는 Guy Blelloch의 scan/prefix computation 관련 인용으로 시작한다.

핵심 메시지:

- scan과 reduction 계열 연산은 병렬 알고리즘에서 매우 중요한 primitive
- arbitrary memory access보다
- 구조화된 병렬 패턴이 구현도 쉽고 하드웨어 친화적

이 강의는 reduction 자체를 다루지만,
scan과 reduction이 병렬 알고리즘의 핵심 building block이라는 큰 맥락을 함께 보여 준다.

### Slide 6: reduction이란

reduction은 **입력 집합을 하나의 값으로 줄이는 연산**이다.

예:

- sum
- product
- min
- max

강의에서 reduction 연산의 조건:

- **Associative**
- **Commutative**
- **Identity value가 정의됨**

예:

- sum의 identity = `0`
- product의 identity = `1`
- min의 identity = 매우 큰 값
- max의 identity = 매우 작은 값

### Slide 7: 일반적인 reduction 형태

연산자 `⊕`와 identity `I`가 주어지면:

```text
result <- I
for each value X in input:
    result <- result ⊕ X
```

즉 reduction은 단순 loop처럼 보이지만,
연산 순서를 바꿔도 결과가 같아야 병렬화가 자연스럽다.

---

## 4. Sequential Reduction and the Bad Atomic Baseline

### Slide 8: sequential reduction

강의의 sum 예시:

```c
sum = 0.0f;
for (i = 0; i < N; ++i) {
    sum += input[i];
}
```

일반형:

```c
acc = IDENTITY;
for (i = 0; i < N; ++i) {
    acc = f(acc, input[i]);
}
```

### Slide 9: atomic 기반 parallel reduction

가장 단순한 parallel version:

```c
unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < N) {
    atomicAdd(sum, input[i]);  // sum is a pointer to global memory
}
```

### 왜 성능이 나쁜가

강의 포인트:

- 모든 addition이 결국 hardware에 의해 serialized

즉 correctness는 맞지만,
실제로는 대부분의 thread가 같은 global memory location `sum` 하나를 두드린다.

이는 14강의 histogram hot-bin 문제와 거의 같은 구조다.

### 핵심 정리

atomic reduction은:

- 구현은 쉬움
- correctness는 맞음
- 성능은 매우 나쁨

따라서 실제 GPU reduction은 tree 형태로 바꿔야 한다.

---

## 5. Reduction Tree

### Slide 10: log(N) 단계 reduction

아이디어:

- 각 단계에서 thread가 두 값을 합침
- 단계가 진행될수록 active thread 수가 절반으로 감소
- 총 `log2(N)` 단계

예를 들어 입력이:

```text
4 7 2 3 8 5 9 6
```

이면:

```text
Step 1: 11 5 13 15
Step 2: 16 28
Step 3: 44
```

즉 pairwise sum을 트리처럼 쌓아 올리는 구조다.

### Slide 11: work efficiency

강의는 연산 수를 계산해:

\[
N - 1
\]

이 됨을 보인다.

즉 reduction tree는 **work-efficient**

왜냐하면 sequential sum도 본질적으로 `N-1`번 덧셈이 필요하기 때문이다.

### Slide 12: step 수는 적지만 resource가 필요함

강의 포인트:

- step 수는 `log2(N)`
- 예: `N = 1,000,000`이면 약 20 steps

좋아 보이지만:

- peak parallelism은 `N/2`
- 평균적으로도 많은 thread가 필요

즉 reduction tree는 빠르지만,
**충분한 병렬 자원**이 있을 때 가장 잘 작동한다.

### Slide 13: diminishing parallelism

reduction은 단계마다 active thread가 반으로 줄어든다.

즉:

- 처음엔 병렬성이 매우 크지만
- 후반으로 갈수록 병렬성이 급격히 줄어든다

이 현상은 combinational logic부터 HPC까지 널리 등장하는 패턴이다.

---

## 6. Segmented Reduction for CUDA

### Slide 14: block 간 동기화 문제

GPU에서는:

- thread block 내부 동기화는 가능
- block 간 동기화는 kernel 안에서 직접 할 수 없음

따라서 전체 reduction을 한 번에 끝내기 어렵다.

해결책:

- **segmented reduction**

### segmented reduction 아이디어

- 각 block이 입력의 한 segment를 reduction
- block마다 partial sum 하나를 생성
- 이 partial sum들을 다시 합침

강의의 간단한 설명:

- every thread block reduces a segment
- produces a partial sum
- partial sum is atomically added to final sum

### Slide 15: 그림 직관

입력 전체를:

```text
Segment 0
Segment 1
...
Segment B-1
```

로 나누고, 각 block이 자기 segment를 처리한다.

### 실전 관점

보통 구현은 두 가지 방식 중 하나다.

1. partial sum을 global memory 배열에 저장하고 host 또는 다음 kernel에서 재감소
2. partial sum을 final result에 atomic add

Lab 6 README에서도 이 맥락이 보인다.

- [README.md](/u/ylee21/ece408git/lab6/README.md)

여기서는 block별 reduction sum을 만들고,
host loop로 최종 합을 구하도록 안내한다.

---

## 7. First CUDA Reduction Strategy

### Slides 16-18

강의의 간단한 CUDA 전략:

- 입력은 global memory에 있음
- block 하나가 `2M`개의 값을 줄여서 1개로 만듦
- `M = blockDim.x`
- intermediate 결과도 global memory에 기록

즉 block당:

- 처음에 `2M` 입력
- 마지막에 1 output

### Slide 18: simple data mapping

각 thread는 처음에 두 인접한 값을 담당한다.

- thread 0 -> indices 0, 1
- thread 1 -> indices 2, 3
- ...

첫 단계 후 결과는 짝수 index 쪽에 저장된다.

다음 단계마다:

- stride는 두 배
- active thread는 절반

마지막엔 index 0에 segment reduction 결과가 남는다.

---

## 8. Naive Per-Block Reduction Tree and Its Problems

### Slide 20: naive code

```c
unsigned int segment = 2 * blockDim.x * blockIdx.x;
unsigned int i = segment + 2 * threadIdx.x;

for (unsigned int stride = 1; stride <= BLOCK_DIM; stride *= 2) {
    if (threadIdx.x % stride == 0) {
        input[i] += input[i + stride];
    }
    __syncthreads();
}
```

### 코드 해설

- block이 담당하는 segment 시작점: `segment`
- 각 thread는 현재 단계에서 두 값을 더하고
- 결과를 앞쪽 위치에 저장
- 매 단계 뒤 전체 block sync

### 강의가 지적한 문제

1. **input access가 coalesced되지 않음**
2. **control divergence**

### 왜 divergence가 생기나

`if(threadIdx.x % stride == 0)` 조건 때문에,

- 어떤 thread는 일하고
- 어떤 thread는 쉬게 된다

warp 안에서 조건이 섞이면 branch divergence가 생긴다.

### Slide 21: control divergence problem

후반 단계로 갈수록:

- 살아남은 thread는 적고
- 나머지는 떨어져 나간다

즉 narrowing parallelism이 SIMD 실행 효율까지 떨어뜨린다.

---

## 9. Reordering the Tree to Improve Coalescing and Reduce Divergence

### Slides 22-24

강의는 reduction 순서를 뒤집어 더 좋은 매핑을 제안한다.

핵심 아이디어:

- 초기 stride를 크게 잡고
- 점점 줄여 가는 방식

### 개선 코드

```c
unsigned int segment = 2 * blockDim.x * blockIdx.x;
unsigned int i = segment + threadIdx.x;

for (unsigned int stride = BLOCK_DIM; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
        input[i] += input[i + stride];
    }
    __syncthreads();
}
```

### 왜 더 좋은가

1. **coalescing 개선**
   - `threadIdx.x < stride` 구조는 인접 thread가 인접 위치를 읽게 만들기 쉬움

2. **divergence 감소**
   - 초반에는 warp 단위로 더 깔끔하게 active/inactive가 갈려서 branch behavior가 나아짐

### Slide 23: divergence reduced

강의 그림은 이 재배치가 후반부 divergence를 완전히 없애진 못해도,
기존 방식보다 훨씬 낫다는 점을 보여 준다.

---

## 10. Shared Memory Reduction

### Slide 25: data reuse

강의 포인트:

- 실제 값 자체가 여러 번 재사용되는 것은 아니지만
- 같은 memory location을 반복 읽고 쓰게 된다

따라서:

- 처음에 global memory에서 shared memory로 가져오고
- 이후 reduction tree는 shared memory에서 수행

하는 것이 좋다.

장점:

- global memory traffic 감소
- input 배열을 직접 덮어쓰지 않아도 됨

### Slide 27: shared memory code

```c
unsigned int segment = 2 * blockDim.x * blockIdx.x;
unsigned int i = segment + threadIdx.x;

__shared__ float input_s[BLOCK_DIM];
input_s[threadIdx.x] = input[i] + input[i + BLOCK_DIM];
__syncthreads();

for (unsigned int stride = BLOCK_DIM / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
        input_s[threadIdx.x] += input_s[threadIdx.x + stride];
    }
    __syncthreads();
}
```

### 해설

초기 단계에서:

- 각 thread가 global memory 두 값을 읽어 바로 합친 뒤
- shared memory에 저장

그 후:

- reduction의 나머지 단계는 shared memory 내부에서 수행

즉 global memory read 수를 크게 줄인다.

### 장점 요약

- coalesced global load 가능
- 이후 access는 shared memory라 빠름
- global memory write/read 반복 제거

---

## 11. Why the Last Warp Should Be Treated Differently

### Slide 28: reducing synchronization

강의 포인트:

- reduction 후반엔 active thread가 한 warp만 남는다
- 이때는 block 전체를 위한 shared memory + `__syncthreads()`가 과하다

왜냐하면:

- warp 내부 thread는 더 빠르게 통신할 수 있고
- block-wide barrier는 불필요한 overhead

### Slide 29: warp-level primitives

CUDA는 warp 내부 데이터 교환을 위한 내장 함수들을 제공한다.

대표:

- `__shfl_sync()`
- `__shfl_up_sync()`
- `__shfl_down_sync()`
- `__shfl_xor_sync()`

강의 요점:

- shared memory를 거치지 않아도 되고
- `__syncthreads()` 없이
- warp 내부에서 register 값 공유 가능

즉 warp 내부 reduction은 **shared memory보다 더 빠를 수 있다.**

---

## 12. Reduction with Warp Shuffle

### Slides 30-32

강의는 마지막 warp 단계에서 shuffle을 사용하는 개선안을 제시한다.

### warpReduce 함수

```c
__device__ __inline__ float warpReduce(float val) {
    float partialSum = val;
    for (unsigned int stride = WARP_SIZE / 2; stride > 0; stride /= 2) {
        partialSum += __shfl_down_sync(0xffffffff, partialSum, stride);
    }
    return partialSum;
}
```

### block reduction과 결합

```c
for (unsigned int stride = BLOCK_DIM / 2; stride > WARP_SIZE; stride /= 2) {
    if (threadIdx.x < stride) {
        input_s[threadIdx.x] += input_s[threadIdx.x + stride];
    }
    __syncthreads();
}

if (threadIdx.x < WARP_SIZE) {
    float partialSum = input_s[threadIdx.x]
                     + input_s[threadIdx.x + WARP_SIZE];
    partialSum = warpReduce(partialSum);
}
```

### 핵심 아이디어

- block reduction의 앞부분은 shared memory 사용
- 마지막 한 warp만 남으면 register + shuffle로 전환

즉:

- shared memory access 감소
- block-wide synchronization 감소
- 후반 단계 latency 감소

---

## 13. Warp-Level Programming with Cooperative Groups

### Slide 33: cooperative groups 소개

cooperative groups API는 thread group 단위 프로그래밍을 더 편하게 해 준다.

지원 단위:

- entire grid
- thread block cluster
- thread block
- block tile
  - multi-warp
  - single-warp
  - sub-warp

사용 예:

```c
#include <cooperative_groups.h>
using namespace cooperative_groups;

thread_block_tile<32> warp =
    tiled_partition<32>(this_thread_block());
```

### Slide 34: warp shuffle with cooperative groups

기존:

```c
__shfl_down_sync(0xffffffff, partialSum, stride)
```

cooperative groups 버전:

```c
warp.shfl_down(partialSum, stride)
```

즉 API가 더 읽기 쉬워지고,
group 개념이 코드에 명시적으로 드러난다.

### Slide 35: cooperative groups 기반 warpReduce

```c
__device__ __inline__ float warpReduce(float val) {
    thread_block_tile<32> warp = tiled_partition<32>(this_thread_block());
    float partialSum = val;
    for (unsigned int stride = warp.size() / 2; stride > 0; stride /= 2) {
        partialSum += warp.shfl_down(partialSum, stride);
    }
    return partialSum;
}
```

### 실전 포인트

cooperative groups는 성능 최적화 그 자체라기보다,

- warp/block 단위를 코드에 명확히 표현하고
- group 기반 synchronization / communication을 깔끔하게 쓰게 해 주는 도구

로 이해하면 좋다.

---

## 14. Two-Stage Warp Reduction

### Slides 36-39

강의는 한 단계 더 나아가,  
shared memory reduction + last warp shuffle보다 더 warp-centric한 구조를 소개한다.

### 핵심 아이디어

1. 각 warp가 자기 내부에서 먼저 register + shuffle로 reduction
2. warp별 partial sum만 shared memory에 기록
3. 첫 번째 warp가 warp partial sums를 다시 shuffle로 reduction

즉 block-wide reduction을:

- warp 내부 reduction
- warp 간 reduction

두 단계로 나누는 구조다.

### Slide 38: 코드

```c
unsigned int segment = 2 * blockDim.x * blockIdx.x;
unsigned int i = segment + threadIdx.x;

float partialSum = input[i] + input[i + BLOCK_DIM];
partialSum = warpReduce(partialSum);

__shared__ float partialSums_s[BLOCK_DIM / WARP_SIZE];
if (threadIdx.x % WARP_SIZE == 0) {
    partialSums_s[threadIdx.x / WARP_SIZE] = partialSum;
}
__syncthreads();

if (threadIdx.x < WARP_SIZE) {
    float partialSum = partialSums_s[threadIdx.x];
    partialSum = warpReduce(partialSum);
}
```

### 왜 좋은가

- block-wide synchronization 횟수 감소
- shared memory access 감소
- 대부분의 reduction을 register + warp shuffle로 처리

### Slide 39: trade-off

강의가 직접 말한 trade-off:

- **장점:** barrier synchronization과 shared memory access 감소
- **단점:** control divergence 증가 가능

즉 최적화는 늘 trade-off다.

좋은 kernel은:

- barrier를 줄이고
- memory access를 줄이면서도
- divergence와 코드 복잡도 증가를 감당할 수 있어야 한다

---

## 15. Thread Coarsening in Reduction

### Slide 40

강의는 reduction에서도 thread coarsening이 유효하다고 설명한다.

동기:

- reduction은 매 단계 synchronization 필요
- final steps에서 divergence 발생
- block 수가 너무 많아 hardware가 serialize하는 상황이면 낭비가 생길 수 있음

### Slides 41-42: before vs after coarsening

비교:

- non-coarsened:
  - 두 block이 순차적으로 serialize될 수 있음
- coarsened:
  - block 하나가 두 block의 일을 대신 수행

즉 hardware resource가 충분치 않아 block execution이 serialize될 때,
coarsening은 전체 overhead를 줄일 수 있다.

### Slide 43: coarsened reduction code

```c
unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
unsigned int i = segment + threadIdx.x;

float partialSum = 0.0f;
for (unsigned int tile = 0; tile < COARSE_FACTOR * 2; ++tile) {
    partialSum += input[i + tile * BLOCK_DIM];
}

partialSum = warpReduce(partialSum);

__shared__ float partialSums_s[BLOCK_DIM / WARP_SIZE];
if (threadIdx.x % WARP_SIZE == 0) {
    partialSums_s[threadIdx.x / WARP_SIZE] = partialSum;
}
__syncthreads();

if (threadIdx.x < WARP_SIZE) {
    float partialSum = partialSums_s[threadIdx.x];
    partialSum = warpReduce(partialSum);
}
```

### 해설

각 thread가:

- 입력 하나가 아니라 여러 tile의 값을 먼저 합산하고
- 그 뒤 reduction tree에 들어간다

즉:

- block 수 감소
- block 간 serialization 비용 감소
- synchronization overhead 상대적 감소

### Slide 44: coarsening benefit 분석

강의 비교:

- 원래 block당 원소 수를 `N = 2 * blockDim.x`라고 하자

#### 모든 block이 동시에 실행 가능할 때

- `log(N)` steps
- `log(N)` synchronizations

#### block이 hardware에 의해 `C`배 serialize될 때

- `C * log(N)` steps
- `C * log(N)` synchronizations

#### coarsening factor가 `C`일 때

- `2*(C-1) + log(N)` steps
- `log(N)` synchronizations

즉 block serialization이 문제일 때 coarsening이 유리할 수 있다.

---

## 16. Performance Intuition Across the Versions

15강의 reduction 커널 진화는 대략 이렇게 이해하면 된다.

### 1. Atomic version

- 가장 단순
- 성능 최악
- 전역 직렬화

### 2. Naive tree in global memory

- atomic보다 나음
- 하지만 coalescing 나쁨
- divergence 큼
- global memory traffic 큼

### 3. Better tree ordering

- coalescing 개선
- divergence 완화

### 4. Shared-memory reduction

- global memory access 대폭 감소
- block-wide sync는 여전히 존재

### 5. Last-warp shuffle

- 마지막 단계의 shared memory와 barrier overhead 감소

### 6. Two-stage warp reduction

- 더 많은 부분을 register + warp shuffle로 이동
- shared memory와 block-wide sync를 최소화

### 7. Coarsened reduction

- block 수가 너무 많아 serialize될 때 유리
- 한 thread가 더 많은 입력 처리

즉 reduction 최적화의 핵심은:

- **memory hierarchy 활용**
- **synchronization 줄이기**
- **divergence 줄이기**
- **hardware utilization 유지**

를 동시에 챙기는 것이다.

---

## 17. Lab 6 Connection

이번 강의는 곧바로 [README.md](/u/ylee21/ece408git/lab6/README.md)와 연결된다.

Lab 6 목표:

- 1D 리스트 reduction(sum) 구현
- lecture에서 다룬 improved kernel 구현
- arbitrary length input 처리

README의 핵심 문장:

- improved kernel discussed in the lecture 구현
- last block boundary는 identity value(`0`)로 채움
- block별 reduction sum은 host loop로 최종 합산

즉 Lab 6를 위한 정답 흐름은 거의 이 강의 전체와 동일하다.

1. block당 partial reduction
2. shared memory 사용
3. improved per-block mapping
4. warp-level optimization
5. host 또는 후속 단계에서 partial sums 재감소

---

## 18. 시험/프로젝트용 핵심 포인트

### 꼭 외워둘 개념

- reduction은 associative, commutative, identity를 가져야 병렬화가 자연스럽다
- atomic reduction은 correctness는 맞지만 성능이 매우 나쁘다
- reduction tree는 `log(N)` 단계, `N-1` 연산
- CUDA에서는 block 간 sync가 안 되므로 segmented reduction이 필요하다

### 꼭 설명할 수 있어야 할 것

- naive reduction mapping의 문제는 무엇인가
- coalescing을 어떻게 개선하는가
- shared memory reduction이 왜 빠른가
- 마지막 warp에서는 왜 `__syncthreads()`가 불필요한가
- `__shfl_down_sync()`는 무엇을 하는가
- cooperative groups는 왜 쓰는가
- two-stage warp reduction의 장점/단점은 무엇인가
- coarsening이 언제 유리한가

### reduction에서 자주 나오는 trade-off

- less synchronization vs more divergence
- fewer blocks vs less parallelism
- register use 증가 vs occupancy 감소 가능성

즉 최적 reduction은 항상 상황 의존적이다.

---

## 19. 빠른 복습용 한 페이지 요약

```text
Lecture 15 = reduction trees on GPU

reduction:
- sum, min, max, product
- associative
- commutative
- identity 필요

naive parallel:
- atomicAdd to one global sum
- correct but very slow

better idea:
- reduction tree
- N inputs -> log2(N) steps
- total work = N-1

CUDA issue:
- block 간 sync 불가
- segmented reduction 필요

naive block reduction 문제:
- uncoalesced access
- control divergence
- global memory traffic

개선 순서:
1. coalesced/reordered tree
2. shared memory reduction
3. last warp uses shuffle
4. cooperative groups version
5. two-stage warp reduction
6. thread coarsening

warp shuffle:
- __shfl_down_sync
- shared memory 없이 warp 내부 register 값 공유

coarsening:
- block 수를 줄임
- synchronization overhead와 block serialization 비용 완화

lab 연결:
- Lab 6 reduction kernel
```

