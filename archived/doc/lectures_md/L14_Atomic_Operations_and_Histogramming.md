# ECE 408 / CS 483 / CSE 408
## Lecture 14: Atomic Operations and Histogramming

**Instructor:** Volodymyr Kindratenko  
**학기:** Spring 2026

> Source: `Lectures_pdf/ece408-lecture14-histogram-vk-SP26.pdf`

---

## 1. Lecture Briefing

이번 강의는 두 가지를 연결해서 설명한다.

1. **atomic operation이 왜 필요한가**
2. **그 atomic operation이 실제 병렬 알고리즘, 특히 histogram에서 어떻게 쓰이는가**

더 정확히 말하면, 강의의 핵심은 다음 질문에 답하는 것이다.

- 여러 thread가 같은 메모리 위치를 동시에 갱신하면 무슨 일이 생기나?
- race condition을 막으려면 무엇이 필요한가?
- atomic은 correctness를 보장하지만 왜 느린가?
- histogram처럼 atomic이 자연스럽게 필요한 문제를 더 빠르게 구현하려면 어떻게 해야 하나?

### 이 강의의 큰 흐름

1. data race와 read-modify-write 문제
2. mutual exclusion, locks, 그리고 SIMD 환경에서의 한계
3. atomic operation의 개념과 CUDA intrinsics
4. histogramming 기본 알고리즘
5. atomic contention이 왜 throughput을 크게 떨어뜨리는가
6. shared memory와 privatization으로 contention 줄이기
7. thread coarsening으로 commit 비용 줄이기

### 이 강의의 핵심 요지

- **여러 thread가 같은 위치를 갱신하면 data race가 생길 수 있다.**
- **atomic operation은 correctness를 보장하지만 같은 주소에 대한 접근은 결국 직렬화된다.**
- **histogram은 atomic operation의 대표적 사용 예다.**
- **성능 향상의 핵심은 atomic을 없애는 게 아니라, contention을 줄이는 것이다.**
- **privatization과 coarsening이 histogram 최적화의 핵심 기법이다.**

---

## 2. Course Reminders and Objectives

### Slide 2: course reminders

강의 슬라이드 공지:

- Lab 5 이번 주 금요일 마감
- Labs 1-4 채점 확인
- Project milestone 1은 이번 주 금요일 마감
- Project milestone 2 곧 release
- Midterm 1 채점 중

### 날짜 주의

표지에는 `3/10/2025`가 보이지만, 파일명과 강의 맥락은 Spring 2026 자료다.  
이전 12강과 비슷하게, 슬라이드에 이전 학기 날짜가 일부 남아 있는 **연도 표기 오타 가능성**이 높다.

### Slide 3: today’s objectives

- atomic operations 이해
- parallel computation의 read-modify-write 문제 이해
- CUDA에서 atomic 사용 방법 학습
- atomic이 메모리 시스템 throughput을 왜 낮추는지 이해
- histogramming 기법 학습
- 기본 histogram 알고리즘
- atomic throughput
- privatization

---

## 3. Why Simple Shared Updates Fail

### Slide 4: collaboration pattern 예시

비유:

- 여러 은행원이 금고 안 돈을 pile별로 나눠 세고
- 중앙 display의 running total에 더한다

나쁜 결과:

- 일부 pile이 total에 반영되지 않을 수 있음

이 비유는 본질적으로 이런 상황을 말한다.

```text
shared_total += local_count
```

를 여러 사람이 동시에 수행할 때, 순서가 꼬이면 업데이트가 사라질 수 있다.

### Slide 5: arbitration pattern 예시

비행기 좌석 예약 비유:

- 여러 고객이 같은 seat map을 보고
- 자리를 선택한 뒤
- seat map을 업데이트

나쁜 결과:

- 여러 명이 같은 자리를 예약

이 역시 같은 패턴이다.

- "읽고"
- "판단하고"
- "업데이트하는"

과정이 분리되어 있으면, 그 사이에 다른 thread가 끼어들 수 있다.

---

## 4. Read-Modify-Write and Data Races

### Slide 6: read-modify-write

강의의 핵심 예시:

```c
oldVal = bins[b];
newVal = oldVal + 1;
bins[b] = newVal;
```

여기서 thread A와 thread B가 같은 `bins[b]`에 대해 동시에 수행하면:

- 둘 다 `0`을 읽고
- 둘 다 `1`을 쓰는 상황이 가능하다

초기값이 `0`인데 최종값이 `2`가 아니라 `1`이 되는 것이다.

### Slide 7: data race 정의

강의 정의:

- 여러 thread가 같은 memory location에 동시 접근
- ordering이 없음
- 그중 하나 이상이 write

이면 **data race**

결과:

- unpredictable output

### Slides 8-9: race example 타임라인

순서가 이렇게 되면 문제가 생긴다.

1. Thread A가 `bins[b]`를 읽음 -> `0`
2. Thread B도 `bins[b]`를 읽음 -> `0`
3. Thread A가 `1`을 계산해 저장
4. Thread B도 `1`을 계산해 저장

즉 두 번 증가했지만 결과는 한 번 증가한 것처럼 보인다.

### 핵심 정리

문제는 `++bins[b]`가 기계적으로 한 번에 일어나는 것이 아니라,

- read
- modify
- write

세 단계로 나뉘기 때문이다.

---

## 5. Mutual Exclusion, Locks, and Why Locks Are Bad on SIMD GPUs

### Slide 10: CPU식 mutual exclusion

CPU에서는 이런 식으로 lock을 쓸 수 있다.

```c
mutex_lock(lock);
++bins[b];
mutex_unlock(lock);
```

이렇게 하면 한 번에 한 thread만 critical section에 들어간다.

### Slide 11: mutual exclusion example

thread A가 lock을 잡으면,
thread B는 lock이 풀릴 때까지 기다려야 한다.

즉 correctness는 확보된다.

### Slide 12: locks and SIMD execution

하지만 GPU는 SIMD/SIMT 기반이라 문제가 생긴다.

예:

- thread 0은 lock 획득
- thread 1은 lock 대기

그런데 같은 warp 안에서 instruction이 lockstep에 가깝게 진행되면,

- 기다리는 thread 때문에
- lock을 가진 thread도 다음 단계로 못 나갈 수 있고
- deadlock 비슷한 문제가 생길 수 있다

### 결론

GPU에서는 일반적인 lock 사용이 부적절하거나 위험하다.  
그래서 하드웨어가 제공하는 **atomic instruction**을 사용한다.

---

## 6. Atomic Operations

### Slide 13: atomic operation의 개념

강의 정의:

- atomic operation은 read-modify-write를 **하나의 ISA instruction**처럼 수행
- 연산이 끝날 때까지 다른 thread가 그 location을 건드리지 못하도록 하드웨어가 보장
- 같은 주소에 대한 concurrent atomic operation은 하드웨어가 직렬화

즉 atomic은 correctness를 위한 하드웨어 수준 critical section이다.

### Slide 14: 언제 atomic이 필요한가

강의 포인트:

- 두 thread가 같은 location에 쓸 가능성이 있으면 atomic이 필요할 수 있다
- sharing은 생각보다 눈에 잘 안 띈다

예:

- hash table insert
- graph update
- bipartite graph 한쪽 노드 갱신

즉 programmer가 "독립적일 것"이라고 생각해도,
입력 데이터에 따라 실제로는 충돌할 수 있다.

또 중요한 점:

- atomicity는 **정합성**은 보장하지만
- **상대적 순서**까지 보장하지는 않는다

---

## 7. How Atomic Operations Are Implemented

### Slide 15: synchronization primitives

많은 ISA가 제공하는 atomic primitive 예:

- bit test and set
- compare and swap / exchange
- swap / exchange
- fetch and increment / add

### Slide 16: microarchitecture가 보장하는 것

atomic이 실행될 때 하드웨어는:

- 다른 thread가 같은 location을 접근하지 못하게 막고
- 다른 요청은 stall시키거나 queue에 넣는다

즉 같은 주소에 대한 atomic은 결국 **serial execution**

### Slide 17: CAS (Compare-And-Swap)

강의는 CAS를 이렇게 소개한다.

```c
bool atomicCAS(int *address, int old, int new) {
    if (*address != old)
        return false;
    *address = new;
    return true;
}
```

실제 의미:

- 현재 값이 예상값(`old`)과 같을 때만
- 새 값(`new`)으로 교체

CAS는 lock-free algorithm의 기본 building block이다.

### Slide 18: atomicAdd를 CAS로 구현하는 아이디어

강의 의사코드:

```c
int atomicAdd(int* address, int value){
    bool done = false;
    while (!done) {
        old_v = *address;
        done = atomicCAS(address, old_v, old_v + value);
    }
    return old_v;
}
```

즉:

- 값 읽기
- old 값 기준으로 CAS 시도
- 실패하면 다시 읽고 반복

실제 하드웨어에서는 `atomicAdd`가 보통 전용 instruction으로 제공되지만,
CAS가 atomic update를 구현하는 기본 원리를 보여 준다.

### Slide 19: CUDA atomic intrinsics

대표 함수:

```c
T atomicAdd(T* address, T val);
```

지원 타입:

- `int`
- `unsigned int`
- `float`
- `double`
- 기타

기능:

- `*address` 값을 읽고
- `val`을 더해 저장하고
- 원래 값을 반환

다른 atomic operation:

- `sub`
- `min`
- `max`
- `inc`
- `dec`
- `and`
- `or`
- `xor`
- `exchange`
- `compare and swap`

---

## 8. Histogramming: A Natural Atomic Use Case

### Slide 20: 코드 비교

정상 버전:

```c
unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < width * height) {
    unsigned char b = image[i];
    atomicAdd(&bins[b], 1);
}
```

문제 버전:

```c
++bins[b];
```

즉 histogram에서 여러 픽셀이 같은 bin으로 가면,
동시에 같은 위치를 증가시킬 수 있기 때문에 atomic이 필요하다.

### Slide 21: histogram이란

histogram은 데이터셋의 각 값을 대응되는 bin에 넣어 개수를 세는 방법이다.

응용:

- 이미지 feature extraction
- fraud detection
- astrophysics correlation
- 기타 통계 분석

기본 형태:

```text
for each input element x:
    bins[x]++
```

바로 이 `++`가 병렬화에서 문제의 핵심이다.

### Slide 22: 문자 빈도수 예시

강의는 문장:

> "Programming Massively Parallel Processors"

의 알파벳 histogram을 예로 든다.

예:

- `A(4)`
- `C(1)`
- `E(1)`
- `G(1)`

등

질문:

- 이걸 병렬로 어떻게 세나?

### Slides 23-27: 나이브 병렬 진행 예시

강의는 입력을 section으로 나눠 여러 thread가 처리하면서,
각 iteration마다 자기 문자를 해당 bin에 더하는 그림을 보여 준다.

핵심 메시지:

- 여러 thread가 같은 글자를 만나면
- 같은 bin을 동시에 갱신할 수 있고
- atomic operation이 correct update를 보장한다

---

## 9. A Better Histogram Kernel Mapping

### Slide 28: 더 나은 접근

강의는 첫 번째 방법의 문제를 지적한다.

- input reads가 coalesced되지 않음

개선:

- 각 thread에 strided pattern으로 input 할당
- adjacent threads가 adjacent input을 처리하게 함

즉:

- 읽기 측면에서는 coalescing을 확보
- 쓰기 측면에서는 여전히 atomic이 필요

### Slide 29: 다음 iteration

모든 thread가 다음 section으로 이동하면서 같은 패턴을 반복한다.

즉 이 방식은:

- memory access는 더 GPU 친화적
- 하지만 hot bin에 대한 contention은 여전히 존재

### Slide 30: histogram kernel

강의 코드:

```c
__global__ void histo_kernel(char *buf, long size, int *histo)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (i < size) {
        atomicAdd(&(histo[buf[i]]), 1);
        i += stride;
    }
}
```

### 코드 해설

- 첫 번째 처리 원소:

\[
i = blockIdx.x \cdot blockDim.x + threadIdx.x
\]

- stride는 grid 전체 thread 수

\[
stride = blockDim.x \cdot gridDim.x
\]

즉 thread들은:

- 첫 번째 iteration에 coalesced하게 한 구간을 읽고
- 그다음엔 `stride`만큼 건너뛰며 반복

이 패턴은 histogram뿐 아니라 많은 1D CUDA loop에서 자주 나온다.

---

## 10. Why Global-Memory Atomics Are Slow

### Slides 31-34: atomic operations on DRAM

강의는 global memory atomic의 느림을 여러 슬라이드에 걸쳐 설명한다.

핵심:

- atomic은 read로 시작
- modify 후
- write로 끝

각각 DRAM latency가 수백 cycle 수준일 수 있다.

그리고 중요한 점:

- 이 whole read-modify-write 동안
- 다른 thread는 같은 location을 건드릴 수 없다

즉 같은 주소에 대한 atomic은 사실상 queue를 서는 것이다.

### Slide 33: 직렬화

강의 문장:

- Each Load-Modify-Store has two full memory access delays
- All atomic operations on the same variable are serialized

즉 한 bin이 hot spot이면,
그 bin에 대한 atomic update들은 줄을 선다.

### Slide 34: high latency

global memory atomic이 느린 이유:

- read + write 둘 다 기다림
- contention이 있으면 추가 대기

즉 latency도 크고 throughput도 나빠진다.

---

## 11. Latency Determines Atomic Throughput

### Slide 35

강의 정의:

- atomic throughput = 특정 location에 대해 atomic operation을 수행할 수 있는 속도

이 속도는 read-modify-write 전체 latency에 의해 제한된다.

강의 설명:

- global memory(= DRAM)에서 atomic latency는 보통 1000 cycle 이상
- 많은 thread가 같은 location에 atomic을 걸면
- memory bandwidth가 `< 1/1000` 수준으로 떨어질 수도 있음

### Slide 36: 슈퍼마켓 비유

강의 비유:

- 손님이 계산대에서 갑자기 물건을 가지러 다시 매장으로 가면
- 줄 전체 throughput이 확 떨어진다

atomic도 마찬가지다.

- operation 하나가 끝나기 오래 걸리면
- 뒤에 있는 operation들이 모두 기다려야 한다

### 핵심 요약

atomic operation의 병목은 단순히 "연산 하나가 느리다"가 아니라,

- **같은 location에 대한 contention**
- **직렬화**
- **긴 latency**

의 조합이다.

---

## 12. Hardware Improvements: L2 and Shared-Memory Atomics

### Slide 37: L2 cache atomics

강의 포인트:

- L2 cache에서 atomic이 처리되면
- global DRAM atomic보다 latency가 줄어든다
- 하지만 여전히 같은 location에 대한 접근은 serialized
- block 전체가 아니라 grid 전체에서 공유되는 위치에 대해 "free improvement"가 될 수 있음

### Slide 38: shared memory atomics

shared memory atomic 특징:

- latency가 매우 짧다
- 하지만 여전히 serialized
- block private
- programmer가 알고리즘을 바꿔야 활용 가능

즉 shared memory atomic은 공짜가 아니라,
**privatization 같은 구조적 최적화와 함께 써야 한다.**

---

## 13. Privatization

### Slide 39: privatizing the histogram

아이디어:

- 각 block이 histogram의 private copy를 따로 가짐
- block 내부에서는 private copy를 갱신
- 마지막에 public copy(global histogram)로 합침

즉 hot global bin 하나를 모든 thread가 두드리는 대신,

- block 수만큼 나눠서 private histogram을 만들고
- 마지막에 block 단위로 합친다

### Slide 40: privatization 정의

강의 정의:

- multiple private copies of an output을 유지
- 완료 후 public copy 업데이트

필요 조건:

- operation이 **associative**
- operation이 **commutative**

histogram의 add는 이 조건을 만족한다.

### 직관

원래 방식:

```text
모든 thread -> 하나의 public histogram
```

privatization:

```text
block 0 -> private histogram 0
block 1 -> private histogram 1
...
마지막에 public histogram으로 merge
```

### 장점

- global contention 크게 감소

### 단점 / 제약

- private copy를 저장할 공간이 필요
- histogram size가 너무 크면 shared memory에 안 들어갈 수 있음

---

## 14. Shared-Memory Privatized Histogram

### Slide 41: private histogram 생성

강의 코드:

```c
__global__ void histo_kernel(unsigned char *buffer,
long size, unsigned int *histo)
{
    __shared__ unsigned int histo_private[256];

    // warning: this will not work correctly if there are fewer than 256 threads!
    if (threadIdx.x < 256)
        histo_private[threadIdx.x] = 0;
    __syncthreads();
```

### 해설

- bin 수가 256이라고 가정
- shared memory에 block-private histogram 생성
- block 시작 시 0으로 초기화
- 모든 thread가 초기화 완료할 때까지 `__syncthreads()`

강의의 warning이 중요한 이유:

- 이 코드는 thread 수가 256보다 적으면 모든 bin을 초기화하지 못한다
- 실전에서는 loop를 돌며 초기화하는 편이 더 안전하다

예:

```c
for (int t = threadIdx.x; t < 256; t += blockDim.x) {
    histo_private[t] = 0;
}
__syncthreads();
```

### Slide 42: private histogram 구축

강의 코드:

```c
int i = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;
while (i < size) {
    atomicAdd(&(private_histo[buffer[i]]), 1);
    i += stride;
}
```

핵심:

- atomic은 여전히 필요하다
- 하지만 target은 global histogram이 아니라 **shared-memory private histogram**

즉 contention 범위가:

- 전체 grid -> block 내부

로 줄어든다.

### Slide 43: final histogram 만들기

강의 코드:

```c
__syncthreads();
if (threadIdx.x < 256)
    atomicAdd(&(histo[threadIdx.x]),
              private_histo[threadIdx.x]);
}
```

이 단계는:

- block 내부 계산이 끝난 뒤
- private histogram의 각 bin을
- global histogram에 한 번씩 더하는 단계다

### 왜 빨라지나

원래는 input 원소 수만큼 global atomic이 필요했다.  
privatization 후에는:

- block 내부 업데이트는 shared-memory atomic
- global atomic은 block당 최대 256개 수준

즉 global contention이 크게 줄어든다.

---

## 15. More on Privatization

### Slide 44

강의 요약:

- privatization은 강력하고 자주 쓰이는 기법
- operation은 associative + commutative여야 함
- histogram add는 가능
- histogram size가 작아 shared memory에 들어가야 유리

질문:

- histogram이 너무 크면 어떻게 하나?

가능한 생각:

- 일부 bin만 privatize
- 여러 pass로 나누기
- block-private를 global memory에 두기
- hierarchical reduction

즉 privatization은 매우 강력하지만, problem size와 memory capacity에 제약이 있다.

---

## 16. Other Optimization Opportunities

### Slide 45

강의는 histogram 최적화를 checklist처럼 정리한다.

#### 1. 입력 메모리 access는 coalesced인가?

- Yes

input 읽기는 적절히 배치하면 coalescing 가능하다.

#### 2. control divergence가 큰가?

- 거의 없음
- boundary check 정도만 존재

#### 3. data reuse가 있는가?

- input data: 거의 없음
- output data: 있음
  - 여러 thread가 같은 bin을 갱신

즉 histogram은 input reuse보다 **output reuse/충돌**이 핵심이다.

#### 4. private histogram은 어디에 둘까?

- 가능하면 shared memory

#### 5. parallelization overhead는 없는가?

- private copy에서 public copy로 옮기는 atomic update가 block마다 한 번씩 생김

그래서 강의는 다음 기법을 제안한다.

- **thread coarsening**
  - block 수를 줄이고
  - public copy로 commit하는 횟수도 줄인다

---

## 17. Thread Coarsening for Histogram

### Slides 46-49

강의는 coarsening 전후 그림으로 설명한다.

### before coarsening

- block이 더 작은 input segment를 맡음
- block 수가 많음
- private histogram 수도 많음
- 최종 public histogram으로 commit할 횟수도 많음

### after coarsening

- block이 더 큰 input segment를 맡음
- block 수가 줄어듦
- private histogram 수가 줄어듦
- public histogram commit 횟수도 줄어듦

### coarsening 전략의 핵심 질문

> 한 thread에 여러 입력을 어떻게 배정할까?

강의는 두 가지 관점을 보여 준다.

#### 나쁜 coarsening

- 첫 iteration은 괜찮아 보여도
- 다음 iteration 접근이 비연속적이면 coalescing이 깨질 수 있음

#### 좋은 coarsening

- 각 iteration에서 adjacent thread가 adjacent input을 읽게 설계
- 즉 여러 입력을 맡더라도 coalesced access를 유지

### 요약

thread coarsening은 histogram에서도 유효하지만,

- commit 횟수를 줄이는 효과와
- input load coalescing 유지

를 동시에 고려해야 한다.

---

## 18. Lab 5 Connection

이 강의는 곧바로 [README.md](/u/ylee21/ece408git/lab5/README.md)와 연결된다.

Lab 5 목표는 histogram equalization 구현인데, 그 중 핵심 단계 하나가:

- grayscale image의 histogram 계산

README의 관련 단계:

```text
Compute the histogram of grayImage
histogram[grayImage[ii]]++
```

이 부분이 바로 이번 강의 내용이다.

즉 Lab 5를 할 때 필요한 사고 흐름은:

1. grayscale 픽셀의 밝기값으로 bin 선택
2. 여러 픽셀이 같은 밝기값을 가질 수 있음
3. 따라서 같은 bin으로 동시 update 가능
4. atomic이 필요
5. 성능을 위해 privatization 고려 가능

즉 14강은 단순 이론이 아니라 Lab 5를 위한 직접적인 준비 강의다.

---

## 19. Problem Solving from the Lecture

### Slide 51: 문제

**Q.**  
어떤 프로세서가 L2 cache에서 atomic operation을 지원한다고 하자.

- L2 atomic latency: `5 ns`
- DRAM atomic latency: `120 ns`
- kernel은 atomic operation당 `20` floating-point operations 수행
- floating-point operation 하나는 `1 ns`
- 각 thread는 atomic `5회`, floating-point operation `100회` 수행
- 이들 연산 시간은 서로 overlap되지 않음
- kernel의 floating-point throughput은 `0.2424 GFLOPS`

질문:

- atomic operation 중 몇 %가 L2 cache에서 일어났는가?

정답:

- **50%**

### Slide 52: 풀이 설명

#### 1. 전부 L2에서 atomic이 일어나는 경우

thread당 시간:

\[
5 \cdot 5ns + 100 \cdot 1ns = 25ns + 100ns = 125ns
\]

#### 2. 전부 DRAM에서 atomic이 일어나는 경우

\[
5 \cdot 120ns + 100 \cdot 1ns = 600ns + 100ns = 700ns
\]

#### 3. DRAM 비율을 `x`라고 두기

그러면 L2 비율은 `1-x`

총 시간:

\[
(1-x)\cdot125 + x\cdot700
\]

#### 4. throughput에서 thread당 시간 구하기

`0.2424 GFLOPS = 242.4 million FLOPs/s`

thread당 100 FLOPs 수행하므로:

\[
2.424 \text{ million threads/s}
\]

thread당 시간:

\[
1 / 2.424 \text{ million}
= 412.5ns
\]

#### 5. 식 세우기

\[
(1-x)\cdot125 + x\cdot700 = 412.5
\]

풀면:

\[
x = 0.5
\]

즉:

- DRAM atomic 50%
- L2 atomic 50%

### 이 문제의 핵심 교훈

- atomic latency가 성능에 직접 반영된다
- atomic이 더 낮은 메모리 계층에서 처리될수록 throughput이 좋아진다
- mixed behavior도 throughput 역산으로 추정할 수 있다

---

## 20. 시험/프로젝트용 핵심 포인트

### 꼭 이해해야 할 것

- `++bins[b]`는 병렬에서 안전하지 않다
- atomic은 read-modify-write를 원자적으로 수행한다
- 같은 주소에 대한 atomic은 직렬화된다
- correctness와 performance는 별개다

### 꼭 설명할 수 있어야 할 것

- data race가 왜 생기는가
- lock이 GPU에서 왜 위험한가
- atomicCAS와 atomicAdd의 관계
- histogram이 왜 atomic의 대표 예시인가
- global memory atomic이 왜 느린가
- privatization이 왜 contention을 줄이는가
- coarsening이 왜 final commit 횟수를 줄이는가

### 구현 관점

- input read는 coalesced하게
- private histogram은 가능하면 shared memory에
- final global merge는 block 단위로
- private histogram 초기화는 thread 수가 bin 수보다 적을 때도 안전하게

---

## 21. 빠른 복습용 한 페이지 요약

```text
Lecture 14 = atomic operations + histogramming

핵심 문제:
- 여러 thread가 같은 memory location을 동시에 update
- read-modify-write가 꼬이면 data race 발생

해결:
- atomic operations

하지만:
- 같은 address에 대한 atomic은 serialized
- global memory atomic은 latency가 매우 큼

histogram 기본형:
for each x:
    atomicAdd(&bins[x], 1)

최적화:
- input reads coalesced하게 배치
- shared memory private histogram 사용
- 마지막에 global histogram으로 merge
- thread coarsening으로 block 수/commit 수 감소

privatization 조건:
- 연산이 associative
- 연산이 commutative

핵심 병목:
- hot bins에 대한 contention
- global memory atomic latency

핵심 아이디어:
- atomic을 없애기보다
- atomic이 몰리는 범위를 줄인다
```

