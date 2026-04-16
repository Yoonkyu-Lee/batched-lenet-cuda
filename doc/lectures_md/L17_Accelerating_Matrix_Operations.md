# ECE 408 / CS 483 / CSE 408
## Lecture 17: Accelerating Matrix Operations

**Instructor:** S. Patel  
**학기:** Spring 2026

> Source: `Lectures_pdf/ece408-lecture17-Tensor-Operations-vk-SP26.pdf`

---

## 1. Lecture Briefing

이번 17강은 matrix operation, 특히 **matrix multiplication acceleration**을 하드웨어 관점에서 더 깊게 보는 강의다.

16강이 project optimization toolbox를 넓게 보여줬다면,
17강은 그중에서도:

- tiled matmul이 하드웨어에서 실제로 어떻게 수행되는지
- warp 수준에서 연산 데이터가 어떻게 흐르는지
- 왜 tensor core 같은 specialized unit이 필요한지

를 직관적으로 설명한다.

강의는 deep learning, 특히 LLM 연산이 얼마나 큰 matmul workload를 만드는지 보여 준 뒤,
기본 tiled kernel에서 tensor-core style execution으로 사고를 확장한다.

### 이 강의의 핵심 요지

- **LLM과 attention 연산의 대부분은 결국 거대한 matrix operation이다.**
- **기본 tiled matmul도 warp 수준에서 보면 register file, multiplier, adder 사이의 dataflow 문제다.**
- **특정 데이터 타입과 연산 패턴에 맞게 하드웨어를 특화하면 throughput을 크게 높일 수 있다.**
- **Tensor Core는 그 특화의 대표 사례다.**
- **현대 GPU/AI hardware는 일반 SIMT만이 아니라 matrix-specialized execution까지 포함해 이해해야 한다.**

---

## 2. Why This Matters for Modern Models

### Slides 4-5

강의는 Llama 3.1 70B 예시를 통해 문제 규모를 보여 준다.

예시로 등장하는 matmul:

- Q, K, V projection
- down projection
- gate projection
- up projection

등은 모두 매우 큰 matrix multiplication이다.

특히 self-attention의 `Q` 생성 예시에서:

- sequence length = `8192`
- weight matrix = `8192 x 8192`

같은 연산이 등장하고,
강의는 이 연산 하나가:

\[
8192 \times 8192 \times 8192 \times 2
\]

규모의 엄청난 operation count를 만든다고 강조한다.

핵심 메시지:

- 현대 모델에서는 matmul이 거의 전부다
- 그래서 matmul acceleration을 이해하는 것이 곧 모델 acceleration을 이해하는 것이다

---

## 3. Revisiting the Tiled Matrix Multiplication Kernel

### Slide 6

강의는 familiar한 tiled matrix multiplication kernel을 다시 가져온다.

구조:

- `subTileM`, `subTileN`을 shared memory에 로드
- 각 thread는 하나의 `P[Row, Col]`를 담당
- tile loop를 돌며 multiply-accumulate

핵심 코드 구조:

1. shared memory tile 선언
2. block / thread index 계산
3. `Row`, `Col` 계산
4. tile loop
5. `subTileM`, `subTileN` collaborative load
6. `__syncthreads()`
7. inner `k` loop에서 `Pvalue += ...`
8. 다시 `__syncthreads()`

이건 이미 Lab 3에서 구현한 것과 같은 큰 구조다.

---

## 4. Operation-Level View: Per Thread

### Slide 7

강의는 high-level CUDA code를 instruction-like 관점으로 바꿔 본다.

thread 하나가 dot product를 계산할 때 반복되는 패턴:

- `subTileM[ty][k]`를 register로 load
- `subTileN[k][tx]`를 register로 load
- multiply-add 수행

즉 inner loop 한 단계는 사실상:

- 두 load
- 한 FMA / MAD

의 반복이다.

강의 메시지:

- source code 수준에서는 간단한 `for (k ...)`
- hardware 수준에서는 register file에서 multiplier / adder로 값이 흘러가는 구조

로 이해해야 한다.

---

## 5. Operation-Level View: Per Warp

### Slides 8-10

강의는 관점을 warp로 확장한다.

warp 수준에서는:

- 여러 thread가 register file에서 값을 읽고
- 여러 multiplier/adder lane이 동시에 일하며
- shared memory에서 읽어 온 tile 데이터를 각 lane에 분배

하는 구조가 된다.

여기서 중요한 포인트는:

- thread가 따로따로 계산하는 것처럼 보여도
- 실제 hardware는 warp-level로 묶여 동작한다는 점이다

강의는 4x4 tile 예시로,
공유 메모리의 M/N subtile이 register file로 어떻게 로드되는지 시각화한다.

### 더 효율적인 패턴

Slides 10-11의 메시지:

- shared memory에서 필요한 데이터를 warp가 더 구조적으로 읽어오면
- 더 적은 load로 tile 전체를 공급할 수 있다
- 즉 단순히 “shared memory를 쓴다”를 넘어서
- **warp-level data movement 자체를 최적화**해야 한다

이 흐름이 곧 tensor-core style hardware motivation으로 이어진다.

---

## 6. Hardware Specialization for Smaller Data Types

### Slides 11-12

강의는 optimized hardware 아이디어를 소개한다.

특히 16-bit type에 맞춰 hardware를 특화하면:

- register 폭 활용도가 커지고
- lane당 더 많은 multiplier를 둘 수 있고
- throughput이 크게 증가할 수 있다

슬라이드 핵심 메시지:

- 16-bit type에서는 한 lane에서 더 많은 multiply 작업을 처리 가능
- add는 보통 더 넓은 precision을 유지
- 결과적으로 max throughput이 크게 증가

즉 datatype choice는 단순 precision 문제가 아니라,
**hardware throughput을 결정하는 구조적 선택**이다.

---

## 7. Tensor Cores as Matrix-Specialized Hardware

### Slide 13 이후

강의는 NVIDIA Tensor Cores를:

- basic tensor operator
- matrix multiply + accumulate 전용 unit

으로 소개한다.

핵심 아이디어:

- 일반 SIMT lane이 scalar-ish한 연산을 많이 하는 대신
- tensor core는 작은 matrix block 단위로 연산을 처리
- 그래서 matrix-heavy workload에서 훨씬 높은 efficiency를 낼 수 있다

이 시점에서 16강과 연결되는 메시지는 분명하다.

- shared memory tiling은 여전히 중요
- 하지만 modern GPU는 그 위에 tensor-core path까지 제공
- 최고 성능을 내려면 이 specialized path를 활용해야 한다

---

## 8. Timeline Intuition: SIMT vs Tensor Core Compute

### Slide 20

슬라이드에는 timeline 비교가 나온다.

SIMT path:

- Load to SMEM
- SIMT Compute
- 다시 Load to SMEM
- 다시 Compute

Tensor Core path:

- load staging 패턴이 다르고
- compute 밀도가 더 높다

강의가 직접 강조하는 바는:

- 단순 load/compute 반복 구조보다
- 더 specialized한 compute block을 사용하면
- 같은 memory staging 위에서 훨씬 높은 compute density를 얻을 수 있다는 점이다

즉 tensor core는 “새 명령 하나” 수준이 아니라,
**전체 load/compute balance를 바꾸는 하드웨어 path**다.

---

## 9. Broader Hardware Context

### Slides 23-24

강의 후반에는 NVIDIA만이 아니라:

- Bill Dally
- Google TPU v1

등이 등장한다.

핵심 메시지:

- AI workload는 너무 커졌고
- general-purpose execution만으로는 비효율이 커서
- industry 전반이 matrix-specialized hardware를 만들고 있다는 점이다

즉 tensor core는 isolated trick이 아니라,
AI accelerator 전체 흐름 속에 있는 설계 철학이다.

---

## 10. Connection to Lecture 16

17강은 16강의 내용을 하드웨어 관점에서 더 강화한다.

16강이 말한 것:

- register tiling
- tensor core
- WMMA
- TF32

17강은 왜 그런 방향이 자연스러운지 설명한다.

정리하면:

- tiled matmul을 thread 수준에서 보면 load + multiply-add 반복
- warp 수준에서 보면 dataflow organization 문제
- hardware 수준에서 보면 matrix-specialized unit으로 바꾸는 것이 훨씬 효율적

즉 16강이 “어떻게 쓸까”였다면,
17강은 “왜 그런 hardware가 생겼나”에 가깝다.

---

## 11. Lab / Project Connection

### Lab 6와의 직접 연결

직접적으로는 Lab 6 reduction 자체를 설명하는 강의는 아니다.
하지만 lab6 전에 읽어두면 도움이 되는 이유가 있다.

- warp-level 사고방식
- thread-level code를 hardware-level execution으로 보는 관점
- throughput과 hardware utilization에 대한 감각

이 reduction 최적화에도 그대로 중요하기 때문이다.

### Project와의 연결

CNN / GPT project에는 훨씬 직접적이다.

- big GEMM bottleneck 이해
- datatype이 throughput에 미치는 영향 이해
- tensor core path의 필요성 이해
- warp-level matrix function을 학습해야 하는 이유 이해

특히 GPT milestone 2에서는 matmul-heavy path를 빠르게 만드는 데 매우 직접적으로 연결된다.

---

## 12. Practical Takeaways

이 강의에서 가져가야 할 실전 감각:

1. matmul은 딥러닝에서 거의 항상 중심 병목이다
2. thread/block mapping만 보는 것으로는 부족하고 warp-level dataflow도 봐야 한다
3. datatype은 성능을 크게 바꾼다
4. tensor core는 단순 옵션이 아니라 현대 GPU 성능의 핵심이다
5. AI hardware는 increasingly matrix-specialized direction으로 가고 있다

---

## 13. One-Line Summary

Lecture 17 = **why modern GPUs accelerate matrix math with specialized hardware**

- LLM workload는 거대한 matmul
- tiled SIMT kernel은 시작점
- warp-level dataflow 최적화가 중요
- lower-precision-friendly hardware가 throughput을 높임
- tensor core가 그 대표적 실현 방식이다

