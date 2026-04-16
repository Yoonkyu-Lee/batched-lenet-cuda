# ECE 408 / CS 483 / CSE 408
## Lecture 19: GPU Systems Architecture

**Instructor:** V. Kindratenko  
**학기:** Spring 2026

> Source: `lectures_pdf/ece408-lecture19-GPU-Systems-Architecture-vk-SP26.pdf`

---

## 1. Lecture Briefing

19강은 특정 CUDA kernel 알고리즘 자체보다, **GPU를 시스템 전체 안에서 어떻게 바라봐야 하는가**를 설명하는 강의다.

초점은 다음과 같다.

- GPU를 co-processor로 쓸 때 데이터 이동이 왜 중요한가
- CPU memory, PCIe, NVLink, GPU memory bandwidth를 어떻게 비교해야 하는가
- 왜 GPU 성능은 kernel만 잘 짠다고 끝나지 않는가
- multi-GPU와 modern GPU memory hierarchy를 어떻게 이해해야 하는가

Lab 7 구현을 직접 설명하는 강의는 아니지만,
Lab 7이 왜 memory-bound인지, 그리고 project에서 profiling을 왜 해야 하는지 이해하는 데 좋은 배경 강의다.

---

## 2. Canonical CUDA Program Structure Revisited

강의 초반은 익숙한 CUDA program skeleton을 다시 보여 준다.

핵심 구조:

1. device memory allocation
2. host-to-device copy
3. execution configuration setup
4. kernel launch
5. device-to-host copy
6. correctness check or follow-up work

이건 사실 거의 모든 lab의 host flow와 같다.

- Lab 1: vector add
- Lab 3: tiled matmul
- Lab 5: histogram equalization
- Lab 7: scan

강의가 이 구조를 다시 강조하는 이유는,
**성능 병목이 kernel 안에만 있지 않기 때문**이다.

---

## 3. Bandwidth as the Real Constraint

가장 중요한 메시지 중 하나는:

> modern GPU computing is governed by bandwidth

강의는 대략 다음 reference point를 비교한다.

- CPU main memory bandwidth
- PCIe bandwidth
- NVLink bandwidth
- GPU on-package memory bandwidth such as HBM

핵심 직관:

- GPU device memory bandwidth는 매우 크다
- 하지만 CPU-GPU 사이 PCIe는 상대적으로 훨씬 느리다
- 따라서 data transfer를 자주 하면 GPU compute speedup이 쉽게 묻힌다

이건 project와 profiling lecture까지 이어지는 중요한 메시지다.

---

## 4. Why PCIe Often Becomes the Bottleneck

강의는 CPU-GPU workload에서 PCIe가 종종 bottleneck이라고 짚는다.

이유:

- CPU 쪽 데이터가 GPU로 들어가야 함
- 결과가 다시 host로 돌아와야 할 수 있음
- GPU 내부 연산은 매우 빠르더라도
- transfer cost가 크면 전체 app-level speedup이 작아질 수 있음

즉,

- kernel optimization
- memory optimization
- transfer minimization

을 따로 보는 게 아니라 같이 봐야 한다.

이는 왜 `nsys` 같은 timeline profiler가 중요한지도 설명해 준다.

---

## 5. Historical and System Context

강의는 old PC architecture, PCI bus, memory-mapped I/O 같은 내용을 통해
GPU가 시스템에 어떻게 연결되어 왔는지를 짚는다.

이 부분의 목적은 역사 설명 자체보다,

- GPU가 독립된 섬이 아니라 시스템 bus와 interconnect 위에 있는 장치
- device access는 시스템 구조 제약을 받음

이라는 점을 이해시키는 데 있다.

즉 “kernel이 빠르다”와 “application이 빠르다”는 다른 문제라는 것이다.

---

## 6. Modern GPU Communication Paths

강의 objective에서 직접 언급하는 중요한 비교는 다음이다.

- PCIe vs NVLink
- HBM vs GDDR

### PCIe vs NVLink

- PCIe는 CPU-GPU, 혹은 일반 시스템 interconnect 관점에서 흔한 경로
- NVLink는 GPU-GPU 혹은 CPU-GPU 고대역폭 연결을 위한 더 특화된 경로

핵심 메시지:

- multi-GPU scaling은 compute capability만으로 결정되지 않는다
- interconnect bandwidth와 topology가 매우 중요하다

### HBM vs GDDR

- HBM은 높은 bandwidth를 제공
- modern accelerator는 이 bandwidth를 활용해 큰 matrix / tensor workload를 처리

즉 memory technology 자체도 performance architecture의 일부다.

---

## 7. Why This Matters for Labs

Lab 7 자체는 prefix sum 구현 lab이므로,
직접적인 알고리즘 reference는 18강이 담당한다.

하지만 19강은 다음 관점에서 보조적이다.

- scan이 memory-bound algorithm이라는 점을 시스템 관점에서 이해
- host-device copy 비용을 과소평가하지 않기
- block-level kernel만이 아니라 application-level timing도 보기
- profiling 결과에서 memcpy와 kernel time을 분리해 해석하기

즉 19강은 Lab 7을 “어떻게 짤까”보다,
“왜 이게 빨라지거나 안 빨라질까”를 생각하게 해주는 강의다.

---

## 8. Why This Matters for the Project

19강의 진짜 힘은 project 문맥에서 더 크게 보인다.

- Milestone 2에서 `nsys`, `ncu`를 쓰는 이유
- app-level timeline에서 memcpy와 kernel을 같이 봐야 하는 이유
- fused kernel이 왜 launch 수와 memory traffic을 줄이는지

이런 질문들이 전부 19강의 시각과 연결된다.

예를 들어:

- unroll pipeline은 kernel이 여러 개고 intermediate global memory traffic이 있음
- fused kernel은 kernel launch 수와 global-memory round trip을 줄임

이건 단순 CUDA syntax 문제가 아니라,
**system-level data movement 문제**다.

---

## 9. Practical Takeaways

1. GPU optimization은 kernel alone 문제가 아니다.
2. CPU-GPU transfer bandwidth는 종종 전체 성능의 ceiling이 된다.
3. PCIe, NVLink, device memory bandwidth를 함께 봐야 현실적인 성능 해석이 가능하다.
4. profiling은 kernel time뿐 아니라 memcpy와 launch overhead도 같이 봐야 한다.
5. modern GPU programming은 algorithm, kernel, memory, system architecture가 함께 맞물린다.

---

## 10. Connection to Lab 7 and Beyond

이번 강의를 Lab 7에 바로 연결하면 다음처럼 생각할 수 있다.

- 18강:
  - scan algorithm 자체
  - segmented scan
  - Brent-Kung
  - hierarchical scan
- 19강:
  - scan이 돌아가는 시스템의 bandwidth reality
  - host/device transfer와 memory behavior 해석

즉 Lab 7은 18강이 “what to implement”를 주고,
19강이 “how to think about performance context”를 준다고 볼 수 있다.
