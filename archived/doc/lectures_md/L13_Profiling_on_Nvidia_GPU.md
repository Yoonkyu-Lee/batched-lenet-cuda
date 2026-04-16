# ECE 408 / CS 483 / CSE 408
## Lecture 13: Profiling on Nvidia GPU

**Instructor:** Lincoln Lin  
**학기:** Spring 2026

> Source: `Lectures_pdf/ece408-lecture13-profiling-SP26.pdf`

---

## 1. Lecture Briefing

이번 강의는 CUDA 최적화를 "감"이 아니라 **측정 가능한 근거**로 진행하는 방법을 설명한다.  
핵심 메시지는 아주 단순하다.

- 코드를 빠르게 만들고 싶다면
- 먼저 **어디가 느린지**
- 왜 느린지
- 무엇이 병목인지
- 실제 하드웨어 수준에서 확인해야 한다

이 강의의 전체 흐름은 다음과 같다.

1. **profiling이 무엇인가**
2. **CUDA 개념과 실제 GPU 하드웨어 대응**
3. **profiling 전에 해야 할 준비**
4. **Nsight Systems (`nsys`)로 시스템 수준 분석**
5. **Nsight Compute (`ncu`)로 커널 수준 분석**
6. **Roofline, arithmetic intensity, occupancy, stall, bank conflict 해석**
7. **프로파일링 결과를 바탕으로 실제 개선 방향 찾기**

### 이 강의의 핵심 요지

- **최적화는 profiling 없이 하면 안 된다.**
- **`nsys`는 전체 프로그램 흐름을 보고, `ncu`는 특정 커널 내부를 본다.**
- **커널이 memory-bound인지 compute-bound인지 먼저 구분해야 한다.**
- **stall, occupancy, cache, bank conflict, uncoalesced access를 읽을 수 있어야 한다.**
- **좋은 최적화는 지표의 변화를 통해 설명 가능해야 한다.**

---

## 2. Resources and Outline

### Slide 2: 강의 자료

강의는 GitHub 리소스를 제공한다.

- profiling 예제 코드
- `nsys`, `ncu` 결과 파일
- Delta에서 실행하는 방법

현재 로컬 레포에도 관련 자료가 있다.

- [README.md](/u/ylee21/ece408git/Profiling-Lecture/README.md)

이 README 기준 사용 흐름:

```bash
bash build.sh
sbatch run.slurm
sbatch nsys.slurm
sbatch ncu.slurm
```

즉 강의 자료 자체가 "설명 + 재현 가능한 실험 세트"로 구성되어 있다.

### Slide 3: outline

강의 순서:

- What is profiling
- Basics of GPU hardware
- Prepare for Profiling
- Nsight System
- Nsight Compute

---

## 3. What Is Profiling?

### Slide 5

강의 정의:

- **profiling**은 실행 가능한 프로그램을 profiler 안에서 실행하여
- 성능 관련 metric을 수집하고 분석하는 행위다

- **profiler**는 실행 중 하드웨어/시스템 수준 metric을 추적하는 도구다

### Slide 6: 일반적인 프로그래밍 흐름

슬라이드에는 대략 이런 흐름이 나온다.

```text
문제 정의
-> 알고리즘 설계
-> 구현
-> 정답 검증
-> 프로파일링
-> 최종 제품
```

즉 profiling은 부가 작업이 아니라,
**제대로 된 최적화 루프의 한 단계**다.

### Slide 7: 왜 profiling이 필요한가

CUDA는 하드웨어 차이를 많이 숨겨 준다.

장점:

- 같은 CUDA 코드가 여러 GPU에서 돌아간다

하지만 현실:

- 실제 스케줄링 방식
- 캐시 구조
- 메모리 병목
- warp stall 원인

은 GPU 세대마다 다르다.

그래서 profiling은 CUDA가 숨겨 준 하드웨어 현실을 다시 드러내는 도구다.

### 한 줄 정리

정답이 맞는 커널과 빠른 커널은 다르다.  
profiling은 "왜 아직 느린지"를 보여 준다.

---

## 4. CUDA Concepts vs Real Hardware

### Slide 9: CUDA 개념

강의는 CUDA에서 보던 추상 개념을 다시 정리한다.

#### Memory 종류

- Global memory
- Shared memory
- Constant memory
- Texture / Surface memory
- Local memory
- Registers

#### Execution 단위

- Grid
- Block
- Warp
- Thread

### Slide 11: 실제 계산 스케줄링

- grid/kernel은 사용 가능한 GPU에 스케줄됨
- block은 사용 가능한 SM(Stream Multiprocessor)에 스케줄됨
- 32 threads 단위가 warp로 스케줄됨

즉 중요한 현실:

- **block은 SM에 간다**
- **warp는 scheduler가 고른다**

이 사실은 occupancy, stall, latency hiding을 이해할 때 핵심이다.

### Slide 12: 실제 메모리 저장 위치

강의 설명 요약:

- global memory -> device memory에 있고 L2/L1을 거칠 수 있음
- shared memory -> SM 당 on-chip 영역
- constant memory -> device memory + constant cache
- texture/surface memory -> device memory + texture cache
- local memory -> 사실상 device memory 기반, cache될 수 있음
- registers -> warp / SM 근처의 register file

### Slide 13: cache의 의미

cache는:

- 칩 위에 있는 작은 메모리
- 지연시간이 낮음
- 최근 접근 데이터(temporal locality)와 주변 데이터(spatial locality)를 기억

개념:

- cache hit: 캐시에 이미 있음
- cache miss: 상위 cache나 device memory에서 다시 가져와야 함

### 실전 해석

커널이 느릴 때 우리가 실제로 묻게 되는 질문은 이런 것이다.

- global memory access가 너무 많은가?
- L1/L2 hit rate가 낮은가?
- shared memory가 오히려 병목인가?
- register pressure 때문에 occupancy가 낮은가?

즉 profiling은 추상 CUDA 코드와 실제 하드웨어 반응을 연결해 준다.

---

## 5. Example Workload: GEMM Variants

### Slides 18-22

강의는 profiling 예제로 GEMM(matrix multiplication)을 사용한다.

비교 대상:

1. **Simple GEMM kernel**
2. **Tiled GEMM kernel**
3. **Joint Register-Shared Memory Tiled GEMM**
4. **cuBLAS GEMM**

### Slide 19: simple GEMM

문제 정의:

\[
A(M,K)\times B(K,N)=C(M,N)
\]

simple GEMM 특징:

- 각 thread가 `C`의 원소 하나 계산
- 필요한 `A`, `B` 값을 global memory에서 직접 읽음

즉:

- 구현은 단순
- 메모리 재사용은 약함

### Slide 20: tiled GEMM

개선 아이디어:

- block 전체가 `A`, `B` tile을 shared memory에 올림
- 각 thread는 shared memory에서 필요한 row/col 값을 읽음
- global memory traffic를 줄이고 data reuse를 늘림

### Slide 21: joint register-shared memory tiling

더 나아가:

- thread coarsening 사용
- thread 하나가 output 여러 개 계산
- 일부 tile은 register, 일부 tile은 shared memory에 저장

즉:

- 메모리 계층을 더 적극적으로 활용
- instruction / memory traffic / reuse를 더 세밀하게 제어

### Slide 22: cuBLAS

NVIDIA는 GEMM 같은 핵심 선형대수 연산을 위해 **초고성능 라이브러리 cuBLAS**를 제공한다.

강의 포인트:

- 직접 짠 커널과 cuBLAS를 비교해 보면,
- 얼마나 큰 최적화 여지가 있는지 감각을 얻을 수 있다

이 강의에서는 tensor core 버전이 아닌 GEMM과 비교한다.

---

## 6. Prepare for Profiling

### Slide 23: NVCC 컴파일 플래그

강의의 중요한 조언:

- `--profile/-pg`: `gprof`용, **사용하지 말 것**
- `--debug/-g`: host debug 정보, **profiling용 아님**
- `--device-debug/-G`: device debug 정보, **profiling용 아님**
- `--generate-line-info/-lineinfo`: **profiling할 때 사용**

### 왜 `-G`를 쓰면 안 되나

디버그 빌드는:

- 최적화가 꺼지거나 달라지고
- 생성되는 기계 코드가 달라질 수 있어서
- 실제 성능 분석을 왜곡한다

따라서 profiling은 보통:

- release에 가까운 빌드
- 대신 line mapping 정보만 포함

형태로 한다.

### Slide 24: correctness 먼저 검증

강의는 profiling 전에 반드시 correctness를 확인하라고 강조한다.

방법:

- known input / known output 테스트
- CPU 또는 reference GPU 구현과 비교

그리고 `compute-sanitizer` 사용:

- `memcheck`: 메모리 접근 오류 / leak
- `racecheck`: shared memory race / hazard
- `initcheck`: 초기화되지 않은 global memory 접근
- `synccheck`: thread synchronization hazard

### 핵심 원칙

틀린 커널을 profiling하는 건 의미가 없다.  
먼저 **맞는 커널**을 만든 뒤, 그 다음 **빠른 커널**로 간다.

### Slide 25: time measurement

시간 측정 관련 개념:

- `cudaStream_t`
  - CUDA 작업 큐
- `cudaEvent_t`
  - stream 위의 marker

활용:

- kernel 실행 시간 측정
- stream 간 동기화
- overlapping operation 분석

### 간단한 이벤트 타이밍 예시

```c
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
myKernel<<<grid, block>>>(...);
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms = 0.0f;
cudaEventElapsedTime(&ms, start, stop);
```

주의:

- CPU 측 wall-clock만 보면 비동기 실행 때문에 오해할 수 있다
- CUDA event를 쓰면 GPU timeline 기준 측정이 가능하다

---

## 7. First Performance Results: GEMM Comparison

### Slides 27-29: A40, L40s, B200 비교

강의는 여러 GPU에서 GEMM 성능 표를 보여 준다.

### A40 결과

| Kernel | Time | GFLOPS | Efficiency |
| --- | --- | --- | --- |
| Simple | 9.92 ms | 677 | 1.81% |
| Shared | 3.34 ms | 2,011 | 5.38% |
| Shared Improved | 2.29 ms | 2,935 | 7.85% |
| Joint | 1.90 ms | 3,545 | 9.48% |
| cuBLAS | 0.30 ms | 22,026 | 58.90% |

### L40s 결과

| Kernel | Time | GFLOPS | Efficiency |
| --- | --- | --- | --- |
| Simple | 3.47 ms | 1,937 | 2.96% |
| Shared | 1.92 ms | 3,498 | 5.35% |
| Shared Improved | 1.31 ms | 5,138 | 7.85% |
| Joint | 1.06 ms | 6,354 | 9.71% |
| cuBLAS | 0.18 ms | 36,631 | 55.98% |

### B200 결과

| Kernel | Time | GFLOPS | Efficiency |
| --- | --- | --- | --- |
| Simple | 4.42 ms | 1,520 | 3.21% |
| Shared | 1.35 ms | 4,980 | 10.53% |
| Shared Improved | 1.07 ms | 6,256 | 13.22% |
| Joint | 1.03 ms | 6,514 | 13.77% |
| cuBLAS | 0.17 ms | 39,907 | 84.35% |

### Slide 30: 좋은 질문들

강의가 던지는 질문:

- 왜 우리의 computation efficiency는 cuBLAS보다 낮은가?
- shared improved kernel은 무엇이 달라졌는가?
- 왜 B200에서는 효율이 더 높게 나오나?

### 이 표에서 읽어야 할 핵심

1. shared memory tiling은 확실히 효과가 있다
2. joint tiling은 더 좋다
3. 그래도 cuBLAS와는 큰 격차가 있다
4. 최신 GPU일수록 같은 커널도 더 좋아질 수 있지만, **단순히 하드웨어만 좋다고 끝은 아니다**

즉 profiling은 이 격차의 원인을 추적하는 단계로 이어진다.

---

## 8. Nsight Systems: System-Level Profiling

### Slide 32: `nsys`의 역할

`Nsight Systems`는 **시스템 전체 타임라인**을 보는 도구다.

무엇을 보나:

- CPU 활동
- GPU 활동
- CUDA API 호출
- memcpy
- kernel launch

목적:

- 실행 전체 그림을 보고 hotspot 찾기

강의의 2-stage workflow:

1. Delta에서 `nsys` 실행해 `.nsys-rep` 생성
2. 로컬 PC에서 `nsys-ui`로 열어 분석

### Slide 33: 주요 subcommand

- `profile`
  - 가장 자주 사용
  - 실행하면서 CPU/GPU/CUDA API 이벤트 수집
- `start`, `stop`
  - interactive profiling
- `launch`
  - 실행만 먼저 시작하고 나중에 profile 시작
- `cancel`, `shutdown`
  - 세션 취소/종료

### Slide 34: 유용한 flags

- `-o`, `--output`
- `-f`, `--force-overwrite`
- `-e`, `--env-var`
- `-y`, `--delay`
- `-p`, `--nvtx-capture`
- `--stats`

### `nsys`로 보는 대표 질문

- CPU가 kernel launch를 너무 느리게 하고 있나?
- memcpy와 kernel이 겹치지 않고 있나?
- 작은 kernel이 너무 많이 실행되나?
- GPU가 쉬는 시간이 긴가?
- 어느 phase가 전체 시간 대부분을 차지하나?

### 언제 `nsys`를 써야 하나

프로그램 전체가 느린데 이유를 모를 때 먼저 `nsys`를 쓴다.  
즉 "어디가 문제인지" 찾는 탐색용이다.

---

## 9. Nsight Compute: Kernel-Level Profiling

### Slide 39: `ncu`의 역할

`Nsight Compute`는 **특정 kernel 내부의 GPU metric**을 본다.

무엇을 보나:

- compute utilization
- memory workload
- cache behavior
- scheduler 상태
- warp stall
- occupancy
- instruction hotspot

즉 `ncu`는 "이 커널이 왜 느린가?"를 해부하는 도구다.

### Slide 41: 유용한 flags

- `-o`, `--export`
- `-f`, `--force-overwrite`
- `--profile-from-start`
- `--set`
- `--kernel-id`
- `-k`, `--kernel-name`
- `-s`, `--launch-skip`
- `-c`, `--launch-count`
- `--nvtx-include`

### Slide 42: selective profiling

강의 포인트:

- `--profile-from-start off`를 쓰면
- `cudaProfilerStart()` / `cudaProfilerEnd()`로 원하는 구간만 profile 가능

또 `--profile` 같은 플래그가 프로그램으로 전달되도록 구분해야 한다는 점도 언급한다.

### 언제 `ncu`를 써야 하나

`nsys`로 hotspot kernel을 찾은 뒤,
그 kernel 하나를 골라서 `ncu`로 들어간다.

즉 일반적인 workflow는:

```text
정확성 검증
-> 시간 측정
-> nsys로 전체 hotspot 찾기
-> ncu로 특정 kernel 내부 분석
-> 수정
-> 다시 측정
```

---

## 10. GPU SoL and Arithmetic Intensity

### Slide 43: GPU SoL

SoL = **Speed of Light**

강의 정의:

- SM(compute) SoL
- memory SoL

은 각 자원의 theoretical max 대비 달성 utilization을 의미한다.

하지만 주의:

- 이 값은 하드웨어 자원의 최대 활용률에 대한 지표일 뿐
- 커널이 전체적으로 얼마나 "유용한 일"을 많이 했는지를 직접 뜻하지는 않는다

즉 SM SoL이 높다고 무조건 좋은 커널은 아니다.  
비효율적인 연산으로 ALU를 바쁘게 만들 수도 있기 때문이다.

### Slide 44: arithmetic intensity

강의 정의:

\[
AI_{kernel}=\frac{FLOP_{total}}{Load_{global}}
\]

\[
AI_{peak}=\frac{FLOPS_{peak}}{Bandwidth_{global}}
\]

단위는 `op/byte`

### bound 분류

- **Compute-bound**
  - `AI_kernel > AI_peak`
  - 계산 성능이 병목

- **Memory-bound**
  - `AI_kernel < AI_peak`
  - global memory bandwidth가 병목

### 중요한 주의

같은 커널이라도 GPU가 바뀌면 `AI_peak`가 달라진다.  
따라서:

- 한 GPU에서는 compute-bound
- 다른 GPU에서는 memory-bound

일 수 있다.

### 간단한 예시

만약 커널이

- 1,000 FLOPs 수행
- 1,000 bytes의 global load

라면:

\[
AI_{kernel}=1 \text{ op/byte}
\]

이 값이 GPU의 `AI_peak`보다 훨씬 작다면 memory-bound일 가능성이 크다.

---

## 11. Roofline Model and Optimization Strategy

### Slide 46: 최적화 방향

강의는 memory-bound / compute-bound에 따라 최적화 전략이 달라진다고 설명한다.

### memory-bound kernel일 때

대표 전략 2개:

1. **global memory access 줄이기**
   - 불필요한 로드/스토어 감소
   - coalescing 개선
   - data reuse 증가
   - shared memory / registers 활용

2. **같은 데이터로 더 많은 계산하기**
   - 하나의 로드로 더 많은 FLOP를 수행
   - 즉 arithmetic intensity를 올리기

### compute-bound kernel일 때

할 수 있는 일은 상대적으로 적다.

1. 더 좋은 하드웨어 사용
2. effective operation 수 줄이기
3. 작은 데이터 타입 사용
   - 예: `fp16`은 `fp32`보다 FLOP throughput이 높을 수 있음

### 핵심 직관

memory-bound 커널은 "데이터를 가져오는 비용"이 문제다.  
compute-bound 커널은 "계산 자체의 비용"이 문제다.

그래서 무작정 shared memory를 넣거나 unrolling을 넣는 게 아니라,
먼저 어떤 bound인지부터 봐야 한다.

---

## 12. Reading Memory Workload Analysis

### Slides 47-49: Memory Chart

강의는 memory workload analysis chart를 통해 다음을 읽으라고 한다.

- 어떤 메모리 공간을 참조하는 instruction이 얼마나 많은가
- L1으로 몇 개의 request가 가는가
- cache 계층 사이에서 실제 얼마나 많은 데이터가 이동하는가

### 실전에서 보는 포인트

- global load/store가 과도한가?
- shared memory 요청이 너무 많은가?
- cache hit rate가 낮은가?
- 예상보다 device memory traffic가 큰가?

### Slide 54: L2 excessive

- uncoalesced global memory access의 신호

즉 warp의 32 thread가 예쁘게 붙은 주소를 읽지 못해
필요 이상의 memory transaction이 발생한다는 뜻이다.

### Slide 55: L1 excessive

- shared memory bank conflict의 신호

즉 shared memory를 쓴다고 무조건 빠른 게 아니라,
접근 패턴이 나쁘면 shared memory 내부에서도 serialization이 생긴다.

---

## 13. Scheduler Statistics, Warp State, and Occupancy

### Slide 50: scheduler statistics

여기서 강의가 읽으라고 하는 평균 지표들:

- scheduler에 실제 할당된 warp 수
  - 너무 낮으면 work 부족 가능
- ready warp 수
  - 낮으면 dependency가 많을 가능성
- 실제 issue된 warp 수
  - 하드웨어와 dependency 상태에 영향

### Slide 52: warp state statistics

핵심 지표:

- **warp cycles per issued instruction**
  - 두 instruction 사이 평균 지연

이 값이 높다면:

- latency가 크고
- 이를 숨기려면 더 많은 warp가 필요하다

주요 stall 종류:

- **Long Scoreboard**
  - local/global memory 기다림
- **Short Scoreboard**
  - shared memory 기다림
- **MIO throttle**
  - memory IO queue가 가득 참
- **Math pipe throttle**
  - 계산 유닛이 바빠서 기다림

### stall 해석 요령

#### Long Scoreboard가 크다

- global/local memory dependency가 큼
- coalescing, caching, tiling, prefetching 등을 의심

#### Short Scoreboard가 크다

- shared memory 접근이 느리거나 conflict 가능

#### MIO throttle가 크다

- memory request를 너무 많이 던지고 있음
- instruction 수를 줄이거나 묶을 필요가 있음

#### Math pipe throttle가 크다

- 연산 유닛이 이미 꽉 찼을 수 있음
- compute-bound 가능성 검토

### Slide 53: occupancy

강의가 구분하는 두 개념:

1. **Theoretical occupancy**
   - 하드웨어가 물리적으로 수용 가능한 warp 수

2. **Achieved occupancy**
   - 실제 실행 중 평균적으로 올라와 있던 warp 수

achieved occupancy가 낮을 수 있는 이유:

- block 간 workload 불균형
- block/warp 수 부족
- 마지막 wave가 GPU를 못 채움
- warp/resource 사용량이 너무 큼

### 중요한 오해 방지

occupancy가 높다고 항상 빠른 건 아니다.  
하지만 latency hiding이 필요한 커널에서는 너무 낮으면 문제가 된다.

즉 occupancy는:

- 목적 그 자체가 아니라
- 병목 해석을 위한 힌트다

---

## 14. Shared Memory Bank Conflicts

### Slide 56: bank conflict

강의 설명:

- shared memory는 32 banks로 나뉨
- 각 bank는 cycle당 한 address를 처리
- 서로 다른 bank를 접근하면 병렬성이 좋다
- 같은 bank를 여러 번 접근하면 conflict

### 직관 예시

warp의 여러 thread가 shared memory의 같은 bank를 향하면:

- 요청이 분할되어 처리되고
- 결과적으로 serialization 발생

즉 shared memory를 도입했는데도 빨라지지 않는 경우,
bank conflict가 원인일 수 있다.

### Slide 55와 연결

`L1 excessive` 경고가 보이면:

- shared memory bank conflict
- shared memory 접근 패턴
- shared memory 배열 레이아웃

을 점검해야 한다.

---

## 15. Instruction Hotspot and SASS-Level Clues

### Slide 57: instruction hotspot

`ncu`는 코드 라인별로 보여 준다.

- register 사용량
- stall
- instruction 수
- PTX / SASS 대응

구분:

- **PTX**: GPU 모델에 독립적인 고수준 assembly
- **SASS**: GPU 모델 종속 저수준 machine instruction

### Slide 58: 왜 이게 중요한가

같은 C/CUDA 코드 한 줄이라도 실제로는:

- 여러 instruction으로 풀릴 수 있고
- register dependency가 생길 수 있고
- data dependency 때문에 stall이 생길 수 있다

예시로 슬라이드는 이런 곱셈-누산 한 줄을 보여 준다.

```c
sum += A[targetRowIndex * K + i] * B[targetColumnIndex + i * N];
```

이 한 줄 안에도:

- 주소 계산
- load
- multiply
- add
- register dependency

가 모두 들어간다.

### 실전 의미

instruction hotspot은 "어느 줄이 뜨거운가?" 뿐 아니라  
"왜 이 줄이 뜨거운가?"까지 보게 해 준다.

---

## 16. Comparing Kernels with NCU

### Slide 59: baseline comparison

`ncu`는 baseline feature를 통해 여러 kernel metric을 비교할 수 있다.

예:

- simple vs shared
- shared vs shared improved
- joint vs cuBLAS

### Slide 61: roofline comparison

강의 그림은

- `Simple`
- `Shared`
- `Joint`
- `cuBLAS`

가 roofline 상에서 어디쯤 위치하는지 비교한다.

이 비교가 좋은 이유:

- 단순히 시간이 줄었다는 사실을 넘어서
- 메모리 병목에서 compute 활용 쪽으로 이동했는지
- arithmetic intensity가 개선됐는지

를 한눈에 볼 수 있기 때문이다.

### Slide 62: memory workload comparison

강의 설명 포인트:

- global access 감소
- shared access 증가
- uncoalesced memory access가 많으면 device memory load 증가
- hit rate 저하

즉 "shared memory를 썼다"는 사실만으로 좋은 것이 아니라,
**global traffic 감소 + 좋은 shared access 패턴**이 함께 있어야 한다.

---

## 17. Example: Improving the Shared Kernel

### Slide 70

강의는 shared kernel이 기대만큼 빠르지 않은 이유를 짚는다.

원인:

1. **Uncoalesced global memory loads**
2. **MIO queue thrashing**

개선 아이디어:

- 여러 개의 연속된 shared memory request를 하나로 합쳐
- `4개의 LDS` 대신 `1개의 LDS.128` SASS instruction을 만들게 함

### 이게 왜 중요한가

즉 단순히 shared memory를 쓴다고 끝이 아니라:

- instruction 개수
- memory instruction 폭
- MIO queue pressure

까지 봐야 한다는 뜻이다.

### 직관

```text
나쁜 버전:
4번 따로 load

좋은 버전:
한 번에 넓게 load
```

이 차이는:

- instruction 수 감소
- queue pressure 감소
- stall 감소

로 이어질 수 있다.

---

## 18. A Practical Profiling Workflow

이 강의를 실전 workflow로 정리하면 다음과 같다.

### 1. correctness 확보

- reference와 결과 비교
- `compute-sanitizer` 돌리기

### 2. 기본 시간 측정

- CUDA events로 측정
- 입력 크기와 반복 횟수 명확히 기록

### 3. `nsys`로 전체 그림 보기

- GPU가 쉬고 있는가?
- memcpy가 병목인가?
- launch overhead가 큰가?
- hotspot kernel이 무엇인가?

### 4. `ncu`로 hotspot kernel 깊게 보기

- memory-bound / compute-bound 판단
- roofline 확인
- occupancy / scheduler / stalls 확인
- cache / coalescing / bank conflict 확인
- instruction hotspot 확인

### 5. 가설 세우기

예:

- uncoalesced global access가 문제다
- shared memory bank conflict가 있다
- 너무 많은 memory instruction이 있다
- occupancy가 낮아 latency를 못 숨긴다

### 6. 수정 후 재측정

- 시간 측정
- `ncu` 비교
- 실제 병목이 줄었는지 확인

### 핵심 원칙

최적화는

- "뭔가 좋아 보이는 기법을 넣는 것"이 아니라
- "측정된 병목을 줄이는 것"이다

---

## 19. 시험/프로젝트용 핵심 체크포인트

### 꼭 구분할 것

- `nsys`: system-level timeline
- `ncu`: kernel-level metric

### 꼭 설명할 수 있어야 할 것

- arithmetic intensity란 무엇인가
- memory-bound와 compute-bound의 차이
- occupancy와 achieved occupancy 차이
- long scoreboard / short scoreboard / MIO throttle 의미
- L2 excessive가 무엇을 시사하는가
- L1 excessive가 무엇을 시사하는가
- bank conflict가 왜 생기는가

### 프로젝트에 바로 쓰이는 것

- profiling은 CNN/GPT 프로젝트 모두에 직접 연결됨
- 최적화 근거를 설명할 때 roofline, memory chart, stall, occupancy를 쓸 수 있어야 함

---

## 20. 강의 중 나온 생각해볼 질문

### Slide 30

- 왜 우리 커널의 computation efficiency가 cuBLAS보다 낮은가?
- shared improved kernel은 정확히 무엇이 달라졌는가?
- 왜 B200에서 효율이 더 높게 나오는가?

### Slide 77

- L40s(serve-oriented)와 B200(training-oriented)는 어떤 트레이드오프를 가졌는가?
- B200 vs 5090, A40 vs 3060 Ti를 보면 datacenter GPU와 consumer GPU의 근본 차이는 무엇인가?

이 질문들은 단순 암기보다 **하드웨어-성능-워크로드 관계를 스스로 해석할 수 있는지**를 묻는다.

---

## 21. 빠른 복습용 한 페이지 요약

```text
Lecture 13 = CUDA profiling 입문 + 실전 지표 읽기

핵심 도구:
- nsys: 프로그램 전체 타임라인
- ncu: 개별 커널 상세 분석

profiling 전:
- correctness 확인
- compute-sanitizer 사용
- -lineinfo로 빌드
- CUDA event로 시간 측정

핵심 개념:
- AI = FLOPs / global bytes
- AIkernel > AIpeak -> compute-bound
- AIkernel < AIpeak -> memory-bound

memory-bound 최적화:
- global memory access 줄이기
- coalescing 개선
- data reuse 늘리기
- 같은 데이터로 더 많은 연산 수행

compute-bound 최적화:
- effective op 수 줄이기
- 더 작은 타입 사용
- 더 좋은 하드웨어 활용

ncu에서 볼 것:
- GPU SoL
- roofline
- memory workload
- scheduler stats
- warp state stats
- occupancy
- L2 excessive
- L1 excessive
- instruction hotspot

경고 신호:
- Long Scoreboard -> global/local memory 대기
- Short Scoreboard -> shared memory 대기
- MIO throttle -> memory request 과다
- L2 excessive -> uncoalesced global access
- L1 excessive -> shared memory bank conflict
```

