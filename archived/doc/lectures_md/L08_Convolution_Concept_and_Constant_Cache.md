# ECE 408 / CS 483 / CSE 408
## Lecture 8: Convolution Concept; Constant Cache

**Instructor:** Volodymyr Kindratenko  
**학기:** Spring 2026

---

## 1. Course Reminders

### Lab updates

- Lab 3 is due this week on Friday
- Project README files are released, and so is the signup form for GPT-2 project

### Midterm 1

- **When:** March 3rd, 7–10pm
- **Where:** Your specific room assignment will be posted in Canvas
- **What:** Lectures 1–12, Labs 1–4
- **How:** paper-based
- **Alternative Exam Time:** as arranged; email instructor by 2/24/26 if you have a valid conflict
- **Study materials:** will be posted on Canvas

---

## 2. Today's Objectives

- Convolution: 중요한 병렬 연산 패턴 학습
- 신호·이미지·비디오 처리에 널리 사용
- 많은 과학/공학 응용에서 쓰이는 stencil 연산의 기초
- Neural Networks 및 Deep Learning의 핵심 구성 요소
- **Important GPU technique:** 캐시 메모리 활용

---

## 3. Convolution Applications

- 신호 처리, 디지털 녹음, 이미지/비디오 처리, 컴퓨터 비전, 머신러닝 등 다양한 형태로 사용
- Convolution은 흔히 **filter**로 수행되며, 입력 신호(오디오, 비디오 등)를 context-aware하게 변환
- 일부 필터는 신호를 부드럽게 해서 큰 그림의 추세를 보는 데 사용

---

## 4. Convolution Mathematics

- (Figure / equation on slide)

---

## 5. Convolution vs. Cross-Correlation

### Convolution

- \((f*g)(t) = \int f(\tau)\, g(t-\tau)\, d\tau\)
- Kernel is **flipped**
- Main use: physics, filtering

### Cross-correlation

- \((f \star g)(t) = \int f(\tau)\, g(t+\tau)\, d\tau\)
- **No** flipping of the kernel
- Main use: pattern matching, ML

### Convolutional Neural Networks (CNNs)

- CNNs use **cross-correlation**, not true convolution
- Whether the kernel is flipped or not does not matter — the network learns the weights
- Implementing convolution without flipping (i.e., cross-correlation) is simpler and computationally cleaner
- Historically the term “convolution” stuck even though CNNs use cross-correlation

---

## 6. Convolution Computation

- 각 **출력 원소**는 인접 **입력 원소들**의 **가중 합**
- 가중치는 **convolution kernel**(mask array)로 정의; 강의에서는 mask array를 convolution mask / convolution filter라고 부름
- 동일한 convolution mask가 보통 배열의 모든 원소에 사용됨

---

## 7. 1D Convolution Example

- 오디오 처리 등에 흔히 사용
- **MASK_WIDTH**는 대칭을 위해 보통 홀수 (예: 5)
- **MASK_RADIUS**는 계산 대상 픽셀 양쪽으로 쓰는 원소 수 (예: 2)
- **P[2] 계산:** (Slide 8 다이어그램)
- **P[3] 계산:** (Slide 9 다이어그램)
- 입력 배열 경계 근처 출력 원소는 “ghost” 원소 처리 필요
- 정책: 0, 경계 값 복제 등

---

## 8. 1D Convolution Kernel with Boundary Handling

- 경계 밖 원소를 0으로 만드는 커널 예:

```c
__global__ void
convolution_1D_basic_kernel(float *N, float *M, float *P, int Mask_Width, int Width)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    float Pvalue = 0;
    int N_start_point = i - (Mask_Width/2);
    for (int j = 0; j < Mask_Width; j++) {
        if (((N_start_point + j) >= 0) && ((N_start_point + j) < Width)) {
            Pvalue += N[N_start_point + j] * M[j];
        }
    }
    P[i] = Pvalue;
}
```

---

## 9. 2D Convolution

- (Slide 12: 2D convolution N, M, P 예)
- **2D Convolution Boundary Condition:** 경계 밖은 0 등 (Slide 13)
- **Ghost cells (apron/halo cells):** (Slide 14)

---

## 10. Example: What Does This Kernel Accomplish?

- (Slide 15: M = 1/(2²⁵) × [1 4 7 4 1; …] — Hint: Assume input N is an image)

---

## 11. Access Pattern for M

- M의 원소: mask (kernel, filter) **coefficients (weights)**
- 모든 출력 P 원소 계산에 M 필요
- M은 grid 실행 동안 **변하지 않음**
- 보너스: M 원소는 모든 P 원소를 계산할 때 **같은 순서**로 접근
- ➡ M은 **Constant Memory**에 두기에 적합

---

## 12. Programmer View of CUDA Memories

- 각 thread:
  - Read/write **per-thread** registers (~1 cycle)
  - Read/write **per-block** shared memory (~5 cycles)
  - Read/write **per-grid** global memory (~500 cycles)
  - **Read-only** **per-grid** constant memory (~5 cycles with caching)

- (Figure: Grid → Block → Thread, Global / Shared/L1 / Registers, Host, Constant Memory)

---

## 13. Memory Hierarchies

- Global memory만 계속 접근하면 GPU 실행 속도는 global memory bandwidth에 제한됨
- Tiled matrix multiplication에서 **shared memory**로 이 한계를 완화함
- 또 다른 해결: **Caches**

---

## 14. Cache

- Cache = cache line들의 “배열”
- Cache line 하나는 보통 **여러 연속 메모리 주소**의 데이터를 담음
- Global memory에서 데이터를 요청하면, 접근한 데이터를 포함하는 **전체 cache line**이 cache에 로드됨
- Cache 안 데이터는 global memory 원본의 **복사**
- 추가 하드웨어가 cache line 데이터의 **주소**를 기억

---

## 15. Caches Store Lines of Memory

- Memory burst: 연속(linear) 주소에서 약 1024 bits (128B) — 이를 한 **line**이라 부름
- **Cache:** cache line(및 tag)들의 배열
- Memory read가 한 line을 만들면, cache가 그 line의 복사본을 저장하고, tag가 그 line의 메모리 주소를 기록

---

## 16. Caches and Locality

- **Spatial locality:** 연속 메모리 위치에 있는 데이터 원소를 연속적으로 접근
- **Temporal locality:** 같은 데이터 원소를 짧은 시간에 여러 번 접근
- 두 locality 모두 cache 성능을 좋게 함

---

## 17. Memory Accesses Show Locality

- 실행 중인 프로그램이 메모리에서 load/store
- 접근 주소 시퀀스는 보통 두 종류의 locality를 보임:
  - **Spatial:** X 접근 시 곧 X+1 (그리고 X+2, …) 접근
  - **Temporal:** X 접근 시 곧 다시 X 접근
- (Caches improve performance for both.)

---

## 18. Caches Can't Hold Everything

- Cache는 메모리보다 작음
- Cache가 가득 차면 새 line을 위해 자리를 만들어야 함 — 보통 **least recently used** line 제거

---

## 19. Shared Memory vs. Cache

- Shared memory: main memory contention 완화를 위한 또 다른 임시 저장소
- SM과의 거리 측면에서 shared memory는 L1 cache와 비슷
- Cache와 달리 shared memory는 반드시 main memory에 있는 데이터의 복사본을 들고 있을 필요 없음
- Shared memory는 **명시적 데이터 전송**으로 채워야 하고, cache는 그렇지 않음

### Caches vs. shared memory

- 둘 다 on-chip*, 비슷한 성능
- (Volta 세대부터는 같은 물리 자원을 쓰며 동적으로 할당)
- **차이:** Programmer가 shared memory 내용을 제어(scratchpad); Cache 내용은 마이크로아키텍처가 자동 결정  
  *Static RAM, not DRAM — ECE120/CS233 참고

---

## 20. Constant Cache in GPUs

- Cached 데이터 수정 시 원본 global memory에 반영해야 함 → 수정 여부 추적 등 필요
- **Constant cache:** grid 실행 동안 **수정되지 않는** constant 데이터를 위한 전용 cache
- Constant memory에 선언된 데이터는 kernel 실행 중 수정되지 않음
- Constant cache는 일부 흔한 패턴에서 L1 cache보다 **높은 throughput**으로 접근 가능

---

## 21. GPU Has Constant and L1 Caches

- 쓰기(라인 수정)를 지원하려면 변경 사항을 메모리로 복사하고, cache가 수정 상태를 추적해야 함
- GPU의 L1 cache (global memory 접근용)는 쓰기 지원
- **Constant / texture memory용 cache:** 읽기 전용이라 특수 처리 → 흔한 GPU kernel 접근 패턴에서 L1보다 높은 throughput 가능

---

## 22. GPU L2/L1 Caches

- (Slide 28: V100, A100 등)

---

## 23. Using Constant Memory

- Constant memory 배열은 **global 변수**로 선언 (CUDA kernel 및 host 함수 밖)

```c
__constant__ float filter_c[FILTER_DIM];
```

- **Host에서 초기화 필수:** 실행 중 수정 불가

```c
cudaMemcpyToSymbol(filter_c, filter, FILTER_DIM * sizeof(float),
                   offset = 0, kind = cudaMemcpyHostToDevice);
```

- 최대 **64KB**만 할당 가능
- 입력이 constant여도 크기가 64KB를 넘으면 constant memory에 넣지 못함

---

## 24. Host Code Example

```c
// MASK_WIDTH is the size of the mask
// global variable, outside any kernel/function
__constant__ float Mc[MASK_WIDTH];

// Initialize Mask
float Mask[MASK_WIDTH];
for (unsigned int i = 0; i < MASK_WIDTH; i++) {
    Mask[i] = (rand() / (float)RAND_MAX);
    if (rand() % 2) Mask[i] = -Mask[i];
}
cudaMemcpyToSymbol(Mc, Mask, MASK_WIDTH*sizeof(float));
ConvolutionKernel<<<dimGrid, dimBlock>>>(Nd, Pd, MASK_WIDTH, WIDTH);
```

---

## 25. 1D Convolution Kernel with Constant Memory

- 경계 밖 원소를 0으로 두는 커널 (M 대신 Mc 사용):

```c
__global__ void
convolution_1D_basic_kernel(float *N, float *M, float *P, int Mask_Width, int Width)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    float Pvalue = 0;
    int N_start_point = i - (Mask_Width/2);
    for (int j = 0; j < Mask_Width; j++) {
        if (((N_start_point + j) >= 0) && ((N_start_point + j) < Width)) {
            Pvalue += N[N_start_point + j] * Mc[j];
        }
    }
    P[i] = Pvalue;
}
```

---

## 26. Are We Memory Limited?

- **1D:** 각 출력 원소당 2×MASK_WIDTH loads (M, N 각각), 2×MASK_WIDTH FLOPs → **Memory limited**
- **2D:** 각 출력 원소당 2×MASK_WIDTH² loads, 2×MASK_WIDTH² FLOPs → **Memory limited**

---

## 27. Tiled 1D Convolution – Basic Idea

- (Slide 33: ghost, N, tile 0/1/2/3, halo, P 다이어그램)

---

## 28. What Shall We Parallelize?

- 한 스레드가 무엇을 할까?
- 한 가지 답: (vector sum, matrix multiply와 같이) **출력 원소 하나**를 계산

---

## 29. Should We Use Shared Memory?

- Global memory에서 읽은 데이터를 **재사용**할 수 있는가?
- 재사용이 global memory bandwidth를 줄이므로 **shared memory** 사용

---

## 30. How Much Reuse Is Possible?

- (MASK_WIDTH = 5 예)
- Element 2: thread 4에서 1회
- Element 3: threads 4, 5에서 2회
- Element 4: threads 4, 5, 6에서 3회
- Element 5: threads 4, 5, 6, 7에서 4회
- Element 6: 4회, Element 7: 3회, Element 8: 2회, Element 9: 1회

---

## 31. What About the Halos?

- Halo도 shared memory에 복사할지?
- 두 가지 선택지를 다음 슬라이드에서 다룸

---

## 32. Tiling Strategies: Halo from Global vs. Shared

### Can access halo from global memory

- **방식:** halo는 global memory에서 직접 읽기
- **장점:** shared memory 재사용 최적화 (halo 재사용은 적음)
- **단점:** Branch divergence (shared vs. global read), halo가 좁아 memory burst를 채우기 어려움

### Can load halo to shared memory

- **방식:** halo도 shared memory에 로드
- **장점:** Global memory 접근 coalescing, 계산 단계에서 branch divergence 없음
- **단점:** 일부 스레드는 1회 이상 로드 → 로드 시 약간의 branch divergence, shared memory 사용량 소폭 증가

---

## 33. Three Tiling Strategies (Overview)

- **Strategy 1:** Block size = output tile; 여러 단계로 input tile 로드 (Step 1, 2, 3)
- **Strategy 2:** Block size = input tile; input tile을 한 번에 로드; 출력 계산 시 일부 스레드 비활성화
- **Strategy 3:** Block size = output tile; input tile의 “core”만 로드; halo는 global memory에서 접근

---

## 34. Strategy 1: Variable Meanings for a Block

- (Slide 41: N inputs, N_ds, left/right halo, radius, i, outputs, P)

---

## 35. Strategy 1 – Loading the Left Halo

```c
int radius = Mask_Width / 2;
int halo_index_left = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
if (threadIdx.x >= (blockDim.x - radius)) {
    N_ds[threadIdx.x - (blockDim.x - radius)] =
        (halo_index_left < 0) ? 0 : N[halo_index_left];
}
```

---

## 36. Strategy 1 – Loading the Internal Elements

```c
int index = blockIdx.x * blockDim.x + threadIdx.x;
if ((blockIdx.x * blockDim.x + threadIdx.x) < Width)
    N_ds[radius + threadIdx.x] = N[index];
else
    N_ds[radius + threadIdx.x] = 0.0f;
```

---

## 37. Strategy 1 – Loading the Right Halo

```c
int halo_index_right = (blockIdx.x + 1)*blockDim.x + threadIdx.x;
if (threadIdx.x < radius) {
    N_ds[radius + blockDim.x + threadIdx.x] =
        (halo_index_right >= Width) ? 0 : N[halo_index_right];
}
```

---

## 38. Strategy 1 – Putting It All Together

```c
// Load left halo
// Load internal elements
// Load right halo
__syncthreads();

float Pvalue = 0;
for (int j = 0; j < Mask_Width; j++) {
    Pvalue += N_ds[threadIdx.x + j] * Mc[j];
}
P[i] = Pvalue;
```

---

## 39. Strategy 1 – Alternative (Step 1)

- `start = i - radius` 기준으로 한 원소씩 로드:

```c
int start = i – radius;
if (0 <= start && Width > start) {
    N_ds[threadIdx.x] = N[start];
} else {
    N_ds[threadIdx.x] = 0.0f;
}
```

---

## 40. Strategy 1 – Alternative (Step 2)

```c
if (MASK_WIDTH – 1 > threadIdx.x) {
    start += TILE_SIZE;
    if (Width > start) {
        N_ds[threadIdx.x + TILE_SIZE] = N[start];
    } else {
        N_ds[threadIdx.x + TILE_SIZE] = 0.0f;
    }
}
```

---

## 41. Strategy 3 – Loading Data

- Core만 shared에 로드 (경계 검사 생략 예):

```c
int i = blockIdx.x * blockDim.x + threadIdx.x;
__shared__ float N_ds[TILE_WIDTH];
N_ds[threadIdx.x] = N[i];  // boundary checking is missing here
__syncthreads();
```

---

## 42. Strategy 3 – Computing

- Halo는 global에서 읽기:

```c
int radius = Mask_Width / 2;
int This_tile_start_point = blockIdx.x * blockDim.x;
int Next_tile_start_point = (blockIdx.x + 1) * blockDim.x;
int N_start_point = i - radius;
float Pvalue = 0;
for (int j = 0; j < Mask_Width; j++) {
    int N_index = N_start_point + j;
    if (N_index >= 0 && N_index < Width) {
        if ((N_index >= This_tile_start_point) &&
            (N_index < Next_tile_start_point))
            Pvalue += N_ds[threadIdx.x - radius + j] * M[j];
        else
            Pvalue += N[N_index] * Mc[j];
    }
}
P[i] = Pvalue;
```

---

## 43. Review: What Shall We Parallelize?

- (Strategy 1 & 3) 한 스레드 = 출력 원소 하나 계산
- Strategy 2도 선택 가능 — 다음 강의에서

---

## 44. Strategy 2: Parallelize Loading of a Tile

- 각 스레드가 **입력 원소 하나**를 로드하고, **일부 스레드만** 출력을 계산
- **장점:** 로드 단계에서 branch divergence 없음(높은 지연 숨기기), 좁은 global 접근(2×halo 폭) 회피
- **단점:** 계산 단계에서 branch divergence (지연은 상대적으로 낮음)
- 다음 강의에서 계속

---

## 45. Things to Read / Things to Do

### Things to Read

- Textbook chapter 7
- CUDA BPG: Memory Optimizations

### Things to Do

- Submit Lab 3
- Sign up for GPT-2 project, if you wish
