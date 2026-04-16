# M2 PrairieLearn Quiz — Answers

Source data regenerated after cleanup:
- gprof: [outputs/m2/m2_cpu_gprof_report.txt](outputs/m2/m2_cpu_gprof_report.txt) (B=1000 CPU run)

---

## Question 1 — CPU Profiling (2 pts)

**Prompt**:
> Show percentage of total execution time of your program spent in your forward pass function with gprof for batch size of 1k images. Paste the text copy of your gprof output. You should only include the line that includes your CPU forward pass function `conv_forward_cpu`, so please do not give more than this line. Please follow the exact format of your gprof output.
>
> For example: `92.xx  44.xx  44.xx  2  22.xx  22.xx  conv_forward_cpu(...)`

**Answer (paste this exact line)**:

```
 88.84     55.67    55.67        2    27.84    27.84  conv_forward_cpu(float*, float const*, float const*, int, int, int, int, int, int)
```

### 해석
- **88.84 %** — 전체 실행 시간 중 `conv_forward_cpu`가 차지하는 비율
- **55.67 s (cumulative)** / **55.67 s (self)** — 본 함수 자체에서 쓴 시간 (호출 자식 제외)
- **calls = 2** — conv 레이어가 2개 (Conv1, Conv2)라 두 번 호출
- **27.84 s/call (self, total)** — 호출 한 번당 평균 self / total 시간

기대대로 **CPU 시간의 거의 90%가 conv에 묶임** — 그래서 GPU로 옮기는 게 의미 있음.

---

## Question 2 — Basic GPU NSys Profiling (1) (1 pt)

**Prompt**:
> Profile the basic GPU convolution for batch size of 10k images using `nsys`. Show a list of all kernels that cumulatively consume more than 90% of the program time (listing from the top of your nsys results in profile.out until the cumulative Time is greater than 90%). You should paste the entire line of output for each kernel.
> If you're pasting multiple lines, please separate them with semicolons `;`.

Source: [profiles/nsys/m1_gpu_nsys_profile.out](profiles/nsys/m1_gpu_nsys_profile.out) section `[6/8] cuda_gpu_kern_sum`.

**Answer**:

```
    100.0         57082961          2  28541480.5  28541480.5  11537253  45545708   24047609.1  conv_forward_kernel(float *, const float *, const float *, int, int, int, int, int, int)
```

### 해석
- basic GPU kernel은 단 한 종류 (`conv_forward_kernel`)만 돌아가고 **100%** 차지 → 한 줄로 90% 컷오프 통과.
- Instances = 2 (Conv1 + Conv2 각 1회), avg 28.5 ms, 최소 11.5 ms / 최대 45.5 ms (Conv1 vs Conv2 비대칭: 채널 수 4→16).
- 나머지 `do_not_remove_this_kernel`, `prefn_marker_kernel`은 각각 0.0%로 무시 가능.

---

## Question 3 — Basic GPU NSys Profiling (2) (1 pt)

**Prompt**:
> Refer to `profile.out` and show a list of all CUDA API calls that cumulatively consume more than 90% of the program time for batch size of 10k images. Paste the entire line for each. Separate multiple lines with `;`.

Source: [profiles/nsys/m1_gpu_nsys_profile.out](profiles/nsys/m1_gpu_nsys_profile.out) section `[5/8] cuda_api_sum`.

Cumulative: 59.4 → 85.7 → **97.8%** (3개 줄로 90% 돌파).

**Answer** (세미콜론으로 연결):

```
     59.4        279596387          8  34949548.4  9237936.0     20258  140694091   54188089.9  cudaMemcpy            ;     26.3        123882167          8  15485270.9   206026.0    108734  122031871   43051710.7  cudaMalloc            ;     12.1         57135747          8   7141968.4     5330.5      1603   45563567   16040713.5  cudaDeviceSynchronize
```

개별 줄:
```
     59.4        279596387          8  34949548.4  9237936.0     20258  140694091   54188089.9  cudaMemcpy
     26.3        123882167          8  15485270.9   206026.0    108734  122031871   43051710.7  cudaMalloc
     12.1         57135747          8   7141968.4     5330.5      1603   45563567   16040713.5  cudaDeviceSynchronize
```

### 해석
- **cudaMemcpy 59.4%** — host↔device 전송이 가장 큰 비용 (B=10000의 input/output 덩어리).
- **cudaMalloc 26.3%** — 8회 (conv 2개 × {input, output, mask, device_output copy-back path}). 첫 alloc이 특히 길어 max 122 ms.
- **cudaDeviceSynchronize 12.1%** — kernel 끝 기다리는 블로킹. max 45.5 ms (Conv2 커널 시간과 일치).

---

## Question 4 — Basic GPU NSys Profiling (3) (2 pts, manual)

**Prompt**:
> In the first of the two convolution layers, determine which type of operation takes the most time. Upload a screenshot (`nsys_timeline.png`) of the Nsight Systems GUI timeline to support your answer.
>
> (a) CUDA kernels / (b) CUDA APIs

**Answer**: **(b) CUDA APIs**

### 근거 (데이터로 추론)
첫 번째 conv 레이어에서 일어나는 각 요소의 시간 (B=10000, `profile.out` max 값 기반):

| 구성 요소 | 시간 | 위치 |
|---|---|---|
| `cudaMemcpy` (H→D input + mask) | **~140 ms (max)** | prolog |
| `cudaMalloc` (초기 device buffer) | **~122 ms (max)** | prolog |
| `conv_forward_kernel` (Conv1) | **~11.5 ms (min)** | conv_forward_gpu |
| `cudaDeviceSynchronize` | ~11.5 ms | kernel launch 이후 |

첫 번째 레이어에서:
- **kernel 자체는 11.5 ms** (`kern_sum`의 min, 두 레이어 중 작은 쪽이 Conv1: 1→4 채널)
- **CUDA API (cudaMemcpy + cudaMalloc)는 도합 260 ms 수준** — prolog에서 모든 device 메모리 할당과 input 전송이 첫 레이어 구간에 몰림

즉 첫 레이어 타임라인에서는 긴 `cudaMalloc`/`cudaMemcpy` 막대가 짧은 `conv_forward_kernel`을 압도함 → **CUDA APIs**가 지배적.

### 스크린샷 가이드
1. Delta에서 로컬로 `profiles/nsys/m1_gpu_profile.nsys-rep` 다운로드 (scp / sftp).
2. 로컬에 설치된 **Nsight Systems GUI**로 열기.
3. Timeline에서 첫 conv 레이어 구간 줌인:
   - `CUDA HW` row의 노란색/주황 `cudaMemcpy` 막대들과 `cudaMalloc` 막대가 긴 부분
   - 그 바로 뒤에 나타나는 얇은 `conv_forward_kernel` 블록
4. 두 영역이 한 화면에 보이게 캡처 → `nsys_timeline.png`로 저장 → 업로드.

(Claude가 GUI 캡처는 직접 못 해서, 이 단계는 로컬에서 직접 해야 함.)

---

## Question 7 — Input Unrolling (1) (1 pt, select all)

**Prompt**:
> Which of the following statements are reasons why input unrolling serves as an optimization for basic GPU convolution?

**Answer**: **(a), (b)**

| | 선택 | 이유 |
|---|---|---|
| (a) matmul이 basic conv보다 병렬성이 좋다 | ✅ | basic conv 커널은 경계/채널 루프로 thread utilization 낮음; tiled matmul은 워프가 꽉 차서 돌아감 |
| (b) GPU는 matmul에 하드웨어/라이브러리가 잘 최적화돼 있음 | ✅ | Tensor Core, cuBLAS, tiled shared-memory GEMM 등 matmul 전용 경로가 풍부 |
| (c) 펼친 matrix가 원본보다 작다 | ❌ | **정반대**. `Channel*K*K × Batch*H_out*W_out`로, 각 입력 픽셀을 K²번 복제하므로 원본보다 훨씬 큼 |
| (d) 펼치면 총 MAC 연산 수가 줄어든다 | ❌ | 같은 수의 multiply-add. 재배치일 뿐 연산량은 동일 |
