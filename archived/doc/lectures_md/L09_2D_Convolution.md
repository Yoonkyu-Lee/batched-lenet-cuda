# ECE 408 / CS 483 / CSE 408
## Lecture 9: 2D Tiled Convolution Kernel; Reuse Analysis

**Instructor:** Volodymyr Kindratenko  
**학기:** Spring 2026

> Source PDF: ece408-lecture9-2D-convolution-vk-SP26.pdf

---

## 1. Course Reminders

### Lab updates

- Lab 4 is due this week on Friday
- Project milestone 1 will be released soon

### Midterm 1

- **When:** March 3rd, 7–10pm
- **Where:** Your specific room assignment will be posted in Canvas
- **What:** Lectures 1–11, Labs 1–4
- **How:** paper-based
- **Alternative Exam Time:** as arranged; email instructor by 2/24/26 if you have a valid conflict
- **Study materials:** now posted on Canvas
- **Exam review session:** 3:00–5:30 PM, Saturday, February 28th, in 1015 ECEB

---

## 2. Today's Objectives

- Tiled convolution 알고리즘을 더 깊이 학습
  - Tiling의 세부 사항
  - Output tiles vs input tiles
- Lab 4 준비
- Tiled convolution/stencil 알고리즘의 **reuse 분석** 학습

---

## 3. Stencil Algorithms

- **Stencil:** 어떤 고정 패턴에 따라 배열 원소를 갱신하는 수치 데이터 처리 알고리즘
- Convolution은 그 한 예
- 예: 2D convolutional kernel (9-point 2D stencil), Nearest neighbor lattice (5-point 2D), Dirac (3D), Finite difference (explicit time-marching, 13-point 3D), Wilson-Dslash (4D stencil)

---

## 4. Review: Three Tiling Strategies

- **Strategy 1:** Block size = output tile; 여러 단계로 input tile 로드 (Step 1, 2, 3); shared memory 사용
- **Strategy 2:** Block size = input tile; input tile을 한 번에 로드; 출력 계산 시 일부 스레드만 사용
- **Strategy 3:** Block size = output tile; input tile의 “core”만 로드; halo는 global memory에서 접근

---

## 5. Review: What Shall We Parallelize?

- 한 스레드가 할 일: (vector sum, matrix multiply와 같이) **출력 원소 하나** 계산
- Strategy 1 & 3에서 이 선택
- Strategy 2는 다른 선택 (다음 슬라이드)

---

## 6. Strategy 2: Parallelize Loading of a Tile

- **방식:** Thread block 크기 = input tile 크기
- 각 스레드가 input tile 원소 **하나**를 로드
- 출력 계산에는 **일부 스레드만** 참여
- **장점:** 로드 시 branch divergence 없음(높은 지연 숨김); 좁은 global 접근(2×halo 폭) 회피
- **단점:** 계산 시 branch divergence (지연은 상대적으로 낮음)

---

## 7. Parallelizing Tile Loading

- N의 한 tile을 shared memory에 로드
- **모든 스레드**가 로드에 참여
- 그 다음 **일부 스레드**만 shared memory의 각 N 원소를 사용
- Block 크기: TILE_WIDTH (출력), 입력 tile은 TILE_WIDTH + (MASK_WIDTH−1) 등으로 더 큼

---

## 8. Output Tiles Still Cover the Output

- 출력 좌표 (output tile 기준):

```c
col_o = blockIdx.x * TILE_WIDTH + threadIdx.x;
row_o = blockIdx.y * TILE_WIDTH + threadIdx.y;
```

---

## 9. Input Tiles Need to Be Larger than Output Tiles

- 입력 tile에는 halo가 포함되어 출력 tile보다 큼
- (Slide 10: Input Tile vs Output Tile 다이어그램)

---

## 10. Setting Block Dimensions

- Block이 **input tile** 전체를 로드할 수 있도록:

```c
dim3 dimBlock(TILE_WIDTH + (MASK_WIDTH - 1), TILE_WIDTH + (MASK_WIDTH - 1), 1);
```

- 일반적으로 (정사각 블록): block 한 변 = `TILE_WIDTH + (MASK_WIDTH - 1)`
- 예: `TILE_WIDTH + 4` when MASK_WIDTH is 5

```c
dim3 dimGrid(ceil(P.width / (1.0*TILE_WIDTH)),
             ceil(P.height / (1.0*TILE_WIDTH)), 1);
```

- Grid: 모든 P 원소를 만들 만큼의 block 수
- Block: 입력 tile 전체를 로드할 만큼의 스레드 수

---

## 11. Shifting From Output Coordinates To Input Coordinates

```c
int tx = threadIdx.x;
int ty = threadIdx.y;
int row_o = blockIdx.y * TILE_WIDTH + ty;
int col_o = blockIdx.x * TILE_WIDTH + tx;
int row_i = row_o - 2;  // MASK_WIDTH / 2 (radius)
int col_i = col_o - 2;
```

---

## 12. Threads That Load Halos Outside N Should Return 0.0

- N 범위 밖의 halo를 로드하는 스레드는 0.0을 넣어야 함

---

## 13. Taking Care of Boundaries

```c
if ((row_i >= 0) && (row_i < Width) &&
    (col_i >= 0) && (col_i < Width)) {
    tile[ty][tx] = N[row_i*Width + col_i];
} else {
    tile[ty][tx] = 0.0f;
}
__syncthreads();  // wait for tile
```

---

## 14. Not All Threads Calculate and Write Output

- 출력을 계산·기록하는 것은 `ty < TILE_WIDTH && tx < TILE_WIDTH` 인 스레드만

```c
float Pvalue = 0.0f;
if (ty < TILE_WIDTH && tx < TILE_WIDTH) {
    for (i = 0; i < 5; i++) {
        for (j = 0; j < 5; j++) {
            Pvalue += Mc[i][j] * tile[i+ty][j+tx];
        }
    }
    if (row_o < Width && col_o < Width) {
        P[row_o * Width + col_o] = Pvalue;
    }
}
```

---

## 15. Alternatively: Strategy 3 in 2D

- 1D Strategy 3 tiled convolution을 2D로 확장 가능
- 각 input tile이 해당 output tile과 크기만 맞추고, **모든 halo**는 global memory에서 로드
- 내적 계산 단계에서 if 조건과 divergence 발생

---

## 16. Reuse Analysis

- Tiled convolution의 이득을 **정량화**
- 커널 구현 전략(Strategy 1/2/3)에 관계없이 적용

---

## 17. 1D Convolution: Small Example

- TILE_WIDTH = 8, MASK_WIDTH = 5
- 한 block이 로드하는 원소 수: \(8 + (5 - 1) = 12\) → **12 memory loads**

---

## 18. Each Output Uses MASK_WIDTH Inputs

- P[8] uses N[6], N[7], N[8], N[9], N[10]
- P[9] uses N[7], …, N[11]
- …
- P[15] uses N[13], …, N[17]
- 출력 tile 하나당 **8 × 5**개의 N 값 사용 (총 40회 접근)

---

## 19. Simple Way to Calculate Tiling Benefit (1D)

- **Unique elements loaded:** \(8 + (5-1) = 12\)
- **Global memory accesses replaced by shared:** \(8 \times 5 = 40\)
- **Bandwidth reduction:** \(40/12 = 3.3\times\)
- N 크기와 무관

---

## 20. General Formula: 1D Convolution

- Global → shared 로드: **(TILE_WIDTH + MASK_WIDTH − 1)** elements
- Global 접근을 shared 접근으로 대체: **(TILE_WIDTH × MASK_WIDTH)** accesses
- **Bandwidth reduction:**
\[
\frac{\text{TILE\_WIDTH} \times \text{MASK\_WIDTH}}{\text{TILE\_WIDTH} + \text{MASK\_WIDTH} - 1}
\]

---

## 21. Another Way to Look at Reuse (1D)

- tile[6] 1회, tile[7] 2회, …, tile[10] 5회, …, tile[14] 4회, …, tile[17] 1회
- 각 접근은 입력 N에 대한 global 접근 한 번을 대체
- 대체되는 총 global 접근 수: \(1+2+3+4 + 5\times(8-5+1) + 4+3+2+1 = 10+20+10 = 40\)
- 원소 12개 기준 평균: \(40/12 = 3.3\)

---

## 22. General 1D: Total Replaced Accesses (Internal Tiles)

- (TILE_WIDTH + MASK_WIDTH − 1)개 N 원소에 대한 global 접근이 shared로 대체되는 총 횟수:
\[
(\text{MASK\_WIDTH}-1)\cdot\text{MASK\_WIDTH} + \text{MASK\_WIDTH}\cdot(\text{TILE\_WIDTH}-\text{MASK\_WIDTH}+1) = \text{MASK\_WIDTH} \times \text{TILE\_WIDTH}
\]

---

## 23. Boundary Tiles (1D)

- 경계 tile에서는 P[0], P[1], P[2], … 가 쓰는 N 원소 수가 더 적음
- 8×5보다 적은 N 원소만 사용

---

## 24. Ghost Elements Change Ratios (1D)

- 경계 tile에서 로드하는 원소 수: **TILE_WIDTH + (MASK_WIDTH−1)/2** (예: TILE_WIDTH=8, MASK_WIDTH=5 → 10)
- 경계 원소 계산 시 ghost는 global에서 읽지 않음
- 총 접근 수 예: \(6\times5 + 4 + 3 = 37\)
- **Reduction:** \(37/10 = 3.7\)

---

## 25. Bandwidth Reduction for 1D

- **Reduction:** \(\displaystyle\frac{\text{TILE\_WIDTH} \times \text{MASK\_WIDTH}}{\text{TILE\_WIDTH} + \text{MASK\_WIDTH} - 1}\)

| TILE_WIDTH | 16   | 32   | 64   | 128  | 256  |
|------------|------|------|------|------|------|
| MASK_WIDTH=5  | 4.0  | 4.4  | 4.7  | 4.9  | 4.9  |
| MASK_WIDTH=9  | 6.0  | 7.2  | 8.0  | 8.5  | 8.7  |

---

## 26. 2D: 8×8 Output Tile, MASK_WIDTH 5

- Input tile 로드: \((8+5-1)^2 = 144\) reads
- 각 출력 원소: \(5^2 = 25\) input 원소 사용
- 출력 tile 계산 시 global 접근 \(8\times8\times25 = 1{,}600\)이 shared 접근으로 대체
- **Bandwidth reduction:** \(1{,}600/144 = 11.1\times\)

---

## 27. General 2D Formula

- Shared로 로드하는 N 원소 수: **(TILE_WIDTH + MASK_WIDTH − 1)²**
- 각 P 원소가 접근하는 N 원소 수: **MASK_WIDTH²**
- Global → shared로 대체되는 접근 수: **(TILE_WIDTH × MASK_WIDTH)²**
- **Bandwidth reduction:**
\[
\frac{(\text{TILE\_WIDTH} \times \text{MASK\_WIDTH})^2}{(\text{TILE\_WIDTH} + \text{MASK\_WIDTH} - 1)^2}
\]

---

## 28. Bandwidth Reduction for 2D

| TILE_WIDTH | 8    | 16   | 32   | 64   |
|------------|------|------|------|------|
| MASK_WIDTH=5 | 11.1 | 16   | 19.7 | 22.1 |
| MASK_WIDTH=9 | 20.3 | 36   | 51.8 | 64   |

---

## 29. Ghost Elements in 2D

- 2D에서도 경계 tile은 비율이 달라짐 — 계산은 독자 연습으로 남김

---

## 30. 2B/FLOP for Untiled Convolution

- Untiled convolution에서 FLOP당 global memory 바이트 수?
- N 원소 하나(4B, global) × M 원소(4B, constant cache) → multiply 1 FLOP, add 1 FLOP
- ➡ **2 B/FLOP**

---

## 31. Full Use of Compute Requires ~13.3× Reuse (~2010 GPU)

- ~2010 GPU: 1,000 GFLOP/s, 150 GB/s
- \(150\ \text{GB/s} \div (2\ \text{B/FLOP}) = 75\ \text{GFLOP/s}\) → peak의 **7.50%**
- Peak를 쓰려면 최소 \(100/7.50 \approx 13.3\times\) reuse 필요

---

## 32. In 2020, Need ~52× Reuse

- 2020 GRID K520: ~5,000 GFLOP/s, 192 GB/s
- \(192\ \text{GB/s} / (2\ \text{B/FLOP}) = 96\ \text{GFLOP/s}\) → **1.92%** of peak
- Peak를 쓰려면 최소 \(100/1.92 \approx 52.1\times\) reuse 필요

---

## 33. Need ~26× Reuse on H100 GPUs

- 2023 H100 PCIe: 26 TFLOP/s, 2 TB/s
- \(2\ \text{TB/s} / (2\ \text{B/FLOP}) = 1\ \text{TFLOP/s}\) → **3.85%** of peak
- Peak를 쓰려면 최소 \(100/3.85 \approx 26\times\) reuse 필요

---

## 34. Need Really Big Mask to Balance (1D, TILE_WIDTH 1024)

- % of peak compute (1D tiled, TILE_WIDTH 1024):

| MASK_WIDTH | ~2010 (1K GFLOP/s, 150 GB/s) | ~2020 (5K GFLOP/s, 192 GB/s) |
|------------|------------------------------|------------------------------|
| 5          | 37%                          | 9.6%                         |
| 9          | 67%                          | 17%                          |
| 15         | 100%                         | 28%                          |
| 55         | 100%                         | 100%                         |

---

## 35. Need Really Big Mask to Balance (2D, TILE_WIDTH 32×32)

- % of peak compute (2D tiled, 32×32):

| MASK_WIDTH | ~2010 (1K GFLOP/s, 150 GB/s) | ~2020 (5K GFLOP/s, 192 GB/s) |
|------------|------------------------------|------------------------------|
| 3          | 60%                          | 15%                          |
| 5          | 100%                         | 37%                          |
| 7          | 100%                         | 67%                          |
| 9          | 100%                         | almost 100%                 |

---

## 36. Food for Thought

- 경계 tile은 비율이 다름
- 각 스레드가 4B씩 shared에 로드 → 2,048 스레드면 **8 kB**
- Shared memory는 보통 64 kB 이상 → 나머지로 무엇을 할 수 있을까?
- 개선된 접근은 숙제로 (예: MW=7에서 67% → 81% 등)

---

## 37. Ampere SM Memory Architecture

- **Access time:** registers (~1 cycle), shared (~5), cache/constant (~5), global (~500 cycles)
- **Register File:** 256 KB
- **L1 Cache / Shared Memory:** 192 KB (max 164 KB to Shared Mem)
- SM에 매핑되는 block 수는 다음에 의해 제한:
  - SM이 지원하는 총 스레드 수
  - SM의 shared memory 양
- (Ampere SM, GPU from 2020)

---

## 38. Overall Data Parallel Pipeline

- (Figure: Warp Scheduler, Coarse Decoder, Order / Collect, Banks 0..n, Execute, Register File, L1 Cache / Shared Mem, To L2 Cache / Memory)

---

## 39. Memory Hierarchy Considerations

- **Register file:** 많이 뱅킹됨 → bank conflict 시 pipeline stall
- **Shared memory:** 많이 뱅킹됨 → bank conflict 시 pipeline stall
- **Global memory:** 여러 channel, bank, page; bursting에 의존; **coalescing** 중요, 프로그래머 관여 필요
- **L1 Cache:** non-coherent

---

## 40. Things to Read / Things to Do

### Things to Read

- Textbook chapter 7
- CUDA BPG: Memory Optimizations

### Things to Do

- Submit Lab 4
- Sign up for GPT-2 project, if you wish

---

## 41. Problem Solving

### Q1: 2D tiled convolution, Strategy 1, block (0,0)

**Q:** 2D tiled convolution, mask 5×5, output tile 16×16, input image 32×32, Strategy 1. Shared memory 사용으로 global memory 접근이 얼마나 줄어드는가? (thread block (0,0) 기준)

**A:**  
- Global memory 접근 수: row 0부터 세면 예) row 0: 9+12+15×14, row 1: 12+16+20×14, 나머지 14 rows: 15+20+25×14 등 (구체적 계산은 슬라이드 44 수식 참고)  
- 사용하는 입력 원소 수: \((16+2)\times(16+2) = 18\times18\)

---

### Q2: Strategy 2, divergence

**Q:** 30×30 output tile, 3×3 mask (→ 32×32 input tile), Strategy 2 (block size = input tile). 각 thread block에서 control divergence가 있는 warp는 몇 개인가?

**A:**  
- Thread block 크기 32×32  
- 모든 32×32 스레드가 데이터 로드에 참여  
- Warp 0: 0만 로드, 계산 안 함  
- Warp 1–30: 30 threads는 계산, 2 threads는 계산 안 함  
- Warp 31: 0만 로드, 계산 안 함  
- ➡ **Divergence가 없는 warp는 2개** (전부 계산하거나 전부 안 하는 warp만)

---

### Q3: A40 GPU, B/FLOP and reuse

**Q:** A40 GPU: 37.4 TFLOPS (FP32), 48 GB GDDR6, 696 GB/s, FP64 미지원. Compute를 완전히 쓰려면 byte-to-FLOP 비율은? 필요한 데이터 reuse는?

**A:**  
- \(696\ \text{GB/s} \div 37.4\ \text{TFLOPS} = 0.019\ \text{B/FLOP}\)  
- 즉, global memory에서 로드한 1 byte당 **53.7 FLOP** 정도를 해야 compute를 꽉 채울 수 있음
