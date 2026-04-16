# Lab 2 Quiz Notes

Detailed takeaways are summarized in [doc/lab2_notes.md](../doc/lab2_notes.md).

# Lab2 Quiz — Question 1: Control Divergence

## 문제 요약

- **이미지:** 208 × 206 (numRows = 208, numCols = 206)
- **커널:** `DiagonalKernel` — 각 스레드가 (row, col)에 대해
  - 경계: `(row < numRows) && (col < numCols)` 통과 시에만
  - `row > col` 이면 `d_Pout = d_Pin`, 아니면 `d_Pout = 0.0`
- **실행 설정:** 2D 그리드, 블록당 **16×16** 스레드
- **Warp 크기:** 32 (고정)

---

## Q1: 생성되는 warp 개수

### 1) 블록당 warp 수

- 한 블록 스레드 수: `16 × 16 = 256`
- Warp 수: `256 / 32 = 8` warps per block

### 2) 그리드 크기

- 행 방향(세로): `ceil(numRows / blockDim.y) = ceil(208 / 16) = 13`
- 열 방향(가로): `ceil(numCols / blockDim.x) = ceil(206 / 16) = 13`
- 총 블록 수: `13 × 13 = 169` blocks

### 3) 총 warp 수

\[
\text{Total warps} = 169 \times 8 = \mathbf{1352}
\]

---

## Q2: Control divergence가 발생하는 warp 개수

**Control divergence:** 같은 warp 안에서 어떤 스레드는 조건을 만족하고 어떤 스레드는 만족하지 않아, 서로 다른 경로(분기)를 타는 경우.

이 커널에는 **분기가 두 군데** 있다.

1. **경계 체크:** `if ((row < numRows) && (col < numCols))`
2. **대각 조건:** `if (row > col)` (그 안쪽)

각각에서 “이 조건 때문에 갈라지는 warp”를 세고, **중복을 한 번만** 세면 된다.

---

### (가) 경계 체크로 인한 divergence

- `numRows = 208 = 13×16` → **행 경계는 블록 단위로 딱 맞음.**  
  즉 모든 블록에서 `row`는 항상 `0..207` 안이므로 `row < numRows`로 인한 갈라짐은 없다.
- `numCols = 206` → **열은 16의 배수가 아님.**  
  그리드 열 개수는 `ceil(206/16)=13`이므로, 마지막 열 블록(`blockIdx.x = 12`)에서  
  `col = 12*16 + 0..15 = 192..207` 인데, 유효한 열은 0..205이므로  
  **col = 206, 207** 인 스레드 두 개는 `col < numCols`에 걸려서 바깥 경로로 빠진다.

같은 블록 안에서:
- `threadIdx.x = 0..13` → col = 192..205 → 조건 만족 (안쪽 실행)
- `threadIdx.x = 14, 15` → col = 206, 207 → 조건 불만족 (바깥쪽)

Warp는 **연속된 32개 스레드** 단위로 묶이므로, 16×16 블록을 어떻게 잘라도 “일부만 경계 안/밖”인 warp가 생긴다.  
즉 **`blockIdx.x = 12` 인 13개 블록 전체**에서, 각 블록의 **모든 8개 warp**가 “일부는 경계 안, 일부는 경계 밖”으로 나뉘어 **전부 divergence** 발생.

- 경계로만 갈라지는 warp 수: **13 blocks × 8 = 104 warps**

(이 13개 블록은 `(12, 0), (12, 1), …, (12, 12)`.)

---

### (나) 대각 조건 `row > col`로 인한 divergence

- `row > col` 이면 한 경로, `row <= col` 이면 다른 경로.
- 블록이 “대각선 `row = col`”을 **관통**할 때, 그 블록 안에 `row > col` 인 스레드와 `row <= col` 인 스레드가 같이 있으므로 **그 블록의 모든 warp가 divergence** 발생.

블록 인덱스로 보면:
- 블록 `(bx, by)` 에서 row ∈ [16·by, 16·by+15], col ∈ [16·bx, 16·bx+15].
- 이 블록을 대각선이 관통하는 조건:  
  “행 구간과 열 구간이 겹친다”  
  → `16·by ≤ 16·bx+15` 이고 `16·bx ≤ 16·by+15`  
  → `by ≤ bx + 1` 이고 `bx ≤ by + 1`  
  → `bx` 와 `by` 가 1 이하로 차이 나는 경우.  
  가장 전형적으로 **`bx = by`** 인 블록들: (0,0), (1,1), …, (12,12) → **13개 블록.**

이 13개 블록 각각에서 8개 warp 전부가 `row > col` vs `row <= col` 로 갈라진다.

- 대각 조건으로만 갈라지는 warp 수: **13 blocks × 8 = 104 warps**

---

### (다) 중복 제거 — block (12, 12)

- **경계 divergence:** blockIdx.x = 12 인 블록들 → (12,0)~(12,12) → 13개 블록
- **대각 divergence:** blockIdx.x = blockIdx.y 인 블록들 → (0,0), (1,1), …, (12,12) → 13개 블록

**블록 (12, 12)는 두 조건에 모두 해당**한다.  
이 블록의 8개 warp는 “경계”로도 갈라지고 “대각”으로도 갈라지지만, **divergence를 “가지고 있는 warp”로는 한 번만 세야** 한다.

- 경계만: 104 warps  
- 대각만: 104 warps  
- (12,12)의 8 warps는 위 둘에 공통이므로 한 번만 계산

\[
\text{Divergence 있는 warp 수} = 104 + 104 - 8 = \mathbf{200}
\]

---

## 최종 답

| 질문 | 답 |
|------|-----|
| Q1: 생성되는 warp 개수 | **1352** |
| Q2: Control divergence가 발생하는 warp 개수 | **200** |

---

## 요약 공식

- **총 warp 수**  
  `(ceil(numRows/blockDim.y) × ceil(numCols/blockDim.x)) × (blockDim.x × blockDim.y / 32)`
- **Divergence**  
  - 경계: `col < numCols` 가 블록을 “잘라” 자르는 열 블록들 → 여기서는 `blockIdx.x = 12` 인 13블록 → 104 warps  
  - 대각: `row = col` 이 블록을 관통하는 블록들 → `blockIdx.x = blockIdx.y` 인 13블록 → 104 warps  
  - 겹치는 블록 (12,12)의 8 warps를 한 번만 세어서 **200 warps**.

---

# Lab2 Quiz — Question 2: Control Divergence II

**문제:** 다음 CUDA 커널 중 control divergence가 **가능한** 것을 모두 고르시오.

---

## 정답: **A**, **D**

---

## 커널별 풀이

### A: `if (index < size)` — **Divergence 가능**

- `index = threadIdx.x + blockIdx.x * blockDim.x` → 스레드마다 다름.
- `size`는 상수이므로, 한 warp 안에 `index < size` 인 스레드와 `index >= size` 인 스레드가 같이 있을 수 있음.
- 예: warp가 index 0~31을 담당하고 `size = 10` 이면, 0~9는 참, 10~31은 거짓 → **한 warp 안에서 경로가 갈라짐.**

---

### B: `if (blockIdx.x < 16)` — Divergence 없음

- `blockIdx.x`는 **한 블록 전체**에서 동일.
- 한 warp는 한 블록 안에 있으므로, 그 warp의 모든 스레드가 같은 분기를 탄다.

---

### C: `if (control != 0)` — Divergence 없음

- `control`은 커널 인자로 **모든 스레드에 동일.**
- 모든 스레드가 같은 조건 판단 → 같은 경로.

---

### D: `io[index] = threadIdx.x > 31 ? 1 : -1` — **Divergence 가능**

- Warp는 **linearized thread index** 순으로 32개씩 묶인다. 2D/3D 블록에서는 `threadIdx.x`가 아니라 (예: 2D면 `threadIdx.x + threadIdx.y * blockDim.x`) **선형 인덱스**로 warp가 구성된다.
- 예: **blockDim = (33, 2)** 인 경우  
  - 선형 인덱스 0~32: 첫 번째 행 (threadIdx.x = 0..32)  
  - 선형 인덱스 33~65: 두 번째 행 (threadIdx.x = 0..32)  
  - **Warp 1** = 선형 32~63 = (threadIdx.x=32, row 0) 한 개 + (threadIdx.x=0..30, row 1) 31개.  
  → 이 warp 안에 `threadIdx.x > 31` 인 스레드(1개)와 `threadIdx.x <= 31` 인 스레드(31개)가 **같이** 있음.  
  → 일부는 `1`, 일부는 `-1` 경로 → **같은 warp 안에서 분기 발생.**

따라서 블록 차원이 32의 배수가 아니거나, 2D/3D에서 선형 순서가 `threadIdx.x = 31` 경계를 warp가 관통하면 **D에서도 divergence가 가능**하다.

---

### E: `if (control < 0) control = 1;` 후 `if (control * threadIdx.x < 0)` — Divergence 없음

- 첫 번째 if 후 모든 스레드에서 `control >= 0`.
- `threadIdx.x >= 0` 이므로 `control * threadIdx.x >= 0` (오버플로 무시 시).  
  → 두 번째 if는 모든 스레드에서 거짓 → 한 경로만 실행.

---

## 요약

| 커널 | 조건 | Divergence |
|------|------|------------|
| A | `index < size` (스레드마다 다름) | **가능** |
| B | `blockIdx.x < 16` (블록 단위 동일) | 없음 |
| C | `control != 0` (상수) | 없음 |
| D | `threadIdx.x > 31` (2D 블록에서 한 warp가 31 경계 관통 가능) | **가능** |
| E | `control * threadIdx.x < 0` (항상 거짓) | 없음 |

---

# Lab2 Quiz — Question 3: Matrix Multiplication Memory and Computation

**주어진 차원 (float = 4 bytes):**

- A: 30 × 23 (numARows × numAColumns)
- B: 23 × 25 (numBRows × numBColumns)
- C: 30 × 25 (numCRows × numCColumns) = A × B

---

## Part A: Global memory에서 읽은 바이트 수

**Naive 커널**에서는 스레드마다 C 원소 하나를 계산할 때, 그 스레드가 **직접 global에서 읽는** A·B 원소 수를 세어야 한다. 같은 A·B 원소를 여러 스레드가 반복 읽어도, **읽기 발생 횟수(트래픽)** 로 센다.

- 스레드 수 = C 원소 수 = 30 × 25 = **750**
- 스레드당 A에서 읽는 float 수 = numAColumns = **23**
- 스레드당 B에서 읽는 float 수 = numBRows = **23**
- float = 4 bytes

\[
\text{Read (A)} = 750 \times 23 \times 4 = 69\,000,\quad
\text{Read (B)} = 750 \times 23 \times 4 = 69\,000
\]

\[
\textbf{Read} = 69\,000 + 69\,000 = \mathbf{138000}\;\text{Bytes}
\]

(참고: 행렬 자체 크기만 쓰면 5060 bytes이지만, 문제에서 “커널이 읽은 바이트”는 보통 **실제 읽기 트래픽**을 묻는 경우가 많다.)

---

## Part B: Global memory에 쓴 바이트 수

커널이 **쓰는** 데이터 = 출력 행렬 C 전체.

- C 원소 수: 30 × 25 = **750** → 750 × 4 = **3,000 bytes**

\[
\textbf{Written} = \mathbf{3000}\;\text{Bytes}
\]

---

## Part C: 수행한 floating-point 연산 수

C[i][j] = Σ_k A[i][k] × B[k][j] 이므로, 한 원소당 **곱셈 K번, 덧셈 K번** (누적 합이므로 곱셈 K개 + 덧셈 K개로 셈).

- M = numCRows = 30, N = numCColumns = 25, K = numAColumns = numBRows = 23
- 원소당 연산: 23 mul + 23 add = 46 FLOPs
- 총 원소 수: M × N = 750

\[
\text{FLOPs} = 2 \times M \times N \times K = 2 \times 30 \times 25 \times 23 = \mathbf{34500}\;\text{floating-point operations}
\]

(일반적으로 행렬곱 FLOPs는 \(2 \cdot M \cdot N \cdot K\) 로 셈.)

---

## 최종 답

| Part | 질문 | 답 |
|------|------|-----|
| A | Global memory 읽기 (Bytes) | **138000** |
| B | Global memory 쓰기 (Bytes) | **3000** |
| C | Floating-point operations | **34500** |

---

# Lab2 Quiz — Question 4: SIMT Statements

**질문:** SIMT에서 control divergence에 대한 설명으로 **맞는** 것을 고르시오.

---

## 선택지 분석

**(a)** "Control divergence happens when threads within the **same block** are executing different branches"

- **틀림.** Divergence는 **같은 warp** 안에서 일부 스레드는 한 분기, 일부는 다른 분기를 탈 때 발생한다. 단위는 **block이 아니라 warp**(32 threads)이다. 같은 블록이라도 서로 다른 warp면 divergence 정의와는 별개다.

**(b)** "Control divergence can lead to inefficiency because if any thread takes a different branch, all threads must wait until the longest path is completed, potentially leading to idle cycles for some threads"

- **맞음.** 한 warp에서 갈라지면, 하드웨어는 한쪽 경로를 실행할 때 나머지 스레드는 마스크로 끄고, 그다음 다른 경로를 실행한다. 그래서 **가장 긴 경로가 끝날 때까지** 그 warp 전체가 기다리게 되고, 짧은 경로만 탄 스레드들은 그동안 **idle**이다. → 비효율·처리량 감소가 맞는 설명이다.

**(c)** "Control divergence increases the computational throughput by utilizing more cores"

- **틀림.** Divergence는 처리량을 **줄이고** 비효율을 만든다. “throughput을 늘린다”는 반대다.

**(d)** "None of these are correct"

- **(b)**가 맞으므로 이건 틀림.

---

## 정답: **(b)**

---

# Lab2 Quiz — Question 5: Maximum Parallelism

**질문:** Compute capability 8.6 GPU에서 SM 하나당 최대 **1536 threads**, 최대 **16 thread blocks**를 지원할 때, 두 자원을 모두 최대한 쓰려면 **블록당 warp를 몇 개**로 해야 하는가? (정수로 답)

---

## 풀이

- SM당 최대 스레드 수와 블록 수를 **동시에** 꽉 채우려면:
  - 블록 수 = **16** (상한)
  - SM당 스레드 = 1536 → **스레드/블록 = 1536 ÷ 16 = 96**
- 블록당 warp 수 = 블록당 스레드 ÷ 32 = **96 ÷ 32 = 3**

\[
\text{warps per block} = \frac{1536}{16 \times 32} = \frac{1536}{512} = \mathbf{3}
\]

---

## 정답: **3** (블록당 3 warps)
