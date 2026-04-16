# Lab 4 Quiz Notes

Detailed takeaways are summarized in [doc/lab4_notes.md](../doc/lab4_notes.md).

# Question 1: Convolution Strategy

## 문제

For 1D convolution, when **MASK_WIDTH = 3**, which of the three tiling strategies presented in lectures is the most suitable?  
**(Hint: think about memory)**

- (a) Strategy 1  
- (b) Strategy 2  
- (c) Strategy 3  

---

## 풀이 과정

### 1. 세 가지 Tiling 전략 요약 (L08)

| 전략 | Block 크기 | Shared memory에 올리는 것 | Shared 원소 수 (1D) |
|------|------------|---------------------------|----------------------|
| **Strategy 1** | 출력 타일 (TILE_WIDTH) | 출력 타일 + halo 전부 (여러 단계로 로드) | TILE_WIDTH + (MASK_WIDTH − 1) = TILE_WIDTH + 2 |
| **Strategy 2** | 입력 타일 (TILE_WIDTH + MASK_WIDTH − 1) | 입력 타일 전부 한 번에 로드 | TILE_WIDTH + (MASK_WIDTH − 1) = TILE_WIDTH + 2 |
| **Strategy 3** | 출력 타일 (TILE_WIDTH) | **core(출력 타일)만** 로드; halo는 global에서 접근 | **TILE_WIDTH** |

### 2. “Think about memory” 해석

- 여기서 **memory**는 GPU에서 제한 자원인 **shared memory**를 의미한다.
- SM당 shared memory 양이 제한되어 있어, 블록당 사용량이 작을수록 **동시에 올릴 수 있는 블록 수(occupancy)** 가 늘어난다.

### 3. MASK_WIDTH = 3일 때

- **Radius = 1** → halo는 **좌우 각 1개**씩, 총 2개.
- Strategy 1, 2: shared에 **TILE_WIDTH + 2**개 저장.
- Strategy 3: shared에 **TILE_WIDTH**개만 저장 → **2개 적음.**

Strategy 3은 halo를 shared에 두지 않고 global에서 읽으므로:
- **Shared memory 사용량이 가장 작다** (core만 사용).
- MASK_WIDTH=3처럼 halo가 작을 때는, halo를 global에서 읽는 비용이 상대적으로 작고, shared 절약으로 인한 occupancy 이득이 유리하다.

### 4. 결론

- **(c) Strategy 3**이 “memory” 관점에서 가장 적합하다.  
- Block당 shared memory를 최소화하여 occupancy를 높일 수 있고, MASK_WIDTH=3일 때 halo 접근 부담이 크지 않다.

---

## 정리

| 질문 | 답 |
|------|-----|
| **MASK_WIDTH=3일 때 memory 관점에서 가장 적합한 1D convolution tiling 전략** | **(c) Strategy 3** |

### 한 줄 요약

- **Strategy 3**: 출력 타일(core)만 shared에 로드하고 halo는 global에서 읽음 → shared 사용량 최소(TILE_WIDTH) → occupancy에 유리.

---

# Question 2: 2D Convolution Tile Size

## 문제

Consider a 2D convolution tiling strategy in which each thread loads one value from input N. Consider **46 × 46 output tiles** and a **mask with radius 5**. In an **internal tile** (i.e. not near any boundary), what is the size of the corresponding **input tile**?

(정수 하나로 답 입력)

---

## 풀이 과정

### 1. 입력 타일과 출력 타일 관계 (L09)

- 2D convolution에서 **한 출력 원소**는 반경 **radius** 만큼의 이웃 입력을 사용한다.
- **출력 타일** 한 변이 **TILE_WIDTH**이면, 그 타일을 계산하려면 **상하좌우로 각각 radius만큼** 더 필요하다.
- 따라서 **입력 타일 한 변** = 출력 타일 한 변 + 2 × radius  
  → **Input tile (per dimension) = Output tile (per dimension) + 2 × radius**

### 2. 주어진 값

- 출력 타일: **46 × 46** → 한 변 = **46**
- Mask radius: **5**
- Internal tile이므로 경계(ghost) 보정 없이 위 공식 그대로 적용.

### 3. 계산

- 입력 타일 한 변 = 46 + 2 × 5 = **46 + 10 = 56**
- 입력 타일은 **56 × 56**. “size”는 **원소 개수**이므로 56 × 56 = **3136**.

---

## 정리

| 질문 | 답 |
|------|-----|
| **Internal tile에서 대응하는 input tile의 크기 (원소 개수)** | **3136** |

### 한 줄 요약

- **Input tile dimension = 46 + 2×5 = 56** → **크기(원소 개수) = 56 × 56 = 3136**.

---

# Question 3: Reuse in 2D Convolution II

## 문제

2D convolution tiling 전략에서 **각 스레드가 입력 N에서 값 하나를 로드**한다고 하자.

- **출력 타일:** 16 × 16  
- **Convolution mask:** 9 × 9  

다음 네 가지를 구하시오. (Q4만 조건이 다름.)

1. GPU가 SM당 최대 **2048 threads**를 지원할 때, **각 SM에서 동시에 실행 가능한 thread block 개수** (integer)  
2. **각 SM에서 사용하는 shared memory 바이트 수** (input은 float 배열, integer)  
3. **Internal tile**에서 shared memory에 로드된 값 하나당 **평균 사용 횟수** (4 significant figures)  
4. **같은 block 크기·같은 mask**이되, **각 스레드가 4개 값(2×2 타일)을 로드**하고 block 내 스레드 수는 그대로일 때, internal tile에서 shared에 로드된 값 하나당 **새 평균 사용 횟수** (4 significant figures)

---

## 풀이 과정

### 공통 설정

- 출력 타일: **16 × 16**, Mask: **9 × 9** → radius = (9−1)/2 = **4**
- 입력 타일 한 변 = 16 + 2×4 = **24** → 입력 타일 크기 = **24 × 24 = 576** (float 개수)
- **“각 스레드가 입력 1개 로드”** ⇒ block 크기 = **입력 타일 크기** = **576 threads/block** (Strategy 2: block = input tile)

---

### Q1: SM당 동시 실행 가능한 block 개수

- Block당 스레드 수 = **576** (입력 타일 원소 수 = 24×24)
- SM당 최대 스레드 = **2048**
- SM당 block 수 = ⌊2048 / 576⌋ = ⌊3.555…⌋ = **3**

**답: 3**

---

### Q2: SM당 shared memory 바이트 수

- Block 하나가 shared에 올리는 입력 타일: **24 × 24 = 576** floats
- Block당 shared memory = 576 × 4 bytes = **2304** bytes
- SM당 block 수 = 3 (Q1)
- SM당 shared memory = 2304 × 3 = **6912** bytes

**답: 6912**

---

### Q3: Internal tile에서 로드된 값 하나당 평균 사용 횟수

- Shared에 로드된 값 개수: **576**
- 출력 원소 수: 16 × 16 = **256**. 각 출력 원소가 mask 9×9로 **81**개 입력을 사용.
- 총 사용 횟수(총 “use” 수) = 256 × 81 = **20736**
- 평균 사용 횟수 = 20736 / 576 = **36**

**답: 36** (4 significant figures: 36.00)

---

### Q4: 스레드당 4개(2×2) 로드 시 평균 사용 횟수

- **“block 내 스레드 수는 그대로”** ⇒ 여전히 **24×24 = 576** threads (출력 타일 16×16이 아님)
- 스레드당 **2×2 = 4개** 로드 ⇒ shared 입력 타일 한 변 = 24 × 2 = **48** → **48×48 = 2304** 값
- 출력 타일 한 변 = 입력 한 변 − (K−1) = 48 − (9−1) = **40** → 출력 **40×40 = 1600** 원소
- 총 사용 횟수 = 1600 × 81 = **129600**
- 새 평균 = 129600 / 2304 = **56.25**

**답: 56.25** (4 significant figures)

---

## 정리

| 질문 | 답 |
|------|-----|
| **Q1. SM당 동시 실행 thread block 개수** | **3** |
| **Q2. SM당 shared memory (bytes)** | **6912** |
| **Q3. Internal tile, 값 하나당 평균 사용 횟수** | **36** |
| **Q4. 스레드당 4개 로드 시 평균 사용 횟수** | **56.25** |

### 요약 식

- Q1: block = 입력 타일 = 24×24 = 576 threads → `⌊2048/576⌋ = 3`
- Q2: `2304 bytes/block × 3 blocks = 6912`
- Q3: `(16×16 × 9×9) / (24×24) = 20736/576 = 36`
- Q4: 스레드 576명 유지, 각 4개 로드 → shared 48×48=2304, 출력 40×40=1600 → `1600×81/2304 = 56.25`

---

# Question 4: Data Reuse in 2D Convolution

## 문제

2D convolution tiling 전략에서 **각 스레드가 입력 N에서 값 하나를 shared memory에 로드**한다고 하자.

- **출력 타일:** 35 × 35  
- **Mask:** 11 × 21  

**특정 입력 원소 하나를 (평균적으로) 접근하는 thread block의 개수**를 N이 커질 때의 극한으로 구하시오.  
(Mask/필터 접근은 제외, 입력 N만 고려. 답은 유효숫자 4자리.)

---

## 풀이 과정

### 1. Block당 입력 타일 크기

- Mask 11×21 → row 방향 반경 (11−1)/2 = **5**, col 방향 반경 (21−1)/2 = **10**
- 출력 타일 35×35이므로, 한 block이 로드하는 입력 타일 한 변:
  - 행: 35 + 2×5 = **45**
  - 열: 35 + 2×10 = **55**
- 즉 block 하나의 입력 타일 = **45 × 55**

### 2. Block 격자와 입력 인덱스

- 출력 타일이 35×35이므로 block (k, ℓ)의 출력 타일 시작 인덱스: **(35k, 35ℓ)**
- 이 block이 사용하는 입력 영역:
  - 행: [35k − 5, 35k + 35 − 1 + 5] = [35k − 5, 35k + 39] → 길이 **45**
  - 열: [35ℓ − 10, 35ℓ + 35 − 1 + 10] = [35ℓ − 10, 35ℓ + 44] → 길이 **55**

### 3. 한 입력 원소 (i, j)를 접근하는 block 개수

입력 원소 (i, j)가 block (k, ℓ)의 입력 타일 안에 있을 조건:

- 35k − 5 ≤ i ≤ 35k + 39  ⇔  **35k ∈ [i − 39, i + 5]** (구간 길이 45)
- 35ℓ − 10 ≤ j ≤ 35ℓ + 44  ⇔  **35ℓ ∈ [j − 44, j + 10]** (구간 길이 55)

즉, (i, j)를 접근하는 block 수 = (구간 [i−39, i+5] 안의 35의 배수 개수) × (구간 [j−44, j+10] 안의 35의 배수 개수).

### 4. N이 클 때 평균

- 행 구간 길이 45, 열 구간 길이 55, block 간 stride = 35.
- 극한에서 균일 분포로 보면, 한 방향당 35의 배수 개수의 기댓값 ≈ 구간 길이 / stride.
- **행 방향 기댓값:** 45 / 35  
- **열 방향 기댓값:** 55 / 35  
- **평균 block 수** = (45/35) × (55/35) = (45×55) / (35×35) = **2475 / 1225 = 2.020408…**

유효숫자 4자리: **2.020**

---

## 정리

| 질문 | 답 |
|------|-----|
| **특정 입력 원소 하나당 (평균) 접근하는 thread block 개수** | **2.020** |

### 한 줄 요약

- 입력 타일 한 변 = 출력 한 변 + 2×반경 → 45, 55. 평균 block 수 = (45/35)×(55/35) = **2.020**.

---

# Question 5: Branch Divergence In 2D Convolution

## 문제

2D convolution tiling 전략에서 **각 스레드가 입력 N에서 값 하나를 shared memory에 로드**한다고 하자. **Internal tile** (경계 아님)에서:

- **출력 타일:** 21 × 21  
- **Mask:** 11 × 11  

**각 thread block에서 control/branch divergence가 발생하는 warp의 개수**를 구하시오 (integer).

- 스레드 수가 32의 배수가 아니면 마지막 warp는 꽉 차지 않음. **비어 있는 스레드는 divergence 원인으로 세지 않음.**
- 필요하면 출력을 만드는 스레드가 block의 **중앙**에 있다고 가정.

---

## 풀이 과정

### 1. Block 크기와 출력·비출력 구역

- “각 스레드가 값 하나 로드” ⇒ block 크기 = **입력 타일** 크기.
- Mask 11×11 → 반경 5. 입력 타일 한 변 = 21 + 2×5 = **31**.
- **Block당 스레드 수** = 31×31 = **961** (row-major로 (i, j) → 31i + j).
- 출력 타일은 **중앙** 21×21만 계산한다고 하면, 출력을 만드는 스레드는 (i, j)가 **행·열 모두 5~25**인 영역 → **21×21 = 441**개.

### 2. 어떤 스레드가 출력을 만드는가

- 출력 생성: (i, j) ∈ [5, 25]×[5, 25].
- 선형 인덱스: 최소 5×31+5 = **160**, 최대 25×31+25 = **800**.
- 따라서 인덱스 **160~800** 중 일부(441개)만 출력을 계산하고, **0~159**와 **801~960**은 로드만 함.

### 3. Divergence가 나는 warp

- **Divergence:** 같은 warp 안에 “출력 계산하는 스레드”와 “로드만 하는 스레드”가 섞여 있을 때.
- Warp 크기 32. Warp w = 스레드 [32w, 32w+31].
- 출력 구간 160~800을 포함하는 warp: 160이 속한 warp = 160/32 = **5**, 800이 속한 warp = 800/32 = **25**.
- 따라서 **warp 5 ~ warp 25** 안에는 출력 구간(160~800)과 그 밖(0~159 또는 801~960)이 같이 섞일 수 있음.
- 예: warp 5 (160~191) → 160~185는 출력 가능, 186~191은 로드만 등으로 같은 warp 내에서 branch가 갈림.  
  warp 25 (800~831) → 800은 출력, 801~831은 로드만 → 역시 divergence.

### 4. 개수

- Divergence가 나는 warp = **warp 5, 6, …, 25** → **21개**.
- 마지막 warp(warp 30)은 스레드가 1개뿐이지만, “비어 있는 스레드”로 인한 divergence는 세지 않으므로, 위 21개만 세면 됨.

---

## 정리

| 질문 | 답 |
|------|-----|
| **Block당 control/branch divergence가 발생하는 warp 개수** | **21** |

### 한 줄 요약

- Block = 31×31 = 961 threads, 출력은 중앙 21×21(인덱스 160~800). 이 구간과 겹치는 warp는 5~25 → **21 warps**에서 출력 유무에 따른 divergence 발생.

---
