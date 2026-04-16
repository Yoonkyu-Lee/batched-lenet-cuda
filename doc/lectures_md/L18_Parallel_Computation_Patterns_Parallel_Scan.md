# ECE 408 / CS 483 / CSE 408
## Lecture 18: Parallel Computation Patterns - Parallel Scan

**Instructor:** V. Kindratenko  
**학기:** Spring 2026

> Source: `lectures_pdf/ece408-lecture18-prefix_sum-vk-SP26.pdf`

---

## 1. Lecture Briefing

이번 18강은 **parallel scan (prefix sum)** 을 reduction에서 자연스럽게 이어지는 핵심 패턴으로 다룬다.

강의의 큰 흐름은:

- scan이 무엇인지 정의
- Kogge-Stone scan으로 low-latency 접근 이해
- Brent-Kung scan으로 work-efficient 접근 이해
- block-level segmented scan에서 시작해
- arbitrary-length input을 위한 hierarchical three-kernel scan으로 확장

Lab 7은 사실상 이 강의의 구현 과제다.

### 핵심 메시지

- scan은 reduction과 비슷하지만 **모든 prefix 결과**를 출력한다.
- parallel scan은 latency와 work efficiency 사이의 tradeoff가 있다.
- **Kogge-Stone**은 빠르지만 `O(n log n)` work라 비효율적이다.
- **Brent-Kung**은 `O(n)` work로 더 효율적이며, Lab 7이 요구하는 방향과 더 가깝다.
- 큰 입력은 block-level scan 하나로 끝나지 않으므로 **scan -> scan block sums -> add block sums** 구조가 필요하다.

---

## 2. What Scan Computes

### Inclusive vs Exclusive

강의는 scan을 associative operator에 대한 prefix computation으로 정의한다.

- Inclusive scan:
  - `y[i] = x[0] + x[1] + ... + x[i]`
- Exclusive scan:
  - `y[i] = x[0] + x[1] + ... + x[i-1]`

sum scan에서는:

```cpp
output[0] = input[0];
for (int i = 1; i < N; ++i) {
  output[i] = output[i - 1] + input[i];
}
```

이 형태가 inclusive scan의 sequential baseline이다.

---

## 3. Segmented Scan as the First Step

강의는 바로 전체 grid synchronization 문제를 짚는다.

- block 안에서는 `__syncthreads()`로 협업 가능
- block 사이에는 kernel 안에서 직접 barrier를 만들 수 없음

그래서 첫 단계는:

- each block scans one segment
- block sums are collected
- later kernels consolidate those block sums

즉 scan도 reduction처럼 hierarchical하게 접근해야 한다.

---

## 4. Kogge-Stone Scan

Kogge-Stone은 reduction tree를 겹쳐서 low-latency scan을 만든다.

특징:

- `log n` 단계
- 각 단계에서 많은 thread가 active
- latency는 좋음
- 하지만 total work는 `O(n log n)`

강의는 naive shared-memory Kogge-Stone 코드가 왜 틀릴 수 있는지도 설명한다.

문제:

- 어떤 thread는 이전 값을 읽어야 하는데
- 다른 thread가 같은 buffer를 먼저 덮어쓸 수 있음

해결:

- read와 write 사이에 synchronization 필요
- 또는 double buffering으로 false dependence 제거

이 부분은 Lab 7 quiz에도 직접 연결된다.

---

## 5. Warp-Level Scan

강의 중반에는 warp-level primitives를 사용한 scan도 소개한다.

핵심 아이디어:

- warp 내부는 `__shfl_up_sync` 같은 primitive로 prefix를 빠르게 전달 가능
- block scan은
  - warp scan
  - warp sums collection
  - warp sums scan
  - add back
  로 분해 가능

이건 synchronization overhead를 줄이는 현대적 구현 방향이지만, Lab 7 skeleton은 더 전통적인 shared-memory 계층 스캔을 요구한다.

---

## 6. Work Efficiency and Why Brent-Kung Matters

강의는 Kogge-Stone의 work inefficiency를 분명히 짚는다.

- sequential scan: `O(n)`
- Kogge-Stone: `O(n log n)`

그리고 balanced tree 관점에서 scan을 다시 구성해 **Brent-Kung**으로 넘어간다.

Brent-Kung의 핵심:

- reduction tree (upsweep)
- post-scan / distribution tree (downsweep)
- total useful adds는 `O(n)`

즉 latency는 더 길 수 있지만, total work는 훨씬 좋다.

강의는 특히 다음 메시지를 남긴다.

- Brent-Kung uses half the threads of Kogge-Stone in the block formulation
- each thread often loads two elements
- total work efficiency가 좋아서 large-scale scan에서 더 실용적이다

---

## 7. Three-Kernel Hierarchical Scan

이 부분이 Lab 7의 직접 구현 범위다.

강의 후반 핵심 슬라이드:

1. First kernel: block-level segmented scan
2. Second kernel: scan block sums
3. Third kernel: add scanned block sums to each scanned segment

즉 전체 흐름은:

```text
input
-> scanned block segments + block sums
-> scanned block sums
-> add scanned block sums back
-> final scanned output
```

강의는 arbitrary-length input에 대해서도 같은 구조를 유지하라고 설명한다.

- block 하나는 최대 `2 * blockDim.x` elements 처리
- block sums array를 다시 scan
- 필요하면 그 block sums scan도 다시 계층적으로 분해

Lab 7 README의 요구사항이 바로 이 구조다.

---

## 8. Memory-Bound Nature of Scan

강의는 scan이 본질적으로 **memory-bound** 라고 설명한다.

three-kernel scan 관점에서 대략:

- first kernel: `N` loads + `N` stores
- third kernel: `N` loads + `N` stores

즉 큰 입력에서는 arithmetic보다 memory traffic이 더 큰 제약이 된다.

이 메시지는 중요한 이유가 있다.

- scan은 matmul처럼 compute-heavy가 아님
- 그래서 “연산 수가 적다”보다
- “메모리를 얼마나 덜 건드리느냐”가 중요하다

강의는 three-kernel `scan-scan-add`보다 더 줄일 수 있는 방식도 언급하지만,
Lab 7에서는 표준 three-kernel 계층형 접근이 목표다.

---

## 9. How This Connects to Lab 7

Lab 7에서 직접 필요한 범위는 다음이다.

- segmented scan 개념
- Brent-Kung scan의 upsweep/downsweep 구조
- 한 block이 `2 * blockDim.x` elements를 처리하는 설계
- block sums array 생성
- auxiliary block sums scan
- scanned block sums add-back kernel
- boundary condition에서 identity value `0` 채우기

즉 Lab 7은 18강의 다음 내용을 코드로 옮기는 과제라고 볼 수 있다.

1. block-local work-efficient scan
2. hierarchical scan for arbitrary length
3. scan correctness + boundary-safe handling

---

## 10. Practical Takeaways

1. scan은 reduction의 가까운 친척이지만, output이 하나가 아니라 전체 prefix array다.
2. Kogge-Stone은 latency 친화적이지만 work-efficient하지 않다.
3. Brent-Kung은 two-phase tree로 `O(n)` work를 만든다.
4. large input scan은 block-level scan 하나로 끝나지 않고 three-kernel hierarchy가 필요하다.
5. scan은 memory-bound이므로 global memory traffic을 의식해야 한다.

---

## 11. Why This Matters Later

scan은 standalone lab 주제이기도 하지만, 이후에도 자주 다시 등장한다.

- stream compaction
- radix sort
- histogram post-processing
- graph traversal
- sparse data structure construction

즉 이번 강의는 단순 prefix sum 하나를 배우는 것이 아니라,
GPU parallel algorithm design에서 매우 널리 쓰이는 building block을 배우는 강의다.
