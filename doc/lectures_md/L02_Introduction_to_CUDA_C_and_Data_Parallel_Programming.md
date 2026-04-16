# ECE 408 / CS 483 / CSE 408
## Lecture 2: Introduction to CUDA C and Data Parallel Programming

**Instructor:** Volodymyr Kindratenko  
**학기:** Spring 2026

---

## 1. Course Reminders

### Lab 0

- **마감:** 금요일 8:00 PM (US Central)
- **목적:**
  - 개발 환경 익숙해지기
  - 제출 워크플로 이해
- 성적에는 포함되지 않음
- 마감 이후 제출해도 됨 (하지만 반드시 제출해야 함)

### Delta Access

- ACCESS 계정 필요
- **프로젝트 이름:** CIS240004: Applied Parallel Programming Course
- ACCESS ID를 제출하지 않으면 Delta 계정 생성 불가

---

## 2. Today's Objectives

- 데이터 병렬 컴퓨팅(Data Parallel Computing)의 기본 개념 이해
- CUDA C 프로그래밍 인터페이스의 기본 요소 학습

---

## 3. What is a Thread?

### Thread란?

- 실행 중인 프로그램의 기본 실행 단위
- **구성 요소:**
  - Program Counter (PC)
  - Registers
  - Memory Context

### Multiple Threads

- 하나의 프로그램에서 다수의 스레드가 동시에 실행 가능
- GPU에서는 수천~수십만 개의 스레드 사용

---

## 4. Types of Parallelism

### Task Parallelism

- 서로 다른 작업을 병렬로 수행
- 병렬성 규모가 비교적 작음
- CPU에 적합

### Data Parallelism

- 같은 연산을 다른 데이터에 적용
- 대규모 병렬성 가능
- GPU에 가장 적합

---

## 5. Data Parallel Example

### Example: Image → Grayscale

```c
for each pixel {
    pixel = gsConvert(pixel);
}
```

- 각 픽셀은 서로 독립
- 이상적인 데이터 병렬 구조

---

## 6. CUDA Execution Model

### Heterogeneous Computing

- Host (CPU) + Device (GPU) 구조

### Host Code

- 직렬 또는 소규모 병렬 작업
- 커널 실행 및 메모리 관리

### Device Code (Kernel)

- 대규모 병렬 연산 수행
- SPMD (Single Program, Multiple Data) 모델

---

## 7. Kernel Execution Model

```c
KernelA<<<numBlocks, numThreads>>>(args);
```

- 커널은 Grid 단위로 실행
- Grid = Thread Blocks의 집합
- 모든 스레드는 같은 커널 코드 실행

---

## 8. Thread Indexing

**기본 인덱싱**

```c
int i = threadIdx.x;
```

**Block + Thread 인덱싱**

```c
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

---

## 9. Thread Blocks

**같은 block 내 스레드:**

- shared memory 사용 가능
- barrier synchronization 가능

**block 간 스레드:**

- 직접적인 협력 어려움
- 확장성 (Scalability) 확보의 핵심 단위

---

## 10. Vector Addition – "Hello World" of CUDA

### 개념

```c
C[i] = A[i] + B[i]
```

### CPU 코드

```c
for (int i = 0; i < n; i++) {
    C[i] = A[i] + B[i];
}
```

---

## 11. System Organization

| CPU (Host)     | GPU (Device)   |
|----------------|----------------|
| Host Memory    | Device Memory  |

- CPU와 GPU는 서로의 메모리에 직접 접근 불가
- 명시적 데이터 복사 필요

---

## 12. CUDA Memory Management Flow

1. GPU 메모리 할당 (cudaMalloc)
2. Host → Device 복사
3. GPU 커널 실행
4. Device → Host 복사
5. GPU 메모리 해제 (cudaFree)

---

## 13. CUDA Memory API

### Memory Allocation

```c
cudaMalloc(void **devPtr, size_t size);
cudaFree(void *devPtr);
```

### Memory Copy

```c
cudaMemcpy(dst, src, size, cudaMemcpyKind);
```

- `cudaMemcpyHostToDevice`
- `cudaMemcpyDeviceToHost`
- `cudaMemcpyDeviceToDevice`

---

## 14. Kernel Launch Configuration

```c
vecAddKernel<<<numBlocks, numThreadsPerBlock>>>(A_d, B_d, C_d, n);
```

**Grid Size 계산**

```c
numBlocks = (n + blockSize - 1) / blockSize;
```

---

## 15. Kernel Implementation Example

```c
__global__
void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}
```

---

## 16. Function Qualifiers in CUDA

| Keyword            | 호출 위치 | 실행 위치 |
|--------------------|-----------|-----------|
| `__host__`         | Host      | Host      |
| `__global__`       | Host      | Device    |
| `__device__`       | Device    | Device    |

**`__host__` `__device__`**

```c
float f(float a, float b) {
    return a + b;
}
```

---

## 17. Compilation Flow

```
.cu
 ↓
NVCC
 ↓
Host Code (CPU) + PTX
 ↓
JIT Compilation
 ↓
GPU Assembly (SASS)
```

---

## 18. Asynchronous Kernel Calls

- 커널 호출은 기본적으로 비동기
- CPU와 GPU 작업 겹치기 가능

**동기화**

```c
cudaDeviceSynchronize();
```

---

## 19. Error Checking

```c
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("%s\n", cudaGetErrorString(err));
}
```

---

## 20. Problem Solving Examples

### Example 1

```c
kernel<<<VECTOR_N, ELEMENT_N>>>();
```

- Threads per block: ELEMENT_N
- Total threads: VECTOR_N * ELEMENT_N

### Example 2

- Vector length = 16000
- Thread computes 8 elements
- Block size = 256
- Required threads = 16000 / 8 = 2000
- Blocks = ceil(2000 / 256) = 8
- Total threads = 2048

---

## 21. Summary

- CUDA는 대규모 데이터 병렬성을 위한 프로그래밍 모델
- **핵심 개념:**
  - Thread / Block / Grid
  - Explicit memory management
  - SPMD execution
- **다음 강의:**
  - Multidimensional Grids
  - 이미지, 행렬, 텐서 처리

---

## 22. To Do

- Textbook Chapter 2
- CUDA Programming Guide – Programming Model
- Delta 계정 확인
- GitHub setup
- Lab 0 제출
