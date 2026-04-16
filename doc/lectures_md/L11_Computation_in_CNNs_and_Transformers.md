# ECE 408 / CS 483 / CSE 408
## Lecture 11: Computation in CNNs and Transformers

**Instructor:** Volodymyr Kindratenko  
**학기:** Spring 2026

> Source: `Lectures_pdf/ece408-lecture11-intro-to-CNN-vk-SP26.pdf`

---

## 1. Lecture Briefing

이번 강의의 큰 흐름은 다음과 같다.

1. **MLP의 한계 복습**
   - 작은 숫자 이미지(MNIST)에서는 MLP가 가능하지만, 이미지가 커지면 fully-connected layer의 파라미터 수와 연산량이 너무 커진다.
2. **왜 CNN이 필요한가**
   - 이미지에는 지역적 구조(local structure)가 있고, 같은 패턴이 여러 위치에 반복되어 나타난다.
   - 따라서 작은 커널을 슬라이딩하며 쓰는 convolution이 더 자연스럽고 효율적이다.
3. **CNN 레이어를 실제로 어떻게 계산하나**
   - convolution layer의 forward pass, pooling layer의 forward pass, GPU에서의 병렬화 방법을 본다.
4. **convolution을 행렬곱으로 바꾸는 시각**
   - `im2col` / unrolling을 통해 convolution을 GEMM(matrix multiplication)으로 변환할 수 있다.
5. **Transformer 계산과의 연결**
   - CNN 이후, 대형 언어모델에서는 self-attention과 MLP가 큰 계산 비용을 차지함을 간단히 소개한다.

### 이 강의의 핵심 요지

- **CNN은 MLP보다 이미지 구조를 더 잘 활용한다.**
- **convolution layer는 채널 합산이 들어간 슬라이딩 dot product다.**
- **GPU 구현에서는 출력 픽셀, 출력 feature map, 입력 채널 축을 어떻게 병렬화할지가 중요하다.**
- **고성능 구현에서는 convolution을 직접 계산할 수도 있고, unrolling 후 matrix multiplication으로 바꿔 계산할 수도 있다.**
- **Transformer도 결국 대규모 matrix multiplication 중심 계산 구조를 가진다.**

---

## 2. Course Reminders and Objectives

### Course reminders

- 이번 주는 lab 마감이 없음
- 그 시간을 프로젝트와 시험 준비에 쓰라고 안내
- Project milestone 1이 곧 배포됨
- Midterm 1:
  - **일시:** 2026년 3월 3일, 오후 7시~10시
  - 충돌이 있으면 수요일까지 연락
  - 리뷰 세션과 예전 시험 예시는 Canvas 참고

### Today’s objectives

- CNN의 여러 레이어를 실제로 구현하는 방법 학습
- CNN Project Milestone 1 간단 소개
- Transformer에 쓰이는 계산 구조 맛보기

---

## 3. MLP Review and Why It Breaks for Large Images

### Slide 4: MLP 복습

MNIST 숫자 인식 예시는 다음과 같다.

- 입력: 28x28 이미지 = 784 pixels
- hidden layer: 10 nodes
- output layer: 10 nodes
- 총 파라미터 수:
  - 첫 레이어: `784*10 + 10`
  - 둘째 레이어: `10*10 + 10`
  - 합계: `7,960`

각 노드는 대체로 다음 형태다.

\[
n_k = activation(\mathbf{w}_k \cdot \mathbf{x} + b_k)
\]

여기서 activation은 `sigmoid`, `sign`, `ReLU` 같은 함수다.

### Slide 5: 큰 이미지에서는 MLP가 너무 비싸다

강의에서는 `250 x 250` 이미지를 예로 든다.

- 입력 픽셀 수: `250^2 = 62,500`
- fully-connected node 하나당 weight 수: `62,500`
- hidden node 수가 비슷한 규모면 총 weight 수는 대략 **수십억 개**

즉,

- 메모리 사용량이 너무 크고
- 연산량도 너무 많고
- 레이어를 더 깊게 쌓으면 더 심각해진다

그래서 전통적인 이미지 처리에서 쓰던 **filter / convolution kernel** 개념을 신경망에 도입한다.

### 직관

MLP는 "모든 입력 픽셀을 모든 출력 노드에 연결"한다.  
반면 CNN은 "작은 지역 패턴만 보고, 그 같은 패턴 탐지기를 이미지 전체에 공유"한다.

---

## 4. Convolution Basics

### Slide 6: 2D convolution

슬라이드에는 입력 `X`, 커널 `W`, 출력 `Y`의 예가 나온다. 핵심은 다음이다.

- 커널 `W`는 작은 크기
- 입력 `X` 위를 슬라이딩
- 각 위치마다 element-wise multiply-accumulate 수행
- 결과를 출력 `Y`의 한 점으로 저장

수식 형태로 쓰면:

\[
Y[h, w] = \sum_{p=0}^{K-1}\sum_{q=0}^{K-1} X[h+p, w+q] \cdot W[p, q]
\]

### 간단한 예시

```text
입력 X 일부:
1 2 3
4 5 6
7 8 9

커널 W:
1 0
0 1
```

좌상단 위치의 출력은:

\[
1\cdot1 + 2\cdot0 + 4\cdot0 + 5\cdot1 = 6
\]

즉 커널 하나가 입력 패치 하나와 dot product를 한다고 보면 된다.

### Slide 7: convolution은 입력 크기가 달라도 자연스럽다

MLP는 입력 길이가 고정되어야 한다.  
반면 convolution은 같은 커널을 더 큰 입력, 더 작은 입력에도 적용할 수 있다.

예:

- 길이가 다른 오디오
- 해상도가 다른 이미지
- 시간 길이가 다른 시계열

이 점이 CNN을 실전 데이터에 강하게 만든다.

### Slide 8: convolution input의 다양한 형태

강의에서 든 예시:

- 1D audio waveform
- skeleton animation의 joint angle 시계열
- 주파수 축 또는 시간 축으로 convolve하는 오디오 표현
- RGB color image
- 3D volumetric data
- color video

즉 convolution은 꼭 2D 이미지 전용이 아니라, **구조적/격자형 데이터** 전반에 적용되는 계산 패턴이다.

---

## 5. CNNs in Computer Vision

### Slide 9: LeNet-5

- 손글씨 문자 인식을 위한 고전적 CNN 예시
- CNN이 실제 비전 문제에 적용된 초기 대표 사례

### Slide 10: Many types of CNNs

- CNN에도 다양한 구조가 존재
- 레이어 수, 필터 크기, pooling 방식, skip connection 여부 등에 따라 계열이 갈린다

### Slide 11: 딥러닝이 컴퓨터 비전에 끼친 영향

- 2012년 Large Scale Visual Recognition Challenge에서 Toronto 팀이 GPU로 120만 장 이미지를 학습
- 이후 computer vision 성능 향상이 크게 가속됨

강의 의도는 분명하다.

- GPU가 단지 "빠른 하드웨어"가 아니라
- **딥러닝의 실제 도약을 가능하게 한 계산 플랫폼**이라는 점을 강조한다

---

## 6. Why Convolution Works Better Than MLP for Images

### Slide 12: Sparse interactions

CNN이 좋은 이유:

- **작은 공간 영역의 특징**만 먼저 본다
- 파라미터 수가 적다
- 저장 공간이 줄고
- 통계적으로도 더 다루기 쉬우며
- 학습도 더 빠르다

단, receptive field가 처음엔 작기 때문에:

- 넓은 문맥을 보려면 여러 레이어를 쌓아야 한다

### Slide 13: Parameter sharing and equivariance

핵심 개념 2개:

1. **Parameter sharing**
   - 같은 커널 mask를 입력 전체 위치에 반복 사용
   - 즉 한 번 학습한 edge detector를 모든 위치에 재사용

2. **Equivariant representation**
   - 입력이 이동하면 출력 feature map도 비슷하게 이동
   - "이 특징이 어디에 있는가"를 지도(map) 형태로 표현

### Slide 14: Convolution vs MLP 비교

#### Convolution

- 입력을 2D 구조로 다룸
- `Y = W ⊗ X`
- 작은 receptive field
- weight 수가 적음

#### MLP

- 입력을 벡터로 펼침
- `Y = w x + b`
- receptive field가 최대
- weight 수가 많음

### 정리

CNN은 MLP보다 표현력이 무조건 크다기보다, **이미지라는 데이터의 구조적 가정에 더 잘 맞는다.**

---

## 7. Anatomy of a Convolution Layer

### Slide 15: convolution layer 구성

- 입력 feature: `A`개, 각각 `N1 x N2`
- convolution kernels: `B`개, 각각 `K1 x K2`
- 출력 feature 수: `B`
- 출력 크기:

\[
(N1-K1+1)\times(N2-K2+1)
\]

슬라이드의 표현은 조금 압축되어 있지만, 실질적으로는:

- 입력 채널 수 = `A`
- 출력 채널 수 = `B`
- 각 출력 채널은 모든 입력 채널을 합쳐서 계산

### Slide 16: channel 개념

입력의 일부 feature들은 서로 관련된 채널 집합이다.

예:

- RGB 이미지의 `R`, `G`, `B`

하나의 출력 feature map을 만들 때는:

- 채널마다 별도의 커널 조각이 있고
- 각 채널 결과를 더해서 최종 출력 한 점을 만든다

즉 multi-channel convolution은 다음처럼 생각하면 된다.

\[
Y[m,h,w]
= \sum_{c=0}^{C-1}\sum_{p=0}^{K-1}\sum_{q=0}^{K-1}
X[c,h+p,w+q]\cdot W[m,c,p,q]
\]

---

## 8. Pooling Layer

### Slide 17: 2D Pooling (Subsampling)

pooling은 convolution output을 더 작게 줄이는 레이어다.

- max pooling
- average pooling
- L2 norm pooling
- weighted average pooling

효과:

- 입력의 작은 이동이나 크기 변화에 덜 민감하게 함
- feature representation을 압축
- 계산량과 메모리 사용량 감소

### average pooling 예시

```text
입력 feature map 일부:
1 3 2 4
5 6 1 2
0 2 7 8
1 3 4 5
```

`2x2 average pooling`을 하면:

- 좌상단 블록 평균: `(1+3+5+6)/4 = 3.75`
- 우상단 블록 평균: `(2+4+1+2)/4 = 2.25`
- 좌하단 블록 평균: `(0+2+1+3)/4 = 1.5`
- 우하단 블록 평균: `(7+8+4+5)/4 = 6.0`

출력:

```text
3.75 2.25
1.50 6.00
```

---

## 9. Forward Propagation in a CNN

### Slide 18: forward propagation의 차원

강의는 다음 텐서 차원을 쓴다.

#### Input `X`

- `B` images
- image당 `C` channels
- channel당 `H x W`

#### Weights `W`

- `M` feature maps
- feature map당 `C` channels
- 각 채널당 `K x K`

#### Output `Y`

- `B` images
- image당 `M` output features
- 출력 크기:

\[
H_{out}=H-K+1,\quad W_{out}=W-K+1
\]

### Slide 19: full kernel이 들어가는 위치만 계산

padding이 없다고 가정하면, 커널이 입력 밖으로 나가면 안 된다.  
그래서 출력은 입력보다 작아진다.

이 강의의 기본 설정은:

- **valid convolution**
- 즉 ghost element 없이, 커널이 완전히 들어가는 위치만 계산

### Slide 20: 작은 예시

예시 설정:

- `B = 1`
- `C = 3`
- 입력 크기 `3x3`
- `M = 2`
- 커널 크기 `2x2`

그러면 출력 크기는:

\[
H_{out}=3-2+1=2,\quad W_{out}=3-2+1=2
\]

즉 각 출력 feature map은 `2x2`, 출력 채널이 2개이므로 전체 출력은 `2 x 2 x 2` 구조다.

---

## 10. Sequential Code for Forward Convolution

### Slide 21: sequential code

```c
void convLayer_forward(int B, int M, int C, int H, int W, int K,
float* X, float* W, float* Y)
{
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    for (int b = 0; b < B; ++b)
    for (int m = 0; m < M; m++)
    for (int h = 0; h < H_out; h++)
    for (int w = 0; w < W_out; w++) {
        Y[b, m, h, w] = 0.0f;
        for (int c = 0; c < C; c++)
        for (int p = 0; p < K; p++)
        for (int q = 0; q < K; q++)
            Y[b, m, h, w] += X[b, c, h + p, w + q] * W[m, c, p, q];
    }
}
```

### 코드 해설

- `b`: 배치의 몇 번째 이미지인가
- `m`: 몇 번째 output feature map인가
- `h, w`: output pixel 위치
- `c`: 입력 채널
- `p, q`: 커널 내부 좌표

즉 출력 원소 하나를 계산할 때:

- 입력 채널 전체를 돈다
- 각 채널에 대해 `K x K` window와 커널을 곱한다
- 모두 더한다

### 연산량 감각

출력 원소 하나당 대략:

- 곱셈 `C*K*K`
- 덧셈도 거의 `C*K*K`

출력 전체는:

\[
B \cdot M \cdot H_{out} \cdot W_{out} \cdot C \cdot K^2
\]

에 비례하는 큰 작업량이 된다.

---

## 11. A Small Convolution Example

### Slides 22-25

강의는 작은 행렬 예시를 여러 슬라이드에 걸쳐 전개하면서,

- 어느 입력 patch가
- 어느 커널과 곱해지고
- 어떤 출력 위치로 가는지

를 시각적으로 보여 준다.

여기서 꼭 기억할 점:

- **출력 한 점 = 입력의 작은 patch와 커널의 inner product**
- **출력 채널 하나 = 모든 입력 채널을 합한 결과**
- **출력 전체 = 이 작업을 모든 위치, 모든 출력 채널에 반복**

---

## 12. Where Is the Parallelism?

### Slide 26: convolution layer의 병렬성

병렬화 가능한 축은 여러 개다.

1. **출력 feature map 간 병렬화**
   - 서로 다른 output map은 독립 계산 가능
   - 하지만 개수가 작을 수 있어 GPU를 다 못 채울 수 있음

2. **출력 feature map 내부 픽셀 간 병렬화**
   - 각 픽셀도 독립 계산 가능
   - 보통 가장 풍부한 병렬성

3. **입력 channel 간 병렬화**
   - 이론상 가능
   - 하지만 같은 출력 원소에 누적해야 해서 atomic 또는 reduction 필요

강의 포인트:

- 레이어마다 shape가 달라 병렬화 전략도 달라질 수 있다

---

## 13. Pooling Forward Code

### Slide 27: scale `N` subsampling

출력 크기:

\[
H_S(N)=\lfloor H_{out}/N \rfloor,\quad
W_S(N)=\lfloor W_{out}/N \rfloor
\]

즉 `N x N` 블록 하나가 pooling output 한 점이 된다.

### Slide 28: sequential pooling code

```c
void poolingLayer_forward(int B, int M, int H_out, int W_out, int N,
float* Y, float* S)
{
    for (int b = 0; b < B; ++b)
    for (int m = 0; m < M; ++m)
    for (int x = 0; x < H_out/N; ++x)
    for (int y = 0; y < W_out/N; ++y) {
        float acc = 0.0f;
        for (int p = 0; p < N; ++p)
        for (int q = 0; q < N; ++q)
            acc += Y[b, m, N*x + p, N*y + q];
        acc /= N * N;
        S[b, m, x, y] = sigmoid(acc + bias[m]);
    }
}
```

### 해설

- pooling은 보통 convolution 결과를 입력으로 받는다
- `N*x + p`, `N*y + q`가 원래 feature map에서 대응되는 블록을 가리킨다
- 강의 예시는 average pooling 뒤에 bias와 sigmoid까지 합친 형태다

### Slide 29: GPU 구현 시 주의점

- 출력 좌표와 이전 layer의 입력 좌표 사이 index mapping이 필요
- pooling은 메모리 대역폭 절약을 위해 convolution과 fused되는 경우도 많다

---

## 14. Designing a Basic CNN CUDA Kernel

### Slide 30: block/grid 매핑 아이디어

기본 아이디어:

- block 하나가
  - 하나의 output feature map 안에서
  - `TILE_WIDTH x TILE_WIDTH` 크기의 output tile을 계산

grid 매핑:

- `grid.x` -> output feature map index `m`
- `grid.y` -> output tile index
- `grid.z` -> batch image index로 확장 가능

### Slide 31: small example

예:

- `M = 4`
- `W_out = H_out = 8`
- `TILE_WIDTH = 4`

그러면 output feature map 하나당 `2 x 2 = 4`개의 tile이 필요하고,
output feature map이 4개이므로 grid가 2차원으로 잘 나뉜다.

### Slides 32-33: host code for grid setup

```c
#define TILE_WIDTH 16
W_grid = W_out / TILE_WIDTH;
H_grid = H_out / TILE_WIDTH;
Y = H_grid * W_grid;
dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
dim3 gridDim(M, Y, 1);
ConvLayerForward_Kernel<<<gridDim, blockDim>>>(...);
```

### 해설

- block 내부의 thread는 tile 안의 output pixel에 대응
- `grid.x = M` 이므로 feature map별 block 묶음이 생긴다
- `grid.y = Y` 는 tile을 linearized한 개수

주의:

- 이 버전은 `W_out`, `H_out`이 `TILE_WIDTH`의 배수라고 가정한 단순화 버전

---

## 15. Basic CUDA Kernel for Convolution

### Slides 34-35

강의의 CPU 코드와 대응되는 GPU 커널은 다음 구조다.

```c
__global__ void ConvLayerForward_Basic_Kernel
(int C, int W_grid, int K, float* X, float* W, float* Y)
{
    int m = blockIdx.x;
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;

    float acc = 0.0f;
    for (int c = 0; c < C; c++) {
        for (int p = 0; p < K; p++)
        for (int q = 0; q < K; q++)
            acc += X[c, h + p, w + q] * W[m, c, p, q];
    }
    Y[m, h, w] = acc;
}
```

### 핵심 mapping

- `m = blockIdx.x`
  - 이 block이 담당하는 output feature map
- `blockIdx.y`
  - 해당 feature map 안의 몇 번째 tile인지
- `threadIdx.(x,y)`
  - tile 내부에서 어떤 output pixel인지

### 왜 단순하지만 비효율적일 수 있나

### Slide 36: observations

- 출력 병렬성은 충분한 경우가 많다
- 하지만 같은 입력 tile이 output feature map마다 반복 로드될 수 있다
- 즉 global memory bandwidth 측면에서 비효율 가능
- 그래도 `X` 차원의 block scheduling이 cache 이점을 줄 수는 있다

---

## 16. Convolution as Matrix Multiplication

### Slide 37: im2col / unrolling 아이디어

convolution을 matrix multiplication으로 바꾸는 대표 방법이 `im2col`이다.

핵심 발상:

- filter들을 행렬 하나로 정리
- 입력 feature map의 각 sliding window를 column으로 펴서 다른 행렬로 정리
- 그러면 convolution이 행렬곱과 같아진다

이 변환의 장점:

- 고도로 최적화된 GEMM 커널 사용 가능
- shared memory tiling 등 기존 matrix multiplication 최적화 재사용 가능

### Slide 38: product matrix의 의미

- 결과 행렬의 각 원소는 output feature map의 한 픽셀에 대응
- 결국 "window와 filter의 inner product"를 matrix multiplication의 `row x column` 내적으로 바꿔 계산하는 셈

### 작은 예시 직관

입력 `X`가 `4x4`, 커널 `K=2`라면:

- 각 출력 위치마다 필요한 입력 patch 크기는 `2x2`
- patch 하나를 길이 4 벡터로 펴기
- 이런 벡터들을 열(column)로 쌓기
- 커널도 길이 4 벡터로 펴기

그러면:

- 커널 행렬 `W_row`
- 입력 unrolled 행렬 `X_col`
- 출력은 `W_row * X_col`

---

## 17. Cost of Unrolling

### Slides 39-41

unrolling은 계산 구조를 단순화하지만, 입력 데이터 복제가 발생한다.

강의 포인트:

- 출력 원소 수는 `H_out * W_out`
- 각 출력 원소는 `K*K`개의 입력 원소를 필요로 함
- 따라서 input unrolled matrix에서는 같은 입력 값이 여러 번 복제될 수 있음

작은 예시에서 강의는:

- `H_out = 2`
- `W_out = 2`
- `K = 2`
- 입력 채널 수 3

일 때 복제 비율을 계산한다.

- 복제된 원소 수: `3 * 2 * 2 * 2 * 2 = 48`
- 원본 입력 원소 수: `3 * 3 * 3 = 27`
- 복제 비율:

\[
48/27 \approx 1.78
\]

즉 matrix multiplication으로 바꾸면 편하지만,
**메모리 사용량과 메모리 트래픽이 증가할 수 있다.**

### Slide 41: 원래 convolution의 메모리 효율

강의는 tiled convolution에서 입력 tile reuse 효과를 분석한다.

입력 tile 크기:

\[
(TILE\_WIDTH + K - 1)^2
\]

반면 필요한 총 접근 수는 원래:

\[
TILE\_WIDTH^2 \cdot K^2
\]

그래서 tile reuse를 통해 memory access를 크게 줄일 수 있다.

---

## 18. How to Build the Unrolled Matrix

### Slide 42: unrolled column의 의미

출력 위치 `(h, w)`에 대응하는 unrolled column index:

\[
w_{unroll} = h \cdot W_{out} + w
\]

즉 출력 feature map 위치를 1차원으로 linearize한다.

### Slide 43: 채널별 section

각 입력 채널 `c`는 unrolled column 안에서 자기 section을 가진다.

- section 시작점: `c * K * K`
- 각 채널 section 길이: `K*K`

### Slide 44: 실제 입력 원소 좌표

출력 `(h, w)`를 계산할 때, 입력 채널 `c`에서 커널 `(p, q)`가 보는 위치는:

\[
(c, h+p, w+q)
\]

### Slide 45: mapping code

```c
int w_unroll = h * W_out + w;
int w_base = c * (K*K);
int h_unroll = w_base + p * K + q;
X_unroll[b, h_unroll, w_unroll] = X[b, c, h + p, w + q];
```

### 해설

- `w_unroll`: 출력 위치 기준 column index
- `h_unroll`: 채널 + 커널 좌표 기준 row index
- 결국 `(c, p, q)` 축을 행 방향으로 펴고, `(h, w)` 축을 열 방향으로 펴는 방식이다

### Slide 46: complete unroll function

```c
void unroll(int B, int C, int H, int W, int K, float* X, float* X_unroll)
{
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    for (int b = 0; b < B; ++b)
    for (int c = 0; c < C; ++c) {
        int w_base = c * (K*K);
        for (int p = 0; p < K; ++p)
        for (int q = 0; q < K; ++q) {
            for (int h = 0; h < H_out; ++h)
            for (int w = 0; w < W_out; ++w) {
                int h_unroll = w_base + p * K + q;
                int w_unroll = h * W_out + w;
                X_unroll[b, h_unroll, w_unroll] = X[b, c, h + p, w + q];
            }
        }
    }
}
```

### Slide 47: project/milestone 방향성

강의는 여러 구현 선택지를 제시한다.

1. Baseline
   - tiled 2D convolution
   - convolution mask는 constant memory 사용
2. Matrix multiplication baseline
   - input unrolling kernel
   - tiled matrix multiplication kernel
3. Built-in unrolling
   - 명시적으로 전체 unrolled matrix를 만들지 않고
   - matrix multiplication tile을 load할 때 필요한 값만 input에서 직접 가져오기
4. More advanced GEMM style
   - register + shared memory tiling 조합

즉 프로젝트 관점에서 보면, 이 강의는 **CNN forward를 어떻게 CUDA 최적화 문제로 바꿀 것인가**를 설명하는 준비 강의다.

---

## 19. Transformer Computation

### Slide 48: Transformer-based language models

강의는 CNN에서 Transformer로 넘어가며, LLM의 계산 흐름을 개괄한다.

- text context
- tokenization
- self-attention layer
- feed-forward layer
- next token prediction

핵심 메시지:

- 현대 모델도 결국 대규모 선형대수 연산의 반복이다

### Slide 49: single layer computational flow

한 attention head 기준 흐름:

1. 입력 `X`에서
2. `Q`, `K`, `V` 생성
3. `softmax(QK^T / sqrt(d_head))`
4. 여기에 `V`를 곱해 attention output 생성
5. 여러 head 결과를 합쳐 projection
6. add & normalize
7. feed-forward network(MLP)

여기서도 핵심 계산은 대부분 **matrix multiplies and other operations**다.

### Slide 50: GPT-3 example

강의 슬라이드의 수치:

- 각 `W`는 `12288 x 128` matrix
- 한 개당 약 **1.5M parameters**
- 16-bit float 기준 약 **3MB**
- 입력 벡터 하나당 약 **6 MFlop**

그리고:

- 이런 행렬이 `96 x 3`개 있어 총 **432M parameters**, 약 **864MB**, 약 **1.7 GFlop per X vector**
- `W_o`는 `12288 x 12288`
  - 약 **150M parameters**
  - 약 **300MB**
  - 약 **600 MFlop per X vector**

강의 결론:

- GPT-3는 self-attention 관련 계산만으로도 엄청난 파라미터 수와 FLOP를 가진다
- 즉 Transformer도 GPU 가속이 필수적인 계산 문제다

---

## 20. Wrap-Up

### Slide 51

- Textbook chapter 16 읽기
- Project checkpoint 1 준비
- Midterm 1 준비

---

## 21. Problem Solving from the Lecture

### Problem 1

**Q.**  
CUDA에서 3D video filtering(convolution)을 수행한다.  
mask 크기는 `5x5x7`, constant memory에 저장되어 있다.  
shared memory에는 `16x16x16` output tile을 만들기 위해 필요한 input tile 전체를 저장한다.  
interior tile만 고려할 때, 한 output tile에 대해  
**총 global memory read operations : shared memory accesses** 비율은?

강의 슬라이드의 답 틀:

- global reads: `20 * 20 * 22`
- shared memory accesses: `5 * 5 * 7 * 16 * 16 * 16`

#### 왜 input tile이 `20 x 20 x 22`인가?

- output tile이 `16 x 16 x 16`
- mask가 `5 x 5 x 7`
- valid convolution에서 필요한 input tile 크기:
  - `16 + 5 - 1 = 20`
  - `16 + 5 - 1 = 20`
  - `16 + 7 - 1 = 22`

#### 계산

\[
\frac{20 \cdot 20 \cdot 22}{5 \cdot 5 \cdot 7 \cdot 16 \cdot 16 \cdot 16}
= \frac{8800}{716800}
= \frac{11}{896}
\approx 0.01228
\]

즉,

- 비율은 **`11/896`**
- 대략 **`0.0123 : 1`**

해석:

- shared memory 안에서는 엄청나게 많이 재사용되고
- global memory read는 그에 비해 매우 적다
- 즉 shared memory tiling이 매우 큰 이득을 준다

### Problem 2

**Q.**  
입력이 `100x200` RGB 이미지(`3` channels)이고,  
첫 번째 convolution layer가 `9x9` filter를 사용해 `10`개의 output feature map을 만든다.  
pooling, activation 등은 무시하고 **convolution만** 고려할 때,  
한 번의 forward pass에 필요한 floating-point operations(곱셈 + 덧셈)는?

슬라이드의 식:

```text
10
* 92*192
* 9*9
* 3
* 2
```

#### 왜 출력 크기가 `92 x 192`인가?

- valid convolution
- 높이: `100 - 9 + 1 = 92`
- 너비: `200 - 9 + 1 = 192`

#### 계산

\[
10 \cdot 92 \cdot 192 \cdot 9 \cdot 9 \cdot 3 \cdot 2
= 85,847,040
\]

정답:

- **85,847,040 FLOPs**

#### 해설

출력 원소 하나당:

- `9*9*3 = 243`개의 곱셈
- 비슷한 수의 덧셈
- 강의는 이를 단순히 `*2`로 계산

출력 원소 수:

- output map 하나당 `92*192`
- 그런 map이 10개

---

## 22. 시험/프로젝트 관점에서 꼭 기억할 포인트

### 개념 체크

- CNN은 local connectivity와 parameter sharing을 사용한다
- multi-channel convolution에서는 채널별 결과를 합산한다
- pooling은 크기를 줄이고 작은 변형에 대한 민감도를 낮춘다
- valid convolution에서는 출력 크기가 줄어든다

### 구현 체크

- 기본 CUDA 구현에서는 output pixel 단위 병렬화가 가장 자연스럽다
- feature map 축, tile 축, batch 축을 grid에 어떻게 배치할지 중요하다
- direct convolution은 직관적이지만 input reuse 최적화가 핵심이다
- im2col + GEMM은 구현과 최적화 관점에서 강력하지만 입력 복제가 생긴다

### 큰 그림

- CNN과 Transformer 모두 본질적으로는 **대규모 선형대수 계산**
- 따라서 GPU에서의 memory hierarchy, tiling, mapping 전략이 성능을 좌우한다

