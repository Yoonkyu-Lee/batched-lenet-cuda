# ECE 408 / CS 483 / CSE 408
## Lecture 10: Introduction to ML; Inference and Training in DNNs

**Instructor:** Volodymyr Kindratenko  
**학기:** Spring 2026

> Source: ece408-lecture10-intro-to-ml-vk-SP26.pdf

---

## 1. Course Reminders

### Lab updates

- Lab 4 is due this week on Friday
- Project milestone 1 will be released soon

### Midterm 1

- **When:** March 3rd, 7–10pm

---

## 2. Today's Objectives

- Feedforward neural network의 기본 접근 학습:
  - Neural model
  - Common functions
  - Gradient descent를 통한 training

---

## 3. What is Machine Learning?

- **Machine learning:** 동작 논리가 완전히 규명되지 않은 응용을 만드는 중요한 방법
- **방식:** 예시(labeled data, 입력–출력 쌍)로 원하는 관계를 나타내고, 프로그램 논리를 반복 조정해 원하는/근사한 답을 얻음 → **training**
- **학습 태스크 유형:**
  - Classification, Regression ← structured data
  - Transcription, Translation ← unstructured data
  - 기타

---

## 4. Why Machine Learning Now?

- **Computing:** GPU와 CUDA 등으로 딥뉴럴넷 학습 사이클이 매우 빨라짐
- **Data:** 저렴한 센서, 클라우드, IoT, 사진 공유 등으로 데이터 풍부
- **Needs:** 자율주행, 스마트 기기, 보안, 기술 수용, 헬스케어 등

---

## 5. Evolution of ML

- **Rule-based:** Hand-designed program, 입력 → 출력
- **Classic ML:** Hand-designed features, 입력 → features → 출력
- **Deep Learning:** Layers of features, 입력 → 추상 features → 출력

---

## 6. Classification

- **분류 문제:** 입력 벡터를 C개 범주로 매핑하는 함수 \(F : \mathbb{R}^N \to \{1, \ldots, C\}\)를 모델링 (F는 미지)
- F를 **가중치 집합 \(\Theta\)**로 파라미터화된 함수 \(f\)로 근사
- 범주 \(i\)에 대해: \(\text{prob}(i) = f(x, \Theta)\)
- (입력: N차원 실수 벡터, 출력: 범주를 나타내는 정수)

---

## 7. Linear Classifier (Perceptron)

- **형식:** \(y = f(x, \Theta)\), \(\Theta = \{W, b\}\)
- \(y = \text{sign}(W \cdot x + b)\)
- Dot product + scalar addition: output = weight · input + bias

---

## 8. Can We Learn XOR with a Perceptron?

- XOR는 AND, OR의 선형 조합으로는 불가능 (한 직선으로 나눌 수 없음)

---

## 9. Multiple Layers Solve More Problems

- 입력 차원이 AND, OR이면 한 직선으로 나눌 수 있음
- AND와 OR의 조합만으로는 XOR 같은 경우를 얻기 어려움 → 여러 레이어 필요

---

## 10. Perceptron: AND, OR, XOR Table

- \(x[0] + x[1] - 1.5 > 0\) (AND), \(x[0] + x[1] - 0.5 > 0\) (OR) 등
- (x[1], x[0]) = (0,0), (0,1), (1,0), (1,1)에 대해 AND, OR, XOR 결과 표
- **XOR는 AND, OR의 선형 조합이 아님**

---

## 11. Multi-Layer Perceptron (XOR)

- OR = sign(x[0] + x[1] - 0.5), AND = sign(x[0] + x[1] - 1.5)
- XOR = sign(2·OR + (-1)·AND - 2)
- sign()이 비선형성을 넣어 다음 레이어를 위해 데이터를 “재배치”
- (x[0], x[1])에 대한 AND, OR, XOR 진리표 일치

---

## 12. Multi-Layer Perceptron – Data Repositioning

- OR, AND 레이어 출력을 다시 선형 결합해 XOR 생성
- XOR = sign(2*OR + (-1)*AND - 2)

---

## 13. Fully-Connected Layer

- **Linear classifier:** 입력 벡터 x × weight 벡터 w → scalar y
- **Fully-connected:** 입력 벡터 x × weight **matrix** W → **vector** y (y[0], y[1], …)

---

## 14. Multilayer Terminology

- **Input layer** → **Hidden layer(s)** → **Output layer**
- \(W_k[i,j]\): k번째 레이어의 i번째 입력과 j번째 출력 사이 weight
- 예: W₁ [4×4], b₁ [4×1]; W₂ [4×3], b₂ [3×1]
- 출력: k[0], k[1], k[2] = 각 클래스일 확률 → **Argmax** → 최종 y

---

## 15. How to Determine the Weights?

- 관측 데이터로 가중치를 정할 수 있는가?
- 무작위? 부분적으로 맞는 초기값?
- **Labeled data가 충분하면** 입력–출력 관계를 자동으로 인코딩 가능

---

## 16. Forward and Backward Propagation

- **Forward (inference):** 파라미터 \(\Theta\)와 입력 \(x\)가 주어지면 레이블 \(y\) 계산
- **Backward (training):**
  - 정확도를 재는 방법 필요 → **loss function** (예: \((x-y)^2\))
  - 모든 입력 데이터에 대해 loss를 최소화하는 \(\Theta\)를 찾음

---

## 17. Forward Propagation (Inference)

- (Figure: Forward: x → … → y)

---

## 18. Backward Propagation (Training)

- (Figure: Forward x → …; Backward: dE/dW 등; 전체 학습 집합에 대해 loss 최소화)

---

## 19. Example: Digit Recognition

- **태스크:** 28×28 그레이스케일 이미지 → 0~9 숫자
- **데이터:** 60,000장, 각각 사람이 레이블 → **MNIST dataset**

---

## 20. Multi-Layer Perceptron (MLP) for Digit Recognition

- **구조:**
  - Input (L0): 784 nodes (28×28)
  - Hidden (L1): 10 nodes
  - Output (L2): 10 nodes (Digit 0 ~ 9)
- **파라미터:** L1: 784×10 weights + 10 biases; L2: 10×10 weights + 10 biases → 총 **7,960**
- 각 노드: \(n_k = \text{activation}(w_k \cdot x + b_k)\)
- Activation: Sigmoid, sign, ReLU 등

---

## 21. How Do We Determine the Weights?

- 첫 레이어: 784 입력, 10 출력, fully connected → [10×784] W, [10×1] b
- Labeled training data로 가중치 선택
- **아이디어:** labeled 데이터가 충분하면 입력–출력 함수를 근사할 수 있음

---

## 22. Forward and Backward Propagation (Summary)

- **Forward (inference):** 입력 x(예: 이미지), 파라미터 \(\Theta\)(각 레이어 W, b)로 확률 k[i] 계산
- **Backward (training):** 입력 x, \(\Theta\), 출력 k[i], target label t로 error E 계산 → E에 비례해 \(\Theta\) 조정

---

## 23. Neural Functions Impact Training

- Perceptron: \(y = \text{sign}(W \cdot x + b)\)
- Error를 뒤로 전파하려면 **chain rule** 필요
- **Smooth 함수**가 유리; **sign**은 smooth하지 않음

---

## 24. Sigmoid / Logistic Function

- ~2017년까지 가장 흔한 선택: **sigmoid (logistic)** \(f : \mathbb{R} \to (0,1)\)
\[
f(x) = \frac{1}{1 + e^{-x}}
\]
- 도함수:
\[
\frac{df(x)}{dx} = \frac{e^{-x}}{(1+e^{-x})^2} = f(x)(1 - f(x))
\]

---

## 25. ReLU (Today’s Choice)

- 2017년 이후 흔한 선택: **ReLU** \(f : \mathbb{R} \to \mathbb{R}_+\)
\[
f(x) = \max(0, x)
\]
- 지수 연산 없어 더 빠름
- **Smooth 근사:** softplus / SmoothReLU \(f(x) = \ln(1 + e^x)\) (logistic의 적분)
- 다양한 변형 존재 (Wikipedia 등 참고)

---

## 26. Softmax for Probabilities

- Sigmoid/ReLU만으로는 확률 벡터를 만들 수 없음
- 출력 벡터 \(Z = (z[0], \ldots, z[C-1])\)에 대해 **softmax**로 확률 벡터 \(K = (k[0], \ldots, k[C-1])\) 생성:
\[
k[i] = \frac{e^{z[i]}}{\sum_{j=0}^{C-1} e^{z[j]}}
\]
- \(\sum_i k[i] = 1\)

---

## 27. Softmax Derivatives (for Training)

\[
\frac{dk[i]}{dz[m]} = k[i]\,(\delta_{im} - k[m])
\]
- \(\delta_{im}\): Kronecker delta (i=m이면 1, 아니면 0)

---

## 28. Forward and Backward (Diagram)

- (Figure: Forward; Backward dE/dk 등)

---

## 29. Choosing an Error Function

- 여러 선택 가능
- **예 1:** 레이블 T에 대해 \(E = 1 - k[T]\) (T로 분류되지 않을 확률)
- **예 2 (수치 범주):** 이차 손실
\[
E = \sum_{j=0}^{C-1} (k[j] - T)^2
\]
- 강의에서는 후자 사용

---

## 30. Stochastic Gradient Descent (SGD)

- 가중치 계산의 한 방법: **Stochastic Gradient Descent**
1. 모든 학습 입력에 대한 error E의 합의 **도함수**를 모든 파라미터 \(\Theta\)에 대해 계산
2. E를 줄이는 방향으로 \(\Theta\)를 조금 갱신
3. 반복

---

## 31. Stochastic Gradient Descent (Steps)

- 각 입력 X에 대해:
1. Forward로 k[i] 계산
2. k[i]와 target T로 error E 계산
3. Backpropagation으로 각 파라미터에 대한 도함수 계산
4. \(\Theta_{i+1} = \Theta_i - \varepsilon \Delta\Theta\) 로 갱신

---

## 32. Parameter Updates and Propagation

- \(fc_1 = W_1 \cdot x + b_1\)
- Weight update에 필요: **Backward에서 전파된 error gradient** (\(dE/dfc_1\)), **Forward에서 온 입력** (x)
\[
\frac{dE}{dW_1} = \frac{dE}{dfc_1} \cdot \frac{dfc_1}{dW_1} = \frac{dE}{dfc_1} \cdot x
\]

---

## 33. Example: Gradient Update with One Layer

- **Parameter update:** \(\Theta_{i+1} = \Theta_i - \varepsilon \Delta\Theta\), \(W_{i+1} = W_i - \varepsilon \Delta W\)
- **Network:** \(y = W \cdot x + b\) → \(\frac{dy}{dW} = x\)
- **Error:** \(E = \frac{1}{2}(y - t)^2\) → \(\frac{dE}{dy} = y - t = Wx + b - t\)
- **Weight update:** \(\Delta W = \frac{dE}{dW} = \frac{dE}{dy}\frac{dy}{dW}\) → \(W_{i+1} = W_i - \varepsilon(Wx + b - t)x\)

---

## 34. Fully-Connected Gradient Detail

- \(fc_1[i] = \sum_j W_1[i,j]\, x_1[j]\)
\[
\frac{dE}{dW_1[i,j]} = \frac{dE}{dfc_1[i]} \cdot \frac{dfc_1[i]}{dW_1[i,j]} = \frac{dE}{dfc_1[i]} \cdot x_1[j]
\]
- 이 레이어로 들어오는 **입력** \(x_1[j]\)가 필요 (forward에서 제공)

---

## 35. Batched Stochastic Gradient Descent

- **한 epoch:** 전체 학습 집합을 한 번 도는 것
- \(\Delta\Theta = 0\)으로 초기화
- 각 labeled 이미지에 대해: 입력 로드 → forward → E 계산 → backprop → \(\Delta\Theta\)에 누적
- \(\Theta_{i+1} = \Theta_i - \varepsilon \Delta\Theta\)
- 전체 gradient를 모아서 갱신하면 진짜 gradient에 가장 가깝게 반영

---

## 36. Mini-batch Stochastic Gradient Descent

- 학습 집합을 **batch** 단위로 나눔
- 각 batch 안의 각 이미지에 대해: 입력 로드 → forward → E → backprop → \(\Delta\Theta\)에 누적
- \(\Theta_{i+1} = \Theta_i - \varepsilon \Delta\Theta\)
- Gradient 추정 정확도와 **병렬성** 사이의 균형

---

## 37. When is Training Done?

- Labeled data를 **training set**과 **test set**으로 나눔
- Training: 파라미터 갱신
- Test: **새 입력에 대한 일반화**가 얼마나 되는지 확인 (궁극 목표)
- 학습 데이터에 지나치게 맞추면 일반화가 나빠질 수 있음

---

## 38. How Complicated Should a Network Be?

- 직관: 다항식 피팅과 비슷
- 고차 항은 피팅을 좋게 하지만, 학습 구간 밖 입력에서 예측 불가한 진동을 만들 수 있음

---

## 39. Overtraining Decreases Accuracy

- 학습 데이터에 너무 잘 맞추면, 새 입력에 대해 출력이 크고 예측하기 어려운 변화를 보일 수 있음

---

## 40. Visualizing Neural Network Weights

- MNIST 1st layer, 2nd layer weight 시각화
- (Reference: https://ml4a.github.io/ml4a/looking_inside_neural_nets/)

---

## 41. No Free Lunch Theorem

- **모든** 분류 알고리즘은, 가능한 모든 입력 생성 분포에 대해 평균하면, **이전에 본 적 없는 입력**에 대한 error rate가 동일하다.
- 뉴럴넷도 **특정 태스크에 맞게 튜닝**해야 함

---

## 42. Things to Read / Things to Do

### Things to Read

- Textbook Chapter 16

### Things to Do

- Submit Lab 4
