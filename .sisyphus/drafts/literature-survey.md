# PINN4CSI 문헌 조사 (Literature Survey)

> **목적**: Physics-Informed Neural Networks (PINNs)를 WiFi CSI 분석에 적용하는 연구의 최신 동향 파악 및 연구 기회 발굴
> **범위**: 2023–2025년 top conference/journal 논문 중심
> **조사일**: 2026-03-12

---

## 목차

1. [PINN 방법론 최신 동향](#1-pinn-방법론-최신-동향)
2. [WiFi CSI 센싱 최신 연구](#2-wifi-csi-센싱-최신-연구)
3. [PINN × 무선통신 교차 연구](#3-pinn--무선통신-교차-연구)
4. [NeRF/3DGS 기반 무선 채널 모델링](#4-nerf3dgs-기반-무선-채널-모델링)
5. [PINN에 활용 가능한 물리 모델](#5-pinn에-활용-가능한-물리-모델)
6. [연구 공백 및 기회 (Research Gaps)](#6-연구-공백-및-기회)
7. [제안 연구 방향](#7-제안-연구-방향)

---

## 1. PINN 방법론 최신 동향

> Top ML 학회 (NeurIPS, ICML, ICLR) 중심의 PINN 핵심 기법 발전

### 1.1 벤치마크 및 체계적 평가

| 논문 | 학회 | 연도 | 핵심 기여 |
|------|------|------|-----------|
| **PINNacle: A Comprehensive Benchmark of Physics-Informed Neural Networks for Solving PDEs** (Hao Zhongkai et al.) | NeurIPS | 2024 | 다양한 PDE에 대한 PINN 성능을 체계적으로 벤치마킹. 10+개 PINN 변형을 20+개 PDE 문제에서 비교 평가 |
| **Physics-Informed Neural Networks and Neural Operators for Parametric PDEs** (Zhang et al., NUDT) | arXiv (Under Review) | 2025 | PINN과 Neural Operator (DeepONet, FNO)의 포괄적 비교 서베이. parametric PDE 문제에서의 성능 체계 분석 |

### 1.2 아키텍처 혁신

| 논문 | 학회 | 연도 | 핵심 기여 |
|------|------|------|-----------|
| **CoPINN: Cognitive Physics-Informed Neural Networks** (Siyuan Duan et al.) | ICML | 2025 | 인지적 구조를 도입한 PINN으로, 학습 과정에서 물리 법칙 위반 영역을 자동 탐지하고 집중 학습 |
| **PINNs with Learnable Quadrature** (Sourav Pal et al.) | NeurIPS | 2025 | 수치 적분을 학습 가능하게 만들어 PINN의 물리 loss 계산 정확도를 동적으로 향상 |
| **Physics-Informed Neural Networks with Fourier Features and Attention-Driven Decoding** (Rohan Arni, Carlos Blanco) | NeurIPS Workshop | 2025 | Fourier feature embedding과 attention 기반 디코딩으로 multi-scale 문제 해결력 향상 |
| **Physics-Informed Neural Networks with Architectural Physics Embedding for Large-Scale Wave Field Reconstruction** (Zhang, Ye, Ma) | arXiv | 2025 | 건축물 물리 정보를 네트워크 아키텍처에 직접 내장하여 대규모 파동장 복원 |

### 1.3 Neural Operator (연산자 학습)

| 논문 | 학회 | 연도 | 핵심 기여 |
|------|------|------|-----------|
| **Derivative-enhanced Deep Operator Network (DE-DeepONet)** (Yuan Qiu et al.) | NeurIPS | 2024 | DeepONet에 미분 정보를 결합하여 적은 학습 데이터로도 정확한 PDE 솔루션 예측. 차원 축소 기법으로 학습 비용 절감 |
| **Physics-Informed Time-Integrated DeepONet** (Mandl et al., Johns Hopkins) | arXiv | 2025 | 시간 의존 PDE에 대한 장시간 추론을 위한 temporal tangent space operator learning |
| **DeepONet for Solving Nonlinear PDEs with Physics-Informed Training** (Yangyyang, Georgia Tech) | arXiv | 2025 | 비선형 PDE에 대한 물리 정보 학습 기반 DeepONet. Branch/Trunk 네트워크 설계의 이론적 분석 |
| **Towards Universal Neural Operators through Multiphysics Pretraining** (Masliaev et al., ITMO) | NeurIPS Workshop | 2025 | transformer 기반 neural operator의 다물리 사전학습 → 전이학습 가능성 입증 |
| **Active Learning for Neural PDE Solvers** (Musekamp et al.) | ICLR | 2025 | Neural PDE Solver를 위한 능동 학습 전략. 효율적 학습 데이터 선택으로 성능 향상 |

### 1.4 학습 안정성 및 견고성

| 논문 | 학회 | 연도 | 핵심 기여 |
|------|------|------|-----------|
| **Efficient Error Certification for Physics-Informed Neural Networks** (Francisco Eiras et al.) | ICML | 2024 | PINN의 오류 상한을 효율적으로 인증하는 방법. 안전-critical 응용에 필수적 |
| **Trustworthy Few-Shot Learning for Scientific Computing: Meta-Learning PINNs with Reliability Guarantees** (Brandon Yee et al.) | NeurIPS | 2025 | 소수 샘플 meta-learning + 신뢰성 보장을 결합한 PINN. 데이터 부족 상황에 적합 |
| **Universal Physics-Informed Neural Networks: Symbolic Differential Operator Discovery with Sparse Data** (Lena Podina et al.) | ICML | 2023 | 희소 데이터로부터 미분 연산자를 자동 발견하는 범용 PINN 프레임워크 |

---

## 2. WiFi CSI 센싱 최신 연구

> Top 시스템/네트워킹 학회 (MobiCom, SenSys, NSDI, UbiComp/IMWUT, INFOCOM) 중심

### 2.1 실내 측위 (Indoor Localization)

| 논문 | 학회 | 연도 | 핵심 기여 |
|------|------|------|-----------|
| **Lessons from Deploying Learning-based CSI Localization on a Large-Scale ISAC Platform** (Zhang et al.) | arXiv | 2025 | 400+ AP 대규모 실환경 CSI 측위 시스템 배치 경험. 미라벨 데이터 활용 및 CSI 이질성 문제 해결 |
| **SiFi: Siamese Networks Based CSI Fingerprint Indoor Localization** | IEEE | 2025 | Siamese Network 기반 CSI 핑거프린트 매칭으로 환경 변화에 강건한 측위 |
| **Wi-Fi CSI fingerprinting-based indoor positioning using deep learning and vector embedding for temporal stability** (Reyes et al.) | Expert Syst. App. | 2024 | 벡터 임베딩으로 시간적 안정성을 확보한 CSI 핑거프린트 측위 |
| **OpenPose-Inspired Reduced-Complexity CSI-Based Wi-Fi Indoor Localization** | IEEE | 2024 | 2D IFFT + heatmap 변환 기반 경량 DL 모델. 90th percentile 정확도 72% 향상, 모델 크기 90% 이상 축소 |

### 2.2 활동 인식 (Activity Recognition)

| 논문 | 학회 | 연도 | 핵심 기여 |
|------|------|------|-----------|
| **WiTS: Wi-Fi-Based Human Action Recognition via Spatio-Temporal Hybrid Neural Network** | arXiv | 2025 | 시공간 하이브리드 신경망으로 WiFi 기반 인간 행동 인식 |
| **CSI2Depth: Spatio-Temporal Depth Images from Wi-Fi CSI Data via Transformer Networks and CGANs** | arXiv | 2025 | CSI → 깊이 영상 변환. Transformer + CGAN으로 WiFi 센싱의 새로운 표현 제안 |
| **Decision Fusion-Based Deep Learning for CSI Channel-Aware Human Action Recognition** | Sensors | 2025 | 결정 융합 기반 다중 채널 CSI 활동 인식 |
| **Real-Time, Non-Intrusive Fall Detection via Wi-Fi CSI** | arXiv | 2025 | CNN-LSTM, GNN, Transformer 3가지 아키텍처 비교. 실시간 낙상 감지 |

### 2.3 도메인 일반화 및 확장성

| 논문 | 학회 | 연도 | 핵심 기여 |
|------|------|------|-----------|
| **A Survey on Wi-Fi Sensing Generalizability: Taxonomy, Techniques, Datasets, and Future Research Prospects** (Wang et al., Xi'an Jiaotong Univ.) | IEEE COMST (accepted) | 2025/2026 | WiFi 센싱의 일반화 문제에 대한 포괄적 서베이. 분류 체계, 기법, 데이터셋 정리. SDP8.org 데이터셋 플랫폼 구축 |
| **Time matters: Empirical insights into the limits and challenges of temporal generalization in CSI-based Wi-Fi sensing** (Brunello et al.) | Internet of Things | 2025 | CSI 기반 WiFi 센싱의 시간적 일반화 한계 실증 분석 |
| **CSI-Bench: A Large-Scale In-the-Wild Dataset for Multi-task WiFi Sensing** | arXiv | 2025 | 대규모 실환경 WiFi 센싱 벤치마크 데이터셋 |
| **A Survey on CSI-based Wi-Fi Sensing Datasets and Models with a Focus on Reproducibility** (Guarino et al.) | Comput. Commun. | 2026 | 재현성 중심의 CSI WiFi 센싱 데이터셋/모델 서베이 |

### 2.4 새로운 센싱 기법

| 논문 | 학회 | 연도 | 핵심 기여 |
|------|------|------|-----------|
| **BFMSense: WiFi Sensing Using Beamforming Feedback Matrix** | NSDI | 2024 | CSI 대신 Beamforming Feedback Matrix (BFM) 활용. 상용 WiFi에서 즉시 사용 가능한 새로운 센싱 신호 소스 |
| **UniFi: Combining Irregularly Sampled CSI from Diverse Communication Packets and Frequency Bands** | arXiv | 2025 | 이종 패킷/주파수 대역의 불규칙 CSI 데이터 통합 프레임워크 |
| **A Short Overview of Multi-Modal Wi-Fi Sensing** | arXiv | 2025 | 다중 모달 WiFi 센싱 개요 (CSI + RSSI + BFM 등) |
| **Deep Learning-Enhanced Human Sensing with Channel State Information: A Survey** (Yue et al., UESTC) | CMC | 2025 | CSI 기반 딥러닝 인간 센싱 포괄적 서베이 |

---

## 3. PINN × 무선통신 교차 연구

> **이 영역이 핵심 연구 프런티어** — 빠르게 성장하지만 WiFi CSI 적용은 아직 미개척

### 3.1 PINN 기반 무선 채널 추정

| 논문 | 학회 | 연도 | 핵심 기여 | 물리 모델 |
|------|------|------|-----------|-----------|
| **Physics-Informed Neural Networks for Wireless Channel Estimation with Limited Pilot Signals** (Javid & González-Prelcic, UCSD) | NeurIPS Workshop (AI4NextG) | 2025 | Ray-traced RSS prior를 transformer cross-attention으로 주입. 제한된 파일럿으로 wideband MIMO 채널 복원, 기존 대비 5–15 dB NMSE 개선 | Ray tracing RSS prior |
| **Environment-Aware MIMO Channel Estimation in Pilot-Constrained Upper Mid-Band Systems** (Javid & González-Prelcic) | arXiv | 2025 | 위 논문의 확장. 환경 인식 MIMO 채널 추정으로 파일럿 제약 환경 성능 극대화 | 전파 환경 모델 |
| **Physics-Informed Generative Modeling of Wireless Channels** (Böck et al.) | ICML | 2025 | 물리 정보 생성 모델로 site-specific 무선 채널 분포 학습. 물리적 해석 가능성과 일반화 동시 달성 | EM 전파 모델 |
| **Physics-Informed Generative Approaches for Wireless Channel Modeling** (Wagle et al., Purdue) | arXiv | 2025 | 물리 기반 생성 모델의 무선 채널 모델링 적용. 학습 데이터 품질/일반화/해석 가능성 문제 해결 | 채널 물리 모델 |

### 3.2 PINN 기반 Radio Environment Mapping

| 논문 | 학회 | 연도 | 핵심 기여 | 물리 모델 |
|------|------|------|-----------|-----------|
| **ReVeal: A Physics-Informed Neural Network for High-Fidelity Radio Environment Mapping** (Shahid et al., Iowa State) | DySPAN | 2025 | PINN으로 전파 환경 지도 구축. shadowing, interference, fading에 강건한 일반화 | 전파 감쇠 모델 |
| **ReVeal-MT: A Physics-Informed Neural Network for Multi-Transmitter Radio Environment Mapping** (Shahid et al.) | arXiv | 2025 | 다중 송신기 환경으로 확장한 PINN 기반 전파 환경 매핑 | 다중 소스 전파 모델 |

### 3.3 Physics-Informed 무선 채널 모델링

| 논문 | 학회/저널 | 연도 | 핵심 기여 | 물리 모델 |
|------|-----------|------|-----------|-----------|
| **Physics-Informed Generalizable Wireless Channel Modeling with Segmentation and Deep Learning** (Zhu, Sun, Ji) | IEEE Wireless Comm. | 2024 | 물리 정보 + 이미지 세그멘테이션으로 환경 일반화 가능한 채널 모델링 | 경로 손실 + 환경 세그멘테이션 |
| **PINNs for Electromagnetic Wave Propagation** (Bulut) | arXiv | 2025 | EM 파동 전파에 PINN 직접 적용. Maxwell 방정식 기반 |  Maxwell's Equations |
| **Physics-Informed Neural Modeling of 2D Transient Electromagnetic Fields** | Applied Sciences | 2024 | 2D 과도 전자기장의 PINN 모델링 | 2D EM 방정식 |

---

## 4. NeRF/3DGS 기반 무선 채널 모델링

> **경쟁 관계 기술** — PINN과 비교/결합 가능한 대안적 접근

| 논문 | 학회 | 연도 | 핵심 기여 | 접근 |
|------|------|------|-----------|------|
| **NeWRF: A Deep Learning Framework for Wireless Radiation Field Reconstruction and Channel Prediction** (Lu et al., UCLA) | ICML | 2024 | NeRF를 RF 도메인으로 확장. 희소 채널 측정으로 무선 방사장 복원 및 채널 예측 | NeRF |
| **GSRF: Complex-Valued 3D Gaussian Splatting for Efficient Radio-Frequency Data Synthesis** (Yang et al.) | NeurIPS | 2025 | 3DGS를 RF 도메인으로 확장. 복소수 값 가우시안으로 amplitude + phase 동시 모델링. NeRF 대비 빠른 학습/추론 | 3DGS |
| **RF-3DGS: Wireless Channel Modeling with Radio Radiance Field and 3D Gaussian Splatting** (Zhang, Sun et al.) | arXiv | 2025 | 3D Gaussian으로 무선 전파 상호작용 (반사, 회절, 산란) 모델링 | 3DGS |
| **WRF-GS: Wireless Radiation Field with 3D Gaussian Splatting** (Wen et al.) | arXiv | 2025 | 3D Gaussian primitives + 신경망으로 환경-전파 상호작용 포착, 공간 스펙트럼 합성 | 3DGS |
| **NeRF-APT: A New NeRF Framework for Wireless Channel Prediction** (Shen et al.) | arXiv | 2025 | NeRF 기반 무선 채널 예측 새 프레임워크 | NeRF |
| **Mip-NeWRF: Enhanced Wireless Radiance Field with Hybrid Encoding for Channel Prediction** (Fu et al.) | arXiv | 2025 | coarse-to-fine importance sampling + 하이브리드 인코딩으로 실내 채널 예측 정확도 향상. **Physics-informed** 접근 표방 | NeRF + Physics |

---

## 5. PINN에 활용 가능한 물리 모델

> WiFi 신호 전파를 지배하는 물리 법칙들 — PINN의 physics loss term으로 활용 가능

### 5.A 대규모 전파 모델 (Large-Scale Propagation)

#### Log-Distance Path Loss Model
$$PL(d) = PL(d_0) + 10n \log_{10}\left(\frac{d}{d_0}\right) + X_\sigma$$
- **적용**: 거리에 따른 평균 신호 감쇠
- **PINN 활용**: 모델 예측 신호 세기가 이 관계를 만족하도록 soft constraint
- **파라미터**: path loss exponent $n$ (실내: 1.6~3.3), $X_\sigma$ (shadow fading, log-normal)

#### ITU-R P.1238 Indoor Model
$$L_{total} = 20\log_{10}(f) + N\log_{10}(d) + L_f(n) - 28$$
- **적용**: ITU 표준 실내 전파 모델
- **PINN 활용**: 주파수, 거리, 층간 손실을 통합한 physics constraint
- **파라미터**: $N$ (거리 감쇠 계수), $L_f(n)$ (바닥 관통 손실), $f$ (주파수 MHz)

#### Multi-Wall (Motley-Keenan) Model
$$L = L_{FS} + \sum_{i=1}^{I} k_{w_i} L_{w_i} + \sum_{j=1}^{J} k_{f_j} L_{f_j}$$
- **적용**: 벽/바닥 투과 손실 모델링
- **PINN 활용**: 환경 구조 정보를 반영한 장애물 기반 loss
- **파라미터**: $L_{w_i}$ (벽 관통 손실), $L_{f_j}$ (바닥 관통 손실), $k$ (관통 횟수)

### 5.B 소규모 페이딩 모델 (Small-Scale Fading)

#### Saleh-Valenzuela Model (다중경로 클러스터 모델)
$$h(\tau) = \sum_{l=0}^{L} \sum_{k=0}^{K} \alpha_{kl} \delta(\tau - T_l - \tau_{kl})$$
- **적용**: 실내 다중경로 채널의 시간 도메인 임펄스 응답
- **PINN 활용**: CSI의 시간 영역 구조를 클러스터 형태로 제약
- **파라미터**: 클러스터 도착률 $\Lambda$, 경로 도착률 $\lambda$, 클러스터/경로 감쇠율

#### Rayleigh / Rician Fading
$$f(x) = \frac{x}{\sigma^2} e^{-x^2/(2\sigma^2)}  \quad \text{(Rayleigh)}$$
$$f(x) = \frac{x}{\sigma^2} e^{-\frac{x^2+s^2}{2\sigma^2}} I_0\left(\frac{xs}{\sigma^2}\right)  \quad \text{(Rician)}$$
- **적용**: CSI 진폭 분포의 통계적 모델
- **PINN 활용**: CSI amplitude 분포가 이론적 분포를 따르도록 regularization
- **파라미터**: $K$-factor (Rician), $\sigma$ (Rayleigh)

### 5.C 주파수 도메인 채널 모델 (Frequency-Domain)

#### OFDM Channel Frequency Response
$$H(f_k) = \sum_{l=0}^{L-1} h_l \cdot e^{-j2\pi f_k \tau_l}$$
- **적용**: CSI의 핵심 수학적 모델. 각 서브캐리어의 채널 응답
- **PINN 활용**: **가장 직접적인 physics constraint**. CSI가 다중경로 합으로 표현됨을 강제
- **파라미터**: $h_l$ (경로 복소 이득), $\tau_l$ (경로 지연), $L$ (경로 수)
- **핵심**: 서브캐리어 간 CSI 값은 독립이 아닌 주파수 상관 구조를 가짐

#### Delay Spread Constraint
$$\sigma_\tau = \sqrt{\frac{\sum_l P_l \tau_l^2}{\sum_l P_l} - \left(\frac{\sum_l P_l \tau_l}{\sum_l P_l}\right)^2}$$
- **적용**: 채널의 시간 분산 특성
- **PINN 활용**: 물리적으로 의미 있는 delay spread 범위를 constraint로 활용

### 5.D 공간 모델 (Spatial Models)

#### Helmholtz Equation (EM Wave Propagation)
$$\nabla^2 u + k^2 u = 0, \quad k = \frac{2\pi f}{c}$$
- **적용**: 전자기파 전파의 기본 방정식
- **PINN 활용**: PINN이 전자기장을 예측할 때 이 PDE를 만족하도록 residual loss
- **파라미터**: $k$ (파수), $f$ (주파수), $c$ (광속)

#### Fresnel Zone / Diffraction
$$r_n = \sqrt{n\lambda \frac{d_1 d_2}{d_1 + d_2}}$$
- **적용**: 회절에 의한 신호 세기 변화
- **PINN 활용**: 장애물 주변 신호 변화가 Fresnel 이론을 따르도록 constraint

#### Friis Free-Space Equation
$$P_r = P_t G_t G_r \left(\frac{\lambda}{4\pi d}\right)^2$$
- **적용**: 자유공간 전파의 기본 법칙
- **PINN 활용**: 거리에 따른 기본 신호 감쇠 관계를 baseline constraint로 사용

---

## 6. 연구 공백 및 기회 (Research Gaps)

> **핵심 발견**: PINN + WiFi CSI의 직접 결합은 아직 거의 미개척 영역

### Gap 1: PINN for CSI-based Sensing Tasks ⭐⭐⭐ (가장 큰 기회)
- **현황**: PINN은 주로 채널 추정/전파 모델링에 적용됨 (5G/mmWave 중심)
- **공백**: CSI 기반 센싱 작업 (활동인식, 제스처인식, 건강모니터링)에 PINN 적용 논문 **전무**
- **기회**: 전파 물리를 활용하여 CSI 센싱의 도메인 일반화/환경 적응 문제 해결
- **잠재적 기여**: "PINN이 환경 변화에서도 물리 법칙은 불변"이라는 점을 활용한 강건한 센싱

### Gap 2: Physics-Informed CSI Domain Generalization ⭐⭐⭐
- **현황**: CSI 센싱의 최대 난제는 환경/시간 일반화 (Wang et al. 2025 서베이에서 지적)
- **공백**: 기존 domain adaptation은 순수 데이터 기반 (adversarial, meta-learning 등)
- **기회**: 물리 법칙 (path loss, multipath 구조)을 domain-invariant prior로 활용
- **잠재적 기여**: 물리 기반 불변 특징 추출로 환경 전이 문제 근본 해결

### Gap 3: PINN for WiFi CSI Super-Resolution / Denoising ⭐⭐
- **현황**: CSI는 노이즈가 심하고 (특히 위상), 서브캐리어 해상도가 제한적
- **공백**: CSI 향상에 물리 모델을 사용한 연구 부재
- **기회**: OFDM 채널 모델 H(f)의 주파수 상관 구조를 활용한 물리 기반 CSI 향상
- **잠재적 기여**: 노이즈 제거/해상도 향상 동시에 물리적 일관성 보장

### Gap 4: Physics-Informed CSI Fingerprint Localization ⭐⭐
- **현황**: CSI 핑거프린트 측위는 순수 데이터 기반. 환경 변화 시 재수집 필요
- **공백**: 전파 모델을 핑거프린트 학습에 통합한 연구 부재
- **기회**: path loss / multipath 구조를 PINN constraint로 활용하여 핑거프린트의 물리적 일관성 보장
- **잠재적 기여**: 더 적은 수집 포인트로 높은 정확도 달성, 환경 변화 적응

### Gap 5: PINN vs NeRF/3DGS for WiFi — 비교 연구 부재 ⭐⭐
- **현황**: NeRF/3DGS 기반 무선 채널 모델링은 활발하지만, PINN과의 체계적 비교 없음
- **공백**: 동일 문제/데이터셋에서 PINN vs NeRF vs 3DGS 비교 부재
- **기회**: PDE 제약 기반 PINN vs 암시적 물리 학습 NeRF의 근본적 비교
- **잠재적 기여**: 각 접근법의 강점/약점 체계화, 하이브리드 방법 제안

### Gap 6: PINN for WiFi Imaging / Through-Wall Sensing ⭐
- **현황**: WiFi 이미징은 주로 역문제 기반 (compressed sensing, backprojection)
- **공백**: Helmholtz/Maxwell 방정식을 활용한 PINN 기반 WiFi 이미징 미개척
- **기회**: PINN이 역문제 (inverse problem) 해결에 강점이 있으므로 자연스러운 적용

---

## 7. 제안 연구 방향

> Top conference (MobiCom, NeurIPS, ICML 등) 타겟 가능한 구체적 연구 주제

### 방향 A: Physics-Informed CSI Sensing for Domain Generalization
**타겟 학회**: MobiCom / UbiComp / NeurIPS
```
핵심 아이디어:
  전파 물리 법칙 (path loss, OFDM 채널 모델)을 PINN loss로 활용하여
  새로운 환경에서도 일반화 가능한 CSI 센싱 모델 구축

기대 기여:
  1. CSI 센싱에서 최초의 physics-informed 접근법
  2. 환경 전이 문제의 근본적 해결 (물리 불변성 활용)
  3. 데이터 효율적 적응 (소수 샘플 + 물리 prior)

필요 실험:
  - 다중 환경 CSI 데이터셋 (CSI-Bench 등)
  - Cross-domain evaluation (실험실 → 복도 → 거실)
  - 기존 domain adaptation 기법 대비 비교
```

### 방향 B: PINN-Enhanced CSI Super-Resolution and Denoising
**타겟 학회**: ICML / ICLR / IEEE TWC
```
핵심 아이디어:
  OFDM 채널의 주파수 상관 구조 H(f) = Σ h_l·exp(-j2πfτ_l)를
  physics constraint로 활용하여 CSI 품질 향상

기대 기여:
  1. 물리적 일관성이 보장된 CSI enhancement
  2. 위상 노이즈 제거에 특히 효과적
  3. 서브캐리어 차원의 super-resolution

필요 실험:
  - 다양한 SNR 환경 CSI 데이터
  - 향상된 CSI를 downstream task (측위, 인식)에 적용하여 간접 평가
```

### 방향 C: Neural Operator for CSI Prediction across Environments
**타겟 학회**: NeurIPS / ICML
```
핵심 아이디어:
  DeepONet/FNO를 활용하여 환경 파라미터 → CSI 매핑을 학습하는
  physics-informed neural operator

기대 기여:
  1. 환경 변경 시 재학습 없이 CSI 예측 가능
  2. 전파 물리를 operator learning에 통합하는 새로운 프레임워크
  3. 실내 환경의 parametric PDE로서의 CSI 해석

필요 실험:
  - 다양한 실내 레이아웃의 CSI 시뮬레이션 + 실측
  - 새로운 환경으로의 zero-shot/few-shot 전이 평가
```

### 방향 D: Physics-Informed WiFi Imaging via PINN
**타겟 학회**: MobiCom / CVPR / NeurIPS
```
핵심 아이디어:
  Helmholtz 방정식을 PDE residual loss로 활용하여
  CSI 데이터로부터 실내 환경 이미징

기대 기여:
  1. 최초의 PINN 기반 WiFi 이미징
  2. NeRF/3DGS 기반 RF 모델링과의 차별화 (명시적 물리 제약)
  3. through-wall sensing 정확도 향상

필요 실험:
  - 제어된 환경의 CSI 데이터 수집
  - ground truth 환경 구조와의 이미징 정확도 비교
```

---

## 핵심 참고문헌 (Quick Reference)

### 반드시 읽어야 할 논문 (Must-Read, 우선순위순)

1. **Javid & González-Prelcic (2025)** — PINN for Wireless Channel Estimation (NeurIPS Workshop) — *PINN+무선의 가장 최신 사례*
2. **Böck et al. (2025)** — Physics-Informed Generative Modeling of Wireless Channels (ICML 2025) — *Top venue에서 인정받은 physics-informed 채널 모델링*
3. **Wang et al. (2025/2026)** — Wi-Fi Sensing Generalizability Survey (IEEE COMST) — *WiFi 센싱 일반화 문제의 포괄적 정리*
4. **Hao et al. (2024)** — PINNacle Benchmark (NeurIPS 2024) — *PINN 방법론 비교 벤치마크*
5. **Lu et al. (2024)** — NeWRF (ICML 2024) — *NeRF 기반 무선 채널 예측의 기준점*
6. **Yang et al. (2025)** — GSRF (NeurIPS 2025) — *3DGS 기반 RF 합성, PINN과 비교 가능*
7. **Shahid et al. (2025)** — ReVeal (DySPAN 2025) — *PINN for Radio Environment Mapping*
8. **Zhu et al. (2024)** — Physics-Informed Generalizable Channel Modeling (IEEE Wireless Comm.) — *세그멘테이션 + 물리 채널 모델링*

### 데이터셋 및 도구

| 이름 | 설명 | URL |
|------|------|-----|
| CSI-Bench | 대규모 실환경 다중작업 WiFi 센싱 데이터셋 | 논문에서 확인 |
| SDP8.org | WiFi 센싱 데이터셋 플랫폼 (Wang et al. 2025) | http://www.sdp8.org/ |
| csiread | Intel 5300 / Atheros / ESP32 CSI 파싱 라이브러리 | https://github.com/citysu/csiread |
| Awesome-WiFi-CSI-Sensing | WiFi CSI 센싱 논문/코드 모음 | GitHub |
| PINN_Paper_List | PINN 논문 목록 (Event-AHU) | https://github.com/Event-AHU/PINN_Paper_List |

---

> **결론**: PINN × WiFi CSI는 양쪽 분야 모두 활발하지만, **직접적 교차 연구가 거의 없는** 높은 잠재력의 연구 영역이다. 특히 CSI 센싱의 도메인 일반화 문제에 물리 정보를 활용하는 방향(Gap 1, 2)이 가장 높은 임팩트를 기대할 수 있다.
