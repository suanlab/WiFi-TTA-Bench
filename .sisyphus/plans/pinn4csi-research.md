# PINN4CSI: Physics-Informed Deep Learning for WiFi CSI Analysis

## TL;DR

> **Quick Summary**: WiFi CSI(Channel State Information) 분석에 PINN(Physics-Informed Neural Networks)을 적용하는 박사 학위 연구 프레임워크. 전파 물리를 neural network에 통합하여 환경 일반화, CSI 품질 향상, 채널 예측, WiFi 이미징을 달성한다.
>
> **Deliverables**:
> - Proof-of-concept: 물리 제약의 CSI 센싱 효과 검증 (feasibility gate)
> - Paper 1: Physics-Informed CSI Representation Learning (ICLR/ICML 타겟)
> - Paper 2: Physics-Informed Domain-Invariant WiFi Sensing (NeurIPS 타겟)
> - Paper 3: Neural Operators for Environment-Parametric CSI (ICML 타겟)
> - Paper 4: Physics-Informed WiFi Imaging via PINN (NeurIPS/CVPR 타겟)
> - Paper 5: PINN4CSI Unified Framework (IEEE TPAMI/journal 타겟)
> - 오픈소스 pinn4csi PyTorch 라이브러리
>
> **Estimated Effort**: XL (PhD thesis, 3-5 years)
> **Critical Path**: Feasibility Gate → Paper 1 → Paper 2 → Papers 3/4 (parallel) → Paper 5

---

## Context

### Original Request
Physics-Informed Neural Networks를 이용하여 WiFi 신호와 CSI 정보를 분석 및 활용하는 딥러닝 모델 연구. Top conference 수준의 논문 출판을 목표로 하는 박사 학위 연구.

### Interview Summary

**Key Decisions**:
- 학위 과정: 박사 (3-5년, 3-5편 논문)
- 연구 범위: 4개 방향 전체 종합 — 통합 PINN4CSI 프레임워크
- CSI 장비: ESP32 + 802.11ax/WiFi 6
- 관심 센싱 태스크: 전체 (측위, 활동인식, 제스처, 건강모니터링, 존재감지, 이미징)
- 첫 논문 타겟: ML 학회 (NeurIPS / ICML / ICLR)
- 전문 분야: ML/DL 중심

**Research Findings (문헌 조사)**:
- 상세 문헌 조사: `.sisyphus/drafts/literature-survey.md` (362줄, 40+편 논문 분석)
- **핵심 발견**: PINN × WiFi CSI 직접 결합 연구가 거의 전무 — 높은 novelty 기회
- 최근 관련 연구: Javid 2025 (PINN + wireless channel estimation, NeurIPS Workshop)
- 경쟁 기술: NeRF/3DGS 기반 RF 모델링 (NeWRF ICML'24, GSRF NeurIPS'25)
- WiFi 센싱 최대 난제: 환경/시간 일반화 (Wang et al. 2025 IEEE COMST)

### Metis Review

**식별된 핵심 리스크 (addressed)**:
1. **물리 제약 효과 미검증** → Feasibility Gate (6주) 도입
2. **ESP32 위상 노이즈** → amplitude-only 옵션 + 위상 보정 전략
3. **"Physics Washing" 비판** → loss term 이상의 깊은 물리 통합 필요
4. **학회 수용성** → 방법론 novelty 중심 프레이밍, backup venue 준비
5. **연구 간 과도한 결합** → 각 논문 자립 가능하게 설계

**식별된 가드레일 (incorporated)**:
- G1: Feasibility Gate 필수 (물리 제약 효과 ≥3% 검증 후 진행)
- G2: Paper 1에 센싱 태스크 1개만
- G3: 프레임워크 엔지니어링 시간 예산 20% 이하
- G4: 가장 단순한 물리 모델(path loss)부터 시작
- G5: 각 논문 backup venue 준비
- G6: 구체적 구현 2개 이상 후에만 추상화

---

## Work Objectives

### Core Objective
전파 물리 법칙을 PINN loss term 및 네트워크 아키텍처에 통합하여, WiFi CSI 기반 센싱의 환경 일반화·신호 품질·예측 정확도를 근본적으로 개선하는 통합 프레임워크를 구축하고, top conference 논문으로 검증한다.

### Concrete Deliverables
- `pinn4csi/` Python 패키지 (PyTorch 기반)
- 5편의 학술 논문 (3편 core + 2편 stretch)
- 공개 데이터셋 기반 재현 가능한 실험
- 자체 수집 CSI 데이터셋 (ESP32 + WiFi 6)

### Definition of Done
- [ ] Feasibility gate 통과: 물리 제약 모델이 baseline 대비 ≥3% 향상
- [ ] Paper 1 제출: ≥2 공개 데이터셋, ≥3 baseline 비교, ablation study
- [ ] Paper 2 제출: cross-domain 정확도 하락 <15% (기존 30-50% 대비)
- [ ] 오픈소스 코드 + 재현 스크립트 공개

### Must Have
- Feasibility gate (물리 제약 효과 검증) — 모든 개발에 선행
- 공개 데이터셋 기반 실험 (자체 데이터에 의존하지 않음)
- ≥3 baseline 비교 (per paper)
- Ablation study: 각 물리 loss component 개별 on/off
- Adaptive physics loss weighting (고정 λ 금지)
- pytest ≥80% coverage (physics/ 및 models/)
- Cross-environment evaluation (train env A → test env B)

### Must NOT Have (Guardrails)
- 프레임워크 먼저 구축 후 실험 — **반드시 proof-of-concept 먼저**
- Paper 1에 2개 이상 센싱 태스크 — **1개만**
- Paper 5(통합) 선행 개발 — **Papers 1-3 이후에만**
- 고정 λ_physics weighting — **adaptive 방식 필수**
- ESP32 데이터 수집 완료를 Paper 1 전제 조건으로 — **공개 데이터 우선**
- Complex-valued neural network 자체 구축 — **amplitude/phase 분리 사용**
- "모델이 잘 작동함" 같은 모호한 평가 — **정확한 메트릭/임계값/데이터셋 명시**
- Helmholtz 방정식으로 시작 — **log-distance path loss부터 점진적 확장**
- 여러 물리 모델 동시 구현 — **하나씩 검증 후 추가**

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: NO (greenfield — 설정 필요)
- **Automated tests**: YES (TDD)
- **Framework**: pytest + torch
- **TDD 적용**: physics 모듈은 RED-GREEN-REFACTOR 필수. 분석 솔루션 대비 검증

### QA Policy
- **Physics 함수**: 해석해(analytical solution) 대비 검증 (`atol=1e-5`)
- **모델 shape**: `model(torch.randn(B, in)).shape == (B, out)` for B ∈ {1, 16, 256}
- **Gradient flow**: `autograd.grad(output.sum(), input, create_graph=True)` → not None, no NaN
- **학습 재현성**: 동일 seed → 동일 loss curve (±1e-5)
- **Feasibility test**: 자동화된 미니 실험 (10 epochs, synthetic data)

---

## Execution Strategy

### Research Phases (PhD Timeline)

```
Phase 0 — Project Scaffold (Week 1-2):
├── T1: 프로젝트 구조 + pyproject.toml + 개발환경 [quick]
├── T2: 데이터 로더 (공개 CSI 데이터셋) [quick]
└── T3: 기본 학습 루프 + 실험 설정 (Hydra) [quick]

Phase 1 — Feasibility Gate (Week 3-8): ⚠️ GO/NO-GO DECISION
├── T4: 물리 모듈: log-distance path loss 구현 + TDD [quick]
├── T5: 기본 PINN 모델 (MLP + physics loss) [deep]
├── T6: Feasibility 실험: 1 태스크, 1 데이터셋, physics vs no-physics [deep]
└── T7: ⚡ GO/NO-GO 판정: ≥3% 향상 시 진행, 미달 시 피봇 [deep]

Phase 2 — Paper 1: Physics-Informed CSI Representation (Month 3-12):
├── T8: OFDM H(f) 물리 모듈 구현 [deep]
├── T9: Physics-informed CSI autoencoder/encoder [deep]
├── T10: 실험 확장: 2+ 데이터셋, 3+ baselines [unspecified-high]
├── T11: Ablation study + cross-environment evaluation [unspecified-high]
├── T12: 자체 CSI 데이터 수집 (ESP32 + WiFi 6) — 병렬 [unspecified-high]
└── T13: Paper 1 작성 + 제출 (ICLR/ICML, backup: AAAI/AISTATS) [writing]

Phase 3 — Paper 2: Domain-Invariant WiFi Sensing (Month 10-20):
├── T14: Physics-informed domain generalization 모듈 [deep]
├── T15: Multi-environment CSI 데이터 구축 + 실험 [unspecified-high]
├── T16: Baselines (DANN, CORAL, MAML 등) 구현/비교 [unspecified-high]
└── T17: Paper 2 작성 + 제출 (NeurIPS, backup: ICML) [writing]

Phase 4 — Papers 3 & 4 (Month 18-36, PARALLEL):
├── T18: Neural Operator (DeepONet OR FNO) + physics [deep]
├── T19: Paper 3 실험 + 작성 (ICML) [deep]
├── T20: Helmholtz 기반 WiFi Imaging PINN [ultrabrain]
├── T21: Paper 4 실험 + 작성 (NeurIPS/CVPR) [deep]
└── T22: NeRF/3DGS 비교 실험 (Paper 4 내 section) [unspecified-high]

Phase 5 — Integration & Thesis (Month 30-48):
├── T23: 프레임워크 통합 + 리팩토링 [unspecified-high]
├── T24: Paper 5 작성 (IEEE TPAMI/journal) [writing]
└── T25: 박사 학위 논문 작성 [writing]

Critical Path: T1→T4→T5→T6→T7(gate)→T8→T9→T10→T13→T14→T17
Feasibility Gate: T7 — PASS 시 Phase 2+, FAIL 시 피봇 전략 실행
```

### Contingency Plan (Negative Results)
```
IF Feasibility Gate FAILS (물리 제약 효과 < 3%):
├── Option A: 물리 모델 변경 (path loss → OFDM H(f) → Helmholtz)
├── Option B: 물리 통합 방식 변경 (loss term → architecture → data augment)
├── Option C: 태스크 변경 (activity recognition → localization → health)
└── Option D: 피봇 — "When Do Physics Priors Help in WiFi Sensing?" 메타 분석 논문

IF Paper 1 Rejected:
├── Reviewer feedback 분석 → 실험 보강 → 다음 학회 제출
├── Backup venue: AAAI, AISTATS, IEEE TWC
└── 최대 3회 submission cycle per paper
```

---

## TODOs

### Phase 0: Project Scaffold (Week 1-2)

- [x] 1. 프로젝트 구조 + 개발 환경 설정

  **What to do**:
  - `pyproject.toml` 생성: setuptools 빌드, `[dev]` extras (pytest, ruff, mypy, hydra-core, omegaconf)
  - `pinn4csi/` 패키지 디렉토리 구조 생성: `models/`, `physics/`, `data/`, `training/`, `utils/`, `configs/`
  - `tests/` 디렉토리 + `conftest.py` (deterministic seed fixture: `torch.manual_seed(42)` + `np.random.seed(42)`)
  - `ruff` 설정: rules E, F, W, I, UP, N, B, SIM. line-length 88
  - `mypy` 설정: strict mode for `pinn4csi/`
  - `pytest` 설정: markers `slow`, `gpu`. default `-m "not slow and not gpu"`
  - `.gitignore`: data/, outputs/, *.pt, *.pth, __pycache__/

  **Must NOT do**:
  - 복잡한 추상 클래스 / 제네릭 프레임워크 설계 — 아직 구현이 없으므로
  - Docker / CI 설정 — 후순위

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Phase 0 (with T2, T3)
  - **Blocks**: T4, T5 (Phase 1 전체)
  - **Blocked By**: None

  **References**:
  - `AGENTS.md` — 코드 스타일, 디렉토리 구조, 도구 설정 전체 참고 (이 파일이 유일한 코드 규약 원천)

  **Acceptance Criteria**:
  - [ ] `pip install -e ".[dev]"` 성공
  - [ ] `ruff check .` → 0 errors
  - [ ] `mypy pinn4csi/` → 0 errors
  - [ ] `pytest` → collected 0 items (테스트 없지만 실행은 성공)

  **QA Scenarios**:
  ```
  Scenario: 개발 환경 설치 검증
    Tool: Bash
    Steps:
      1. pip install -e ".[dev]" 실행
      2. python -c "import pinn4csi" 실행
      3. ruff check . 실행
      4. mypy pinn4csi/ 실행
      5. pytest --co 실행 (collect only)
    Expected Result: 모든 명령 exit code 0
    Evidence: .sisyphus/evidence/task-1-dev-env.txt
  ```

  **Commit**: YES
  - Message: `chore(project): scaffold PINN4CSI project structure`
  - Files: pyproject.toml, pinn4csi/**/__init__.py, tests/conftest.py, .gitignore

- [ ] 2. CSI 데이터 로더 구현 (공개 데이터셋)

  **What to do**:
  - `pinn4csi/data/csi_dataset.py`: PyTorch Dataset 클래스 for CSI data
    - Shape convention: `(num_packets, num_subcarriers, num_antennas)` — complex or (amplitude, phase) pair
    - amplitude/phase 분리 옵션 (complex → 2-channel real)
  - 최소 1개 공개 CSI 데이터셋 로더 구현 (후보: SignFi, UT-HAR, Widar3.0)
  - `pinn4csi/data/transforms.py`: CSI 전처리 (amplitude 추출, phase 정규화, subcarrier 선택)
  - Train/val/test split 유틸리티 + cross-environment split 유틸리티

  **Must NOT do**:
  - ESP32/WiFi 6 자체 수집 데이터 포맷 — 아직 수집 전
  - 복잡한 augmentation — 후순위

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Phase 0 (with T1, T3)
  - **Blocks**: T6, T10
  - **Blocked By**: None (T1과 병렬 가능하지만, import 구조 필요 시 T1 이후)

  **References**:
  - `AGENTS.md:CSI Data Conventions` — shape 규약, amplitude/phase, csiread
  - SignFi dataset: https://github.com/yongsen/SignFi (gesture recognition CSI)
  - UT-HAR dataset: activity recognition CSI (6 activities, 3 environments)
  - `.sisyphus/drafts/literature-survey.md:데이터셋 및 도구` — 공개 데이터셋 목록

  **Acceptance Criteria**:
  - [ ] `from pinn4csi.data import CSIDataset` 성공
  - [ ] Dataset.__getitem__ 반환 shape: `(num_subcarriers, features)` — features = 2 (amp+phase) or num_antennas*2
  - [ ] DataLoader 1 epoch 순회 성공 (no crash)
  - [ ] Cross-environment split: 환경 A train, 환경 B test 구성 가능

  **QA Scenarios**:
  ```
  Scenario: 데이터셋 로딩 + shape 검증
    Tool: Bash (python script)
    Steps:
      1. python -c "from pinn4csi.data import CSIDataset; ds = CSIDataset('path'); print(ds[0][0].shape, ds[0][1])"
      2. Assert output shape matches (num_subcarriers, features)
      3. Assert labels are integer class indices
    Expected Result: Shape printed without error, labels are valid ints
    Evidence: .sisyphus/evidence/task-2-dataset-shape.txt

  Scenario: Cross-environment split
    Tool: Bash (python script)
    Steps:
      1. 환경 A 데이터로 train split, 환경 B로 test split 생성
      2. Assert train과 test에 환경 겹침 없음
    Expected Result: 0 overlap between environments
    Evidence: .sisyphus/evidence/task-2-cross-env-split.txt
  ```

  **Commit**: YES
  - Message: `feat(data): add CSI dataset loader with cross-environment splits`
  - Files: pinn4csi/data/csi_dataset.py, pinn4csi/data/transforms.py, tests/test_data.py

- [ ] 3. 기본 학습 루프 + Hydra 실험 설정

  **What to do**:
  - `pinn4csi/training/trainer.py`: 기본 학습/평가 루프
    - 모델 학습, loss 로깅, checkpoint 저장
    - device 자동 선택 (cuda/cpu)
    - 개별 loss component 로깅 (loss_data, loss_physics, loss_total)
  - `pinn4csi/configs/`: Hydra YAML 설정
    - `config.yaml`: default config
    - `model/pinn.yaml`: PINN 모델 설정
    - `data/default.yaml`: 데이터 설정
  - `scripts/train.py`: Hydra entry point
  - `pinn4csi/utils/device.py`: device 관리 유틸리티
  - `pinn4csi/utils/metrics.py`: 평가 메트릭 (accuracy, NMSE, F1)

  **Must NOT do**:
  - W&B / TensorBoard 통합 — 아직 불필요
  - 분산 학습 — 후순위
  - 복잡한 스케줄러 — Adam + StepLR로 시작

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Phase 0 (with T1, T2)
  - **Blocks**: T5, T6
  - **Blocked By**: T1 (패키지 구조 필요)

  **References**:
  - `AGENTS.md:Device Management` — device 관리 패턴
  - `AGENTS.md:PINN-Specific Conventions:Loss Structure` — loss component 분리 패턴

  **Acceptance Criteria**:
  - [ ] `python scripts/train.py` → 1 epoch 완료 (synthetic data)
  - [ ] `python scripts/train.py model=pinn data=default` → Hydra override 동작
  - [ ] Checkpoint 저장/로드 성공
  - [ ] 개별 loss components (data, physics) 별도 로깅 확인

  **QA Scenarios**:
  ```
  Scenario: 학습 루프 smoke test
    Tool: Bash
    Steps:
      1. python scripts/train.py trainer.max_epochs=2 data.batch_size=16
      2. ls outputs/ 로 체크포인트 파일 확인
      3. 로그에서 loss_data, loss_physics 두 값 모두 출력 확인
    Expected Result: 2 epochs 완료, checkpoint 존재, loss components 로깅
    Evidence: .sisyphus/evidence/task-3-training-loop.txt
  ```

  **Commit**: YES
  - Message: `feat(training): add base training loop with Hydra configuration`
  - Files: pinn4csi/training/trainer.py, pinn4csi/configs/*.yaml, scripts/train.py

### Phase 1: Feasibility Gate (Week 3-8) ⚠️ GO/NO-GO

- [ ] 4. 물리 모듈: Log-Distance Path Loss (TDD)

  **What to do**:
  - **RED**: `tests/test_physics.py::test_path_loss_analytical` — 알려진 거리/주파수에서 path loss 값이 Friis 공식과 일치하는지 검증 (atol=1e-5)
  - **GREEN**: `pinn4csi/physics/path_loss.py` — `compute_path_loss(distance, frequency, n=2.0) -> Tensor`
    - 수식: `PL(d) = PL(d0) + 10n·log10(d/d0)`
    - d0=1m reference distance
    - gradient-compatible: `requires_grad=True` 입력에서 backward 가능
  - **REFACTOR**: docstring, type annotation, edge case (d=0 guard)
  - **RED**: `test_path_loss_gradient_flow` — autograd.grad 통과 검증
  - **GREEN**: gradient flow 보장

  **Must NOT do**:
  - Multi-wall model, ITU model 등 복잡한 모델 — path loss만 먼저
  - OFDM H(f) 모델 — Phase 2에서

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (Phase 1 첫 번째)
  - **Blocks**: T5, T6
  - **Blocked By**: T1

  **References**:
  - `AGENTS.md:PINN-Specific Conventions:Automatic Differentiation` — `create_graph=True` 패턴
  - `.sisyphus/drafts/literature-survey.md:5.A Log-Distance Path Loss Model` — 수학적 정의
  - Friis equation: `P_r = P_t·G_t·G_r·(λ/4πd)²`

  **Acceptance Criteria**:
  - [ ] `pytest tests/test_physics.py::test_path_loss_analytical -v` → PASS
  - [ ] `pytest tests/test_physics.py::test_path_loss_gradient_flow -v` → PASS
  - [ ] path_loss(d=10m, f=2.4GHz, n=2.0) 결과가 Friis와 일치 (atol=1e-5)
  - [ ] `mypy pinn4csi/physics/path_loss.py` → 0 errors

  **QA Scenarios**:
  ```
  Scenario: Path loss 해석해 검증
    Tool: Bash (pytest)
    Steps:
      1. pytest tests/test_physics.py -v -k "path_loss" 실행
      2. 테스트가 d=1, 10, 100m에서 해석해와 비교
    Expected Result: All pass, atol=1e-5
    Evidence: .sisyphus/evidence/task-4-path-loss-test.txt

  Scenario: Gradient flow 검증
    Tool: Bash (python script)
    Steps:
      1. x = torch.tensor([10.0], requires_grad=True)
      2. y = path_loss(x, f=2.4e9)
      3. grad = torch.autograd.grad(y, x, create_graph=True)
      4. Assert grad is not None and not NaN
    Expected Result: Valid gradient tensor
    Evidence: .sisyphus/evidence/task-4-gradient-flow.txt
  ```

  **Commit**: YES (2 commits: test first, then implementation)
  - Message 1: `test(physics): add path loss analytical validation`
  - Message 2: `feat(physics): implement log-distance path loss with autograd support`

- [ ] 5. 기본 PINN 모델 아키텍처

  **What to do**:
  - `pinn4csi/models/pinn.py`: PINN base class
    - `__init__(in_features, out_features, hidden_dim=64, num_layers=4)`
    - `forward(x) -> Tensor` — data prediction only
    - `nn.ModuleList` 기반 MLP, 활성화 함수 선택 가능 (Tanh, ReLU, GELU)
  - `pinn4csi/training/pinn_trainer.py`: PINN-specific trainer
    - `loss_data`: MSE/CE for task
    - `loss_physics`: path_loss residual (from T4)
    - `loss_total = loss_data + λ·loss_physics`
    - **Adaptive λ**: gradient normalization 방식 — `λ = ||∇loss_data|| / ||∇loss_physics||`
  - Tests: shape test, forward/backward test, adaptive λ test

  **Must NOT do**:
  - Physics-informed architecture (attention, Fourier features) — 아직 기본 MLP만
  - 여러 물리 loss 결합 — path loss 1개만

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (T4 이후)
  - **Blocks**: T6
  - **Blocked By**: T1, T4

  **References**:
  - `AGENTS.md:PINN-Specific Conventions:Model Architecture` — PINN 클래스 패턴
  - `AGENTS.md:PINN-Specific Conventions:Loss Structure` — loss 분리 패턴
  - PINNacle (NeurIPS 2024) — PINN 아키텍처 비교 결과 참고
  - Gradient normalization: Wang et al. "Understanding and Mitigating Gradient Flow Pathologies in PINNs" (2021)

  **Acceptance Criteria**:
  - [ ] `model(torch.randn(B, in_features)).shape == (B, out_features)` for B ∈ {1, 16, 256}
  - [ ] Backward pass produces valid gradients (no NaN/Inf)
  - [ ] Adaptive λ changes across epochs (not fixed)
  - [ ] `pytest tests/test_models.py -v` → ALL PASS

  **QA Scenarios**:
  ```
  Scenario: PINN model shape + gradient test
    Tool: Bash (pytest)
    Steps:
      1. pytest tests/test_models.py -v
      2. Assert shape tests pass for batch sizes 1, 16, 256
      3. Assert gradient flow test passes
    Expected Result: All tests pass
    Evidence: .sisyphus/evidence/task-5-pinn-model.txt

  Scenario: Adaptive lambda verification
    Tool: Bash (python script)
    Steps:
      1. 2 epoch 학습 실행 (synthetic data)
      2. epoch 1 vs epoch 2의 lambda_physics 값 비교
      3. Assert lambda 값이 변화함 (not constant)
    Expected Result: lambda changes between epochs
    Evidence: .sisyphus/evidence/task-5-adaptive-lambda.txt
  ```

  **Commit**: YES
  - Message: `feat(models): implement PINN base architecture with adaptive physics weighting`

- [ ] 6. ⚡ Feasibility 실험 실행

  **What to do**:
  - `scripts/feasibility.py`: 자동화된 feasibility 실험 스크립트
  - 실험 설계:
    - Task: 1개 센싱 태스크 (실내 측위 또는 활동인식) — 공개 데이터셋 사용
    - Model A: MLP baseline (physics loss 없음)
    - Model B: PINN (MLP + path loss physics constraint)
    - 동일 아키텍처, 동일 하이퍼파라미터, 차이는 physics loss만
    - λ_physics sweep: {0.001, 0.01, 0.1, 1.0, 10.0}
    - 3 random seeds per configuration
    - 메트릭: accuracy (분류) 또는 mean distance error (측위)
  - 결과 자동 집계 + 보고서 생성
  - `tests/test_feasibility.py`: 미니 실험 (10 epochs, synthetic data) — CI에서 실행 가능

  **Must NOT do**:
  - 복잡한 아키텍처 실험 — MLP만
  - 여러 데이터셋 — 1개만
  - 논문 수준 실험 — 빠른 검증 목적

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (T5 이후)
  - **Blocks**: T7 (GO/NO-GO)
  - **Blocked By**: T2, T3, T5

  **References**:
  - 공개 데이터셋: SignFi (gesture), UT-HAR (activity), 또는 UJIndoorLoc (localization)

  **Acceptance Criteria**:
  - [ ] 실험 완료: 5 λ값 × 3 seeds = 15 runs
  - [ ] 결과 테이블 생성: λ별 평균 성능 ± std
  - [ ] Best PINN vs Baseline 비교 수치 산출
  - [ ] `pytest tests/test_feasibility.py -v` → PASS (synthetic mini-experiment)

  **QA Scenarios**:
  ```
  Scenario: Feasibility 실험 전체 실행
    Tool: Bash
    Steps:
      1. python scripts/feasibility.py --seeds 3 --lambdas 0.001,0.01,0.1,1.0,10.0
      2. 결과 CSV 파일 확인
      3. PINN best 성능 vs baseline 성능 비교
    Expected Result: 실험 완료, 결과 파일 생성
    Evidence: .sisyphus/evidence/task-6-feasibility-results.csv

  Scenario: Mini feasibility test (CI)
    Tool: Bash (pytest)
    Steps:
      1. pytest tests/test_feasibility.py -v
      2. 10 epoch synthetic 실험으로 physics loss가 gradient에 기여하는지 확인
    Expected Result: PASS — physics loss가 total loss에 영향
    Evidence: .sisyphus/evidence/task-6-mini-feasibility.txt
  ```

  **Commit**: YES
  - Message: `feat(experiments): add feasibility experiment with lambda sweep`

- [ ] 7. ⚡ GO/NO-GO 판정

  **What to do**:
  - T6 결과 분석:
    - **GO 조건**: Best PINN 성능이 Baseline 대비 ≥3% accuracy 향상 또는 ≥1dB NMSE 개선
    - **NO-GO 시 피봇 전략 실행** (위 Contingency Plan 참조)
  - GO 시: Phase 2로 진행, 결과를 Paper 1의 preliminary result로 활용
  - NO-GO 시:
    - Option A 시도: OFDM H(f) 모델로 교체하여 재실험
    - Option A 실패 시: Option B (아키텍처 변경) 또는 Option D (메타 분석 논문)
  - 판정 결과 문서화

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Blocks**: Phase 2 전체 (T8-T13)
  - **Blocked By**: T6

  **Acceptance Criteria**:
  - [ ] GO/NO-GO 판정 문서 작성: 결과 수치, 판정 근거, 다음 단계
  - [ ] GO 시: Paper 1 detailed experiment plan 작성
  - [ ] NO-GO 시: 피봇 전략 선택 + 재실험 계획

  **Commit**: YES
  - Message: `docs(results): feasibility gate decision — [GO/NO-GO]`

### Phase 2: Paper 1 — Physics-Informed CSI Representation (Month 3-12)

- [ ] 8. OFDM H(f) 물리 모듈 구현 (TDD)

  **What to do**:
  - **RED**: `tests/test_physics.py::test_ofdm_channel_model` — 알려진 multipath 파라미터로 H(f) 계산, 해석해와 비교
  - **GREEN**: `pinn4csi/physics/ofdm_channel.py`:
    - `ofdm_channel_response(h_l, tau_l, f_k) -> Tensor` — H(f_k) = Σ h_l·exp(-j2πf_k·τ_l)
    - `ofdm_residual(csi_pred, coords) -> Tensor` — CSI가 OFDM 모델을 따르는 정도
    - `subcarrier_correlation_loss(csi) -> Tensor` — 서브캐리어 간 주파수 상관 구조 제약
  - **REFACTOR**: amplitude/phase 분리 인터페이스
  - Gradient flow 검증: `create_graph=True`로 2차 미분 가능

  **Must NOT do**:
  - Saleh-Valenzuela, Helmholtz — 아직 OFDM만
  - Complex-valued network — amplitude+phase 2채널로 처리

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (Phase 2 첫 번째)
  - **Blocks**: T9
  - **Blocked By**: T7 (GO 판정 필요)

  **References**:
  - `.sisyphus/drafts/literature-survey.md:5.C OFDM Channel Frequency Response` — H(f) 수식
  - Javid 2025 — RSS prior 통합 방식 참고 (cross-attention)
  - `AGENTS.md:PINN-Specific Conventions:Automatic Differentiation`

  **Acceptance Criteria**:
  - [ ] `pytest tests/test_physics.py -k "ofdm" -v` → ALL PASS
  - [ ] 알려진 2-path 채널에서 H(f) 값이 해석해와 일치 (atol=1e-4, complex)
  - [ ] Gradient flow: autograd.grad 통과, no NaN
  - [ ] `mypy pinn4csi/physics/ofdm_channel.py` → 0 errors

  **Commit**: YES (test first, then implementation)
  - Message 1: `test(physics): add OFDM channel model analytical tests`
  - Message 2: `feat(physics): implement OFDM H(f) channel model with autograd`

- [ ] 9. Physics-Informed CSI Encoder/Autoencoder

  **What to do**:
  - `pinn4csi/models/csi_pinn.py`: CSI-specific PINN
    - Encoder: CSI → latent representation (물리적으로 제약된)
    - Decoder: latent → reconstructed CSI (또는 task prediction)
    - Physics constraint in latent space: 표현이 물리 법칙을 만족하도록
  - Loss 구조:
    - `loss_reconstruction`: CSI 재구성 오류
    - `loss_task`: downstream task loss (classification/regression)
    - `loss_ofdm`: OFDM H(f) 모델 정합성
    - `loss_path`: path loss 정합성
    - Adaptive weighting for all components
  - Fourier feature embedding 옵션 (spectral bias 완화)

  **Must NOT do**:
  - Transformer 아키텍처 — MLP/CNN 기반으로 시작
  - 여러 센싱 태스크 동시 학습 — 1개 태스크에 집중

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Blocks**: T10, T11
  - **Blocked By**: T8

  **References**:
  - CoPINN (ICML 2025) — cognitive physics integration 패턴
  - PINNs with Fourier Features (NeurIPS 2025 Workshop) — spectral bias 완화
  - `AGENTS.md:PINN-Specific Conventions` — 전체 PINN 패턴

  **Acceptance Criteria**:
  - [ ] Model forward/backward shape tests pass
  - [ ] 4개 loss component 각각 개별 on/off 가능 (ablation-ready)
  - [ ] Fourier feature embedding on/off 토글 가능
  - [ ] Training 1 epoch 완료 (public dataset)

  **Commit**: YES
  - Message: `feat(models): implement physics-informed CSI encoder with multi-component loss`

- [ ] 10. Paper 1 실험: 다중 데이터셋 + Baselines

  **What to do**:
  - **데이터셋 확장**: ≥2 공개 데이터셋 (예: SignFi + UT-HAR, 또는 Widar3.0 + UT-HAR)
  - **Baselines 구현/통합** (≥3):
    1. Vanilla MLP/CNN (no physics)
    2. Standard autoencoder (no physics)
    3. Domain-specific method (예: AirFi, EI, DGSense 중 1개)
  - **실험 매트릭스**: {모델} × {데이터셋} × {환경 설정} × {3 seeds}
  - Cross-environment evaluation: train env A → test env B
  - `scripts/run_paper1_experiments.py`: 전체 실험 자동화
  - 결과 테이블 + 시각화 자동 생성

  **Must NOT do**:
  - 자체 수집 데이터에만 의존 — 공개 데이터 필수
  - cherry-pick된 결과 — 전체 결과 + mean±std 보고

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (T11, T12와 부분 병렬)
  - **Blocks**: T13
  - **Blocked By**: T9

  **Acceptance Criteria**:
  - [ ] ≥2 데이터셋 × ≥3 baselines × ≥2 환경 설정 × 3 seeds 결과 완료
  - [ ] PINN이 ≥1 설정에서 baseline 대비 통계적으로 유의미한 향상 (p<0.05)
  - [ ] 결과 테이블 LaTeX 형식 자동 생성

  **Commit**: YES
  - Message: `feat(experiments): add Paper 1 full experiment suite`

- [ ] 11. Ablation Study + Cross-Environment Evaluation

  **What to do**:
  - **Ablation study**: 각 물리 loss component 개별 제거
    - Full model vs -path_loss vs -ofdm_loss vs -both (=baseline)
    - Fourier features on/off
    - Adaptive λ vs fixed λ
  - **Cross-environment analysis**:
    - In-domain accuracy vs cross-domain accuracy 비교
    - Physics-constrained model의 cross-domain 성능 하락률 분석
    - t-SNE/UMAP으로 latent space 시각화 (물리 제약 유무 비교)
  - **Hyperparameter sensitivity**: λ range, hidden dim, num layers

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (T10과 부분 병렬)
  - **Blocks**: T13
  - **Blocked By**: T9

  **Acceptance Criteria**:
  - [ ] Ablation 테이블: 각 component 제거 시 성능 변화 수치
  - [ ] Cross-domain 성능 하락률: PINN < Baseline
  - [ ] Latent space 시각화 그림 생성

  **Commit**: YES
  - Message: `feat(experiments): add ablation study and cross-environment analysis`

- [ ] 12. 자체 CSI 데이터 수집 (ESP32 + WiFi 6) — 병렬 작업

  **What to do**:
  - ESP32 CSI 수집 환경 구축 (firmware, 수집 스크립트)
  - WiFi 6 CSI 수집 환경 구축
  - 최소 3개 환경에서 데이터 수집 (연구실, 복도, 사무실/강의실)
  - 수집 프로토콜 문서화
  - `pinn4csi/data/esp32_loader.py`: ESP32 CSI 로더
  - `pinn4csi/data/wifi6_loader.py`: WiFi 6 CSI 로더

  **Must NOT do**:
  - Paper 1 실험을 자체 데이터에 의존 — 공개 데이터 우선, 자체 데이터는 보너스
  - 데이터 수집 완료를 Paper 1 제출의 전제조건으로 삼기

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (T10, T11과 완전 병렬)
  - **Blocks**: T15 (Paper 2의 multi-environment 데이터)
  - **Blocked By**: T1

  **Acceptance Criteria**:
  - [ ] ≥3 환경에서 CSI 수집 완료
  - [ ] 수집 프로토콜 문서 작성
  - [ ] DataLoader로 로딩 + shape 검증 통과

  **Commit**: YES
  - Message: `feat(data): add ESP32 and WiFi 6 CSI data loaders`

- [ ] 13. Paper 1 작성 + 제출

  **What to do**:
  - 논문 구조:
    - Title: "Physics-Informed Representation Learning for WiFi Channel State Information"
    - Abstract → Introduction → Related Work → Method → Experiments → Conclusion
    - 핵심 기여: (1) CSI에 최초의 physics-informed 접근, (2) OFDM 물리 기반 표현 학습, (3) 환경 전이 효과
  - 타겟 학회: ICLR 또는 ICML (deadline 기준 선택)
  - Backup: AAAI, AISTATS, IEEE TWC
  - LaTeX 작성 + 실험 결과 통합
  - 내부 리뷰 → 수정 → 제출

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Blocks**: T14 (Paper 2 시작)
  - **Blocked By**: T10, T11

  **Acceptance Criteria**:
  - [ ] 논문 초고 완성 (8-10 pages)
  - [ ] 모든 실험 결과 테이블/그림 포함
  - [ ] 학회 formatting 준수
  - [ ] 제출 완료

  **Commit**: NO (논문은 별도 repo)

### Phase 3: Paper 2 — Domain-Invariant WiFi Sensing (Month 10-20)

- [ ] 14. Physics-Informed Domain Generalization 모듈

  **What to do**:
  - `pinn4csi/models/domain_invariant.py`:
    - Physics-informed feature extractor (Paper 1의 encoder 확장)
    - Domain-invariant loss: 물리 법칙 기반 불변 특징 학습
    - "물리적으로 일관된 표현은 환경 불변" 가설 구현
  - 전략: Paper 1의 물리 기반 표현이 환경 변화에 얼마나 강건한지를 핵심 기여로
  - Physics-guided data augmentation: 물리 모델로 가상 환경 CSI 생성
  - Multi-environment training with physics regularization

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Blocked By**: T7 (GO), T9 (Paper 1 모델), T13 (Paper 1 제출)

  **Acceptance Criteria**:
  - [ ] Cross-domain 성능 하락 <15% (기존 30-50% 대비 대폭 개선)
  - [ ] Physics-augmented data로 학습 시 추가 향상

- [ ] 15. Multi-Environment 실험 + Paper 2 데이터

  **What to do**:
  - 자체 수집 데이터 (T12) + 공개 데이터 결합
  - ≥3 환경 × ≥2 센싱 태스크 실험 매트릭스
  - Leave-one-environment-out evaluation
  - 기존 domain adaptation/generalization 기법과 비교

  **Blocked By**: T12, T14

- [ ] 16. Domain Adaptation Baselines 비교

  **What to do**:
  - Baselines (≥4):
    1. DANN (Domain-Adversarial Neural Network)
    2. CORAL (Correlation Alignment)
    3. MAML (Model-Agnostic Meta-Learning)
    4. WiFi-specific: EI, WiTrans, 또는 DGSense
  - 공정한 비교: 동일 backbone, 동일 데이터, 동일 evaluation protocol

  **Blocked By**: T14

- [ ] 17. Paper 2 작성 + 제출

  **What to do**:
  - Title: "Physics Laws Don't Change: Domain-Invariant WiFi Sensing via Physics-Informed Learning"
  - 타겟: NeurIPS, Backup: ICML, AAAI
  - 핵심 기여: (1) physics prior → domain invariance, (2) cross-environment 일반화 SOTA, (3) physics-augmented training

  **Blocked By**: T15, T16

### Phase 4: Papers 3 & 4 (Month 18-36, Parallel)

- [ ] 18. Neural Operator + Physics Constraints (Paper 3)

  **What to do**:
  - ONE operator 선택: DeepONet OR FNO (둘 다 아님)
  - 환경 파라미터 → CSI 매핑 학습
  - Physics loss integration into operator training
  - Zero-shot transfer to new environment evaluation

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Blocked By**: T13

- [ ] 19. Paper 3 실험 + 작성

  **What to do**:
  - 타겟: ICML, Backup: NeurIPS, AISTATS
  - 핵심 기여: (1) 무선 채널의 parametric PDE 해석, (2) physics-informed operator learning for CSI

  **Blocked By**: T18

- [ ] 20. Helmholtz 기반 WiFi Imaging PINN (Paper 4)

  **What to do**:
  - `pinn4csi/physics/helmholtz.py`: Helmholtz equation PDE residual
    - ∇²u + k²u = 0 (k = 2πf/c)
  - `pinn4csi/models/wifi_imager.py`: PINN for inverse problem
    - CSI 측정 → 환경 구조 (permittivity map) 복원
  - NeRF/3DGS 기반 방법과 비교 (NeWRF, GSRF 등)

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain`
  - **Skills**: []

  **Blocked By**: T13

- [ ] 21. Paper 4 실험 + 작성

  **What to do**:
  - 타겟: NeurIPS 또는 CVPR, Backup: ECCV, IEEE TWC
  - 핵심 기여: (1) 최초 PINN 기반 WiFi 이미징, (2) NeRF/3DGS 대비 명시적 물리 제약의 장점

  **Blocked By**: T20

- [ ] 22. NeRF/3DGS 비교 실험

  **What to do**:
  - Paper 4 내 section으로 포함 (별도 논문 아님)
  - NeWRF (ICML'24) 또는 GSRF (NeurIPS'25) 재현 → 동일 데이터에서 PINN과 비교
  - 비교 축: 정확도, 학습 속도, 데이터 효율성, 물리적 해석 가능성

  **Blocked By**: T20

### Phase 5: Integration & Thesis (Month 30-48)

- [ ] 23. 프레임워크 통합 + 리팩토링

  **What to do**:
  - Papers 1-4에서 반복되는 패턴 추출 → 공통 인터페이스
  - `pinn4csi` 라이브러리 정리: 문서화, API 안정화
  - 재현 가능한 실험 스크립트 전체 정리

  **Blocked By**: T19, T21

  **Must NOT do**:
  - 새로운 기능 추가 — 기존 것만 통합
  - 과도한 추상화 — Papers에서 쓴 패턴 그대로 정리

- [ ] 24. Paper 5: Unified Framework

  **What to do**:
  - 타겟: IEEE TPAMI 또는 Nature Communications (journal)
  - 핵심 기여: PINN4CSI 통합 프레임워크, 전체 실험 종합, design guideline

  **Blocked By**: T23

- [ ] 25. 박사 학위 논문 작성

  **What to do**:
  - Papers 1-5를 중심으로 학위 논문 구성
  - Chapter 구조: Introduction → Background → Paper 1 → Paper 2 → Paper 3 → Paper 4 → Framework → Conclusion

  **Blocked By**: T24

---

## Final Verification Wave

- [ ] F1. **연구 계획 준수 감사** — `oracle`
  전체 계획 대비 구현 상태 검증. Must Have/Must NOT Have 체크.

- [ ] F2. **코드 품질 리뷰** — `unspecified-high`
  ruff check + mypy + pytest 통과. AI slop 패턴 검사.

- [ ] F3. **실험 재현성 QA** — `unspecified-high`
  seed 고정 후 동일 결과 재현 검증. 전체 실험 파이프라인 실행.

- [ ] F4. **Scope Fidelity Check** — `deep`
  각 task의 spec vs actual diff 1:1 비교. scope creep 탐지.

---

## Commit Strategy

Phase 0:
- `chore(project): scaffold project structure with pyproject.toml` — pyproject.toml, pinn4csi/__init__.py
- `feat(data): add CSI dataset loaders` — pinn4csi/data/
- `feat(training): add base training loop with Hydra config` — pinn4csi/training/, configs/

Phase 1:
- `test(physics): add path loss analytical validation` — tests/test_physics.py
- `feat(physics): implement log-distance path loss` — pinn4csi/physics/path_loss.py
- `test(models): add PINN base model shape tests` — tests/test_models.py
- `feat(models): implement PINN base architecture` — pinn4csi/models/pinn.py
- `feat(training): add feasibility experiment script` — scripts/feasibility.py
- `docs(results): feasibility gate results` — results/feasibility/

Phase 2+: 각 논문별 feature branch → 완료 시 merge

---

## Success Criteria

### Feasibility Gate (Week 6-8)
```python
# Automated feasibility test
assert physics_model_accuracy - baseline_accuracy >= 0.03  # ≥3% improvement
# OR
assert baseline_nmse - physics_model_nmse >= 1.0  # ≥1dB NMSE improvement
```

### Paper 1 Readiness
```
- [ ] ≥2 public datasets with reproducible results
- [ ] ≥3 baseline comparisons (specific models + implementations)
- [ ] Ablation: each physics component on/off
- [ ] Cross-environment evaluation (train A → test B)
- [ ] Statistical significance: 3+ seeds, mean ± std
- [ ] ruff check + mypy + pytest pass
```

### Overall PhD Completion
```
- [ ] ≥3 papers published/accepted at top venues
- [ ] Open-source pinn4csi library on GitHub
- [ ] Reproducible experiments with public datasets
- [ ] PhD dissertation combining all contributions
```
