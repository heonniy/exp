# 7/31 MoE Offloading 실험 검증 요약

## 현재 결론

현재 `7-31 exp` 결과는 전부 같은 신뢰도로 보면 안 된다.

- **유지 가능:** 데이터셋 개수, routing Expert ID trace, fixed-B50 기준 논리적 fetch/hit 계산
- **재실험 필요:** H2D 시간, prefetch overlap, throughput, Bmax, full-resident 성능
- **폐기:** 기존 `50.4% overlap` 주장과 3-tensor fetch 기반 runtime 결과

최근 수정한 **packed contiguous Expert**와 **overlap 계측 수정**이 반영된 새 commit에서 결과를 처음부터 다시 생성해야 한다.

---

## 가장 중요한 문제

### 1. 기존 결과가 자동 재사용될 수 있음

일부 sweep script는 결과 파일이 이미 존재하면 실행을 건너뛴다. 따라서 코드를 고쳐도 과거 3-tensor 결과가 그대로 남을 수 있다.

- 기존 결과를 `legacy_3tensor/`로 이동
- 새 결과는 commit SHA별 디렉터리에 저장
- 모든 JSON에 `git_sha`, 실행 명령, timestamp, trace hash 기록

### 2. 기존 H2D/prefetch 결과는 무효

기존 구현은 Expert 하나를 `gate/up/down` 3개 tensor로 나눠 복사했다. 또한 `50.4% overlap`은 단순 CUDA interval 교집합이었고, 실제 makespan 감소는 없었다.

새 실험에서는 다음을 직접 측정해야 한다.

- `compute-only`
- `copy-only`
- `sequential copy + compute`
- `double-buffer overlap`
- `useful hidden time = sequential makespan - overlap makespan`
- compute/copy slowdown ratio

Packed Expert가 실제로 **miss당 H2D copy 1회**인지 Nsight로 확인해야 한다.

### 3. Forced routing correctness 문제

현재 trace는 Expert ID만 저장하고 routing weight는 저장하지 않는다. Runtime에서 Expert ID만 강제하고 natural routing weight를 그대로 사용하면 hidden state와 logits가 틀릴 수 있다.

- trace에 `routing_expert_weights` 추가
- Expert ID와 weight를 함께 replay
- full model과 offloaded model의 logits 비교

### 4. Bmax 측정이 너무 약함

기존 Bmax는 peak KV를 할당하고 decode 1 step만 성공하면 feasible로 판단했다. 실제 256-step 중 더 큰 workspace나 다른 routing pattern에서 OOM이 날 수 있다.

- 모든 k에서 Bmax 재측정
- `Bmax-1`, `Bmax`, `Bmax+1`을 256 steps로 검증
- packed allocation으로 fragmentation이 바뀌므로 기존 `157/41`은 재사용하지 않음

---

## Cache 결과 해석 시 주의점

### Permanent-k

현재 선택 기준은 사실상 token-frequency top-k다. Batch-level fetch 절감을 직접 최적화하려면 `wave × step × layer`에서 Expert가 등장했는지를 세는 batch-step presence 기준도 비교해야 한다.

### Quota-LRU

현재 결과는 `ascending Expert ID + every-miss admission` 조건에서 발생한 scan thrashing이다. Layer-local cache 자체가 실패했다는 뜻은 아니다.

비교가 필요한 control:

- resident-hit-first ordering
- router order 보존
- miss bypass
- frequency-based admission
- 여러 request-order seed

---

## 데이터와 runtime 측정 주의점

- workload는 일반 LMSYS가 아니라 **4K input + 256 output을 만족하는 long-context stress subset**이다.
- `static_zero KV`는 메모리/성능 fixture일 뿐 correctness 검증용이 아니다.
- 실제 correctness는 `real_prefill`에서 확인해야 한다.
- Python `tolist`, `torch.where`, layer별 synchronize가 포함되어 있어 현재 runtime은 최적화된 H100 engine이 아니라 reference implementation 성능이다.
- 성능 결과는 warm-up 후 최소 5회 반복하고 median과 분산을 기록해야 한다.

---

## 재실험 순서

1. Packed Expert와 overlap 수정 코드를 GitHub에 push
2. 기존 결과를 legacy 디렉터리로 격리
3. Packed tensor/view/MLP correctness test
4. Expert ID + routing weight replay correctness test
5. H2D 및 prefetch 4-way microbenchmark
6. 모든 k의 256-step Bmax 재측정
7. Permanent/Quota policy control 실험
8. 각 k의 maximum-B와 common-B throughput 측정
9. 마지막에만 summary/progress 문서 갱신

> 현재 가장 안전한 판단은 **routing trace 기반의 논리적 cache 경향만 참고하고, 기존 runtime·Bmax·prefetch 수치는 새 packed 구현에서 전부 다시 측정하는 것**이다.
