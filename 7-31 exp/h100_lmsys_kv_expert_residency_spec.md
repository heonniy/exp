# H100 기반 KV–Expert Residency Trade-off 실험 명세서

## 0. 문서 목적

본 문서는 **Qwen3-30B-A3B MoE inference offloading** 환경에서, 고정된 H100 HBM을 다음 두 용도 사이에 어떻게 배분할지 실험하기 위한 코딩 명세서다.

1. GPU-resident KV cache를 늘려 더 큰 decode batch를 처리한다.
2. Layer별 Expert residency를 늘려 반복적인 Expert H2D refetch를 줄인다.

핵심 비교는 다음과 같다.

> 동일한 HBM에서, 모델 전체에 2개의 transient Expert slot만 두고 KV batch를 최대화하는 것이 유리한가?  
> 아니면 일부 KV capacity와 batch를 포기하더라도, 각 layer에 소수의 Expert를 유지하여 cyclic refetch를 줄이는 것이 유리한가?

이 명세서의 핵심 실험에서는 **Global LRU Expert cache를 사용하지 않는다.**

---

# 1. 핵심 연구 질문

## RQ1. KV extreme vs Expert extreme

- **KV extreme**: persistent Expert cache 없이 transient slot 2개만 사용하고, 나머지 HBM을 KV에 할당한다.
- **Expert extreme**: layer당 128개 Expert를 모두 GPU에 상주시켜 decode 중 Expert refetch를 제거하고, 남은 HBM만 KV에 할당한다.
- 두 극단 사이에서 layer당 resident Expert 수 `k`를 변화시켜 throughput curve를 측정한다.

## RQ2. 같은 Expert 용량의 구성 방식

같은 `k`와 같은 HBM을 사용할 때 다음 두 정책을 비교한다.

- **Permanent-k**: calibration workload에서 선정한 layer별 top-k Expert를 영구 고정한다.
- **Quota-LRU-k**: 각 layer에 k개의 local LRU slot을 보장하되 Expert identity는 runtime에 적응한다.

## RQ3. Expert refetch 감소가 실제 throughput으로 이어지는가

다음 현상을 구분한다.

- Expert H2D traffic은 감소했지만 decode가 compute-bound라 throughput은 거의 변하지 않는 경우
- Expert H2D traffic 감소가 exposed stall 감소로 이어져 throughput이 개선되는 경우
- Expert residency 때문에 feasible KV batch가 줄어 전체 throughput이 악화되는 경우

---

# 2. 용어 정리

## 2.1 On-demand fetch

Router가 현재 layer에서 필요한 Expert를 확인한 뒤, Expert가 GPU에 없을 때 CPU에서 GPU로 가져오는 방식이다.

엄격한 `no-prefetch on-demand`에서는 다음 순서가 반복된다.

```text
Expert miss 확인
→ H2D fetch 완료 대기
→ Expert GEMM 실행
→ 다음 Expert miss 확인
```

이 경우 H2D와 GEMM이 거의 겹치지 않는다.

## 2.2 Prefetch

Expert가 실제로 계산에 사용되기 전에 H2D를 미리 시작하는 것이다.

본 실험의 기본 prefetch는 **current-layer one-ahead prefetch**다.

```text
Slot A의 Expert i 계산
동시에 Slot B로 다음 miss Expert i+1 H2D
→ 둘 다 완료되면 A/B 역할 교체
```

중요한 점:

- 다음 layer를 예측하는 prefetch가 아니다.
- 현재 layer의 Router 결과로 이미 필요한 Expert 목록을 알고 난 뒤 수행한다.
- 별도의 routing predictor가 필요 없다.
- 첫 번째 miss의 fetch는 일반적으로 숨길 수 없다.
- 이후 fetch는 이전 Expert compute와 일부 또는 전부 overlap될 수 있다.

## 2.3 Prefetch depth

미래에 사용할 Expert를 몇 개까지 동시에 준비하는지를 의미한다.

- `depth=0`: fetch 후 대기, compute, 다음 fetch
- `depth=1`: 현재 Expert compute 중 다음 Expert 1개 fetch
- `depth>1`: 여러 future Expert를 queue에 미리 올림

본 핵심 실험은 `depth=1`로 고정한다.

## 2.4 Serial per-Expert execution

한 layer에서 활성화된 Expert를 하나씩 순서대로 실행한다.

```text
Expert e0의 token들을 모음
→ e0 MLP 실행
→ Expert e1 MLP 실행
→ Expert e2 MLP 실행
→ ...
```

특성:

- compute stream은 1개다.
- 한 시점에 하나의 Expert MLP만 실행한다.
- Expert H2D copy stream은 별도로 둘 수 있다.
- 현재 Expert 1개와 다음 Expert 1개만 있으면 pipeline을 구성할 수 있다.
- 작은 Expert별 GEMM이 많으면 kernel launch 및 작은-M GEMM 비효율이 커질 수 있다.

`serial`은 Expert MLP 내부에 kernel이 하나만 있다는 뜻이 아니다.  
하나의 Expert에서도 gate/up projection, activation, down projection 등 여러 kernel이 실행될 수 있다. 다만 **서로 다른 Expert의 MLP를 동시에 실행하지 않는다**는 의미다.

## 2.5 Individual GEMM

Expert마다 별도의 GEMM 호출을 발행한다.

예를 들어 활성 Expert가 32개라면, projection 종류별로 여러 개의 작은 GEMM 호출이 반복된다.

장점:

- 구현이 단순하다.
- Expert weight가 하나씩 도착하는 streaming과 잘 맞는다.
- transient slot 2개로 실행 가능하다.

단점:

- Expert당 token 수 `M_e`가 작을 때 GPU 활용도가 낮을 수 있다.
- kernel launch 수와 host dispatch overhead가 많아진다.
- H100의 Tensor Core를 충분히 채우지 못할 수 있다.

## 2.6 Multi-stream individual GEMM

여러 CUDA compute stream에서 서로 다른 Expert GEMM을 동시에 발행하는 방식이다.

```text
compute stream 0: Expert e0
compute stream 1: Expert e1
compute stream 2: Expert e2
...
```

가능한 이점:

- 각 Expert GEMM이 작아 GPU 자원을 일부만 사용할 때 concurrent kernel execution으로 utilization이 높아질 수 있다.

제약:

- 동시에 실행할 Expert weight들이 모두 GPU에 있어야 한다.
- compute stream 수만 늘린다고 실제 동시 실행이 보장되지는 않는다.
- GEMM들이 이미 GPU 자원을 많이 사용하면 서로 경쟁할 수 있다.
- transient Expert 공간이 2개보다 더 필요할 수 있다.

본 핵심 실험에는 포함하지 않는다.

## 2.7 Batched GEMM

여러 GEMM을 하나의 batched API로 호출하는 방식이다.

적합한 경우:

- 각 Expert의 matrix shape가 동일하거나
- 일정한 stride로 저장되어 있거나
- token 수를 padding하여 동일한 `M`으로 맞출 수 있는 경우

MoE에서는 Expert별 token 수 `M_e`가 다르므로 padding 낭비가 생길 수 있다.

본 핵심 실험에는 포함하지 않는다.

## 2.8 Grouped GEMM

서로 독립적인 여러 Expert GEMM을 하나의 grouped operation으로 묶어 실행하는 방식이다.

예:

```text
Expert e0: M0 × K  @  K × N
Expert e1: M1 × K  @  K × N
Expert e2: M2 × K  @  K × N
...
```

`M0`, `M1`, `M2`가 달라도 하나의 grouped kernel 또는 grouped scheduling 경로에서 처리할 수 있다.

가능한 이점:

- 여러 작은 GEMM의 launch overhead를 줄인다.
- 서로 다른 Expert 문제를 GPU 전체에 함께 scheduling한다.
- 작은 `M_e`에서 individual GEMM보다 occupancy와 Tensor Core 활용도가 좋아질 수 있다.

중요한 비용:

- 한 grouped GEMM에 포함되는 모든 Expert weight가 계산 시작 전에 GPU에 있어야 한다.
- group size가 `g`이면 최소 `g`개의 Expert weight가 동시에 필요하다.
- 다음 group까지 double buffering하려면 최대 `2g`개의 transient slot이 필요하다.
- 따라서 grouped GEMM은 Expert capacity 자체를 바꾸며, 단순한 `streaming-2`와 동일 조건이 아니다.

## 2.9 Fused MoE kernel

Token dispatch, Expert GEMM, activation, combine 등의 여러 단계를 하나의 fused path로 통합하는 방식이다.

이는 grouped GEMM보다 더 넓은 최적화이며, 본 1차 실험 범위에서 제외한다.

---

# 3. 1차 실험에서 고정할 실행 방식

핵심 실험에서는 cache organization만 비교하기 위해 다음을 고정한다.

```yaml
prefetch_mode: current_layer_one_ahead
prefetch_depth: 1
expert_execution: serial_per_expert
compute_streams: 1
copy_streams: 1
grouped_gemm: false
batched_gemm: false
fused_moe: false
global_lru: false
transient_expert_slots: 2
```

## 왜 prefetch depth 1을 기본으로 사용하는가

2-slot 구조의 의미가 다음 Expert를 미리 가져오는 double buffering에 있기 때문이다.

`prefetch_depth=0`이면 두 번째 slot의 필요성이 작다. 따라서 다음 두 가지를 구분한다.

### 메인 baseline

```text
stream2_one_ahead
= transient 2 slots + one-ahead H2D overlap
```

### Micro-ablation

```text
stream1_no_prefetch
= transient 1 slot + fetch/compute 직렬화
```

`stream1_no_prefetch`는 prefetch overlap의 효과를 측정하기 위한 보조 실험일 뿐, 메인 operating curve에는 넣지 않는다.

## 왜 grouped GEMM을 1차 실험에서 끄는가

Grouped GEMM을 켜면 다음 두 변수가 동시에 바뀐다.

1. Expert compute 효율
2. 동시에 필요한 transient Expert capacity

그러면 `KV capacity vs layer-local Expert residency`의 효과와 `GEMM implementation`의 효과를 분리하기 어렵다.

따라서:

- **Phase A**: serial individual GEMM으로 cache policy와 HBM trade-off를 측정
- **Phase B**: 대표 operating point에서 grouped GEMM을 별도 비교

순서로 진행한다.

---

# 4. 하드웨어 및 모델 설정

```yaml
gpu: NVIDIA H100
model: Qwen3-30B-A3B
dtype: bfloat16
num_moe_layers: 48
num_experts_per_layer: 128
router_top_k: 8

attention_device: gpu
kv_offload: false
prefix_sharing: false
continuous_batching: false

expert_weights_location: cpu
host_memory_mode: pinned_or_pinned_staging
decode_mode: forced_token_replay
routing_mode: forced_routing_replay
```

## 환경에서 반드시 기록할 값

H100 variant와 서버 구성을 hard-code하지 말고 runtime에서 기록한다.

```text
GPU name
GPU total HBM
H100 PCIe 또는 H100 SXM
CUDA runtime 및 driver version
PyTorch version
CPU model
NUMA topology
GPU와 CPU NUMA affinity
PCIe link generation/width
cudaDeviceProp.asyncEngineCount
pinned H2D bandwidth
```

---

# 5. HBM accounting

기존의 특정 GPU budget을 재사용하지 않는다.

먼저 다음을 측정한다.

```text
Total HBM
- Dense resident weights
- CUDA context 및 library workspace
- Attention workspace
- Router/dispatch workspace
- Activation workspace
- Allocator fragmentation reserve
- Safety margin
= KV + Expert residency에 사용 가능한 budget
```

기본 safety margin:

```yaml
hbm_safety_margin_gib: 2
```

실제 환경에서 OOM probe 결과에 따라 조정 가능하지만 모든 policy에서 동일하게 사용한다.

---

# 6. 실제 Expert 크기 측정

`9 MB`를 코드에 고정하지 않는다.

각 layer/expert의 모든 weight tensor에 대해 다음을 계산한다.

```python
expert_bytes = sum(
    tensor.numel() * tensor.element_size()
    for tensor in expert_weight_tensors
)
```

확인 항목:

```text
gate projection 포함 여부
up projection 포함 여부
down projection 포함 여부
bias 포함 여부
padding/alignment 포함 여부
모든 layer/expert의 크기가 동일한지
```

근사적으로 Expert 하나가 9 MB일 경우:

| k/layer | Persistent Expert 수 | 근사 persistent 용량 |
|---:|---:|---:|
| 0 | 0 | 0 GB |
| 1 | 48 | 0.43 GB |
| 2 | 96 | 0.86 GB |
| 4 | 192 | 1.73 GB |
| 8 | 384 | 3.46 GB |
| 16 | 768 | 6.91 GB |
| 32 | 1,536 | 13.82 GB |
| 48 | 2,304 | 20.74 GB |
| 64 | 3,072 | 27.65 GB |
| 96 | 4,608 | 41.47 GB |
| 128 | 6,144 | 55.30 GB |

Transient 2 slots의 용량은 별도로 더한다.

---

# 7. LMSYS-Chat-1M workload

## 7.1 기본 workload

```yaml
dataset: LMSYS-Chat-1M
input_tokens: 4096
output_tokens: 256
peak_tokens_per_request: 4352
prefix_sharing: false
```

Secondary stress workload:

```yaml
input_tokens: 8192
output_tokens: 256
peak_tokens_per_request: 8448
```

4K/256을 먼저 완료한 후 8K/256을 실행한다.

## 7.2 Sample 조건

각 request는 서로 다른 `conversation_id`를 사용한다.

필터:

```text
English 우선
moderation flagged sample 제외
conversation_id 중복 제외
Qwen tokenizer 기준 input source가 4096 tokens 이상
target assistant text가 Qwen tokenizer 기준 256 tokens 이상
```

## 7.3 Input 구성

```text
원본 conversation에서 target assistant turn 제거
→ 이전 conversation history와 마지막 user turn을 Qwen chat template에 적용
→ 가장 오래된 history부터 truncate
→ 마지막 user turn 유지
→ 정확히 4096개의 실제 token으로 구성
```

Padding으로 4096을 맞추지 않는다.

## 7.4 Output 구성

1차 실험은 데이터셋의 assistant text를 Qwen tokenizer로 변환한 뒤 첫 256개 token을 teacher-forced continuation으로 사용한다.

```text
assistant text
→ Qwen tokenizer
→ 처음 256 token 저장
→ 모든 policy에서 동일 token forced replay
```

이 방식의 목적은 output quality 평가가 아니라 동일한 routing/computation trace를 재현하는 것이다.

보조 검증에서는 Qwen3가 생성한 256-token trace를 별도로 사용한다.

## 7.5 Split

```yaml
calibration_requests: 256
evaluation_requests: 1200
```

- Calibration: Permanent Expert 선택
- Evaluation: 최종 cache/throughput 평가
- conversation_id는 split 간 중복 금지

저장 형식:

```json
{
  "conversation_id": "...",
  "input_ids": [],
  "forced_output_ids": [],
  "input_length": 4096,
  "output_length": 256
}
```

---

# 8. KV 크기 측정

Qwen3-30B-A3B의 이론적 KV bytes/token을 참고하되 실제 runtime tensor를 기준으로 측정한다.

기록:

```text
logical KV bytes/token
allocated KV bytes/token
peak KV bytes/request
allocator reserved bytes
fragmentation overhead
```

Peak KV는 `input_len + output_len` 전체를 기준으로 사전 할당한다.

```text
4K/256 workload:
peak sequence length = 4352
```

Decode 도중 KV가 증가해 OOM이 발생하지 않도록 시작 전에 peak capacity를 확보한다.

---

# 9. Cache policy

## 9.1 Policy A: `stream2`

Persistent Expert cache가 없다.

```text
Persistent slots: 0
Transient slots: 2
Global LRU: 없음
```

동작:

1. 현재 layer Router 결과에서 active Expert 목록 생성
2. active Expert 실행 순서를 고정
3. 첫 miss Expert를 Slot A에 fetch
4. Slot A의 Expert를 계산하는 동안 다음 miss Expert를 Slot B에 fetch
5. compute/copy event를 기다린 뒤 Slot A/B 역할 교체
6. 계산 완료 Expert는 다음 step을 위해 유지하지 않음

이 정책이 KV extreme이다.

## 9.2 Policy B: `permanent_k`

각 layer에 calibration에서 고른 k개의 Expert를 고정한다.

```text
Persistent slots: 48 × k
Transient slots: 2
Global LRU: 없음
```

동작:

- Permanent hit: resident weight로 계산
- Permanent miss: transient 2-slot pipeline으로 fetch하고 계산한 뒤 discard
- Permanent Expert는 evaluation 동안 절대 eviction하지 않음
- Non-permanent Expert는 persistent admission하지 않음

### Permanent Expert 선정 기준

기본 fetch-aligned 점수:

```text
batch_step_union_presence(layer, expert)
= calibration의 (batch wave × decode step × layer) 중 해당 Expert가
  한 번 이상 필요했던 횟수
```

이 점수는 weight fetch를 얼마나 자주 방지할 수 있는지에 직접 대응한다.

추가 점수:

```text
token_assignment_count(layer, expert)  # 기존 presence 구현, baseline
reload_count_under_streaming(layer, expert)
```

구현 policy:

```text
permanent_batch_step_union_presence_topk
permanent_token_frequency_topk
permanent_oracle_topk
```

`oracle`은 evaluation trace를 미리 본 upper bound이며 실제 deployable policy가 아니다.

## 9.3 Policy C: `quota_lru_k`

각 layer에 k개의 local LRU slot을 제공한다.

```text
Persistent slots: 48 × k
Transient slots: 2
Global LRU: 없음
```

동작:

### Local hit

```text
해당 layer의 quota slot에서 Expert 계산
→ LRU timestamp 갱신
```

### Local miss

```text
transient slot으로 H2D
→ Expert 계산
→ 해당 layer의 LRU victim 선정
→ transient slot과 victim slot의 logical ownership 교환
```

주의:

- Admission을 위해 weight를 별도 D2D copy하지 않는다.
- physical slot의 역할/mapping만 교환한다.
- 각 layer resident count는 항상 k 이하이다.
- 다른 layer는 해당 quota를 빌려 쓸 수 없다.

### 실행 순서 고정

Active Expert 순서가 LRU survivor를 바꿀 수 있으므로:

```yaml
expert_execution_order: ascending_expert_id
```

로 고정한다.

Primary run은 이 순서를 사용한다. 다음 sensitivity control은 별도 결과로
반드시 기록한다.

```text
resident-hit-first ordering
miss bypass when quota is full
no-admission
window-frequency admission
random Expert order (3개 이상의 고정 seed)
```

Control 결과는 primary ascending-ID always-admit LRU와 구분한다.

## 9.4 `k=128`

Layer별 128 Expert가 모두 resident이므로:

- Permanent와 Quota-LRU가 동일한 full-resident 상태로 수렴한다.
- warm-up 이후 Expert H2D는 0이어야 한다.
- 결과는 `full_resident` 한 configuration으로만 실행한다.
- 초기 loading cost 포함/제외 결과를 각각 기록한다.

---

# 10. k sweep

## Trace simulator

```python
K_VALUES_TRACE = [0, 1, 2, 4, 8, 16, 32, 48, 64, 96, 128]
```

## H100 runtime 1차 sweep

```python
K_VALUES_RUNTIME = [0, 2, 4, 8, 16, 32, 64, 96, 128]
```

최적점 주변을 추가한다.

예:

```text
k=8과 k=16 사이가 유망 → k=12 추가
k=32와 k=64 사이가 유망 → k=48 추가
```

구성:

```text
k=0:
  stream2

k=2,4,8,16,32,64,96:
  permanent_k
  quota_lru_k

k=128:
  full_resident
```

---

# 11. Maximum feasible batch 탐색

각 policy와 k는 persistent Expert 용량이 다르므로 남는 KV 공간과 최대 batch가 다르다.

이론값만 사용하지 않고 실제 OOM probe로 결정한다.

## Probe 절차

```python
def find_max_feasible_batch(config):
    # binary search 또는 exponential search + binary search

    for candidate_batch in search_order:
        reset_runtime_state()
        clear_cuda_allocator()

        allocate_dense_runtime()
        initialize_expert_policy(config)
        preallocate_peak_kv(
            batch=candidate_batch,
            peak_seq_len=input_len + output_len,
        )
        allocate_attention_router_dispatch_workspace(candidate_batch)

        run_one_decode_step_with_real_routing()
        synchronize()

        if no_oom_and_all_invariants_hold:
            mark_feasible(candidate_batch)
        else:
            mark_infeasible(candidate_batch)

    return largest_feasible_batch
```

기록:

```text
policy
k
theoretical Bmax
measured Bmax
total HBM
static HBM
KV HBM
persistent Expert HBM
transient HBM
workspace HBM
peak allocated
peak reserved
```

---

# 12. Decode-only 실행

1차 primary metric은 decode-only다.

## 절차

```text
1. Input 4096 token의 KV를 준비
2. KV를 GPU에 적재
3. peak 4352 token capacity 확보
4. Policy state 초기화
5. Permanent Expert preload 또는 Quota empty state 준비
6. CUDA synchronize
7. Timer 시작
8. forced token/routing으로 256 decode steps 실행
9. CUDA synchronize
10. Timer 종료
```

Prefill 시간과 KV loading 시간은 primary decode timer에서 제외한다.

후속 실험에서 별도로 E2E를 측정한다.

---

# 13. Prefetch 구현 명세

## 13.1 Streams

```text
compute_stream: Expert GEMM 및 관련 kernel
copy_stream: CPU→GPU Expert H2D
```

Host source는 pinned memory 또는 pinned staging buffer여야 한다.

## 13.2 Events

각 transient slot에 다음 event를 둔다.

```text
copy_done_event[slot]
compute_done_event[slot]
```

규칙:

- compute stream은 해당 slot의 `copy_done_event`를 기다린 뒤 weight를 사용한다.
- copy stream은 해당 slot의 `compute_done_event`가 끝난 뒤에만 slot을 덮어쓴다.
- device-wide synchronize를 Expert마다 호출하지 않는다.

## 13.3 Current-layer one-ahead algorithm

```python
active_experts = ordered_active_experts(layer_routing)

for each expert position i:
    current = active_experts[i]
    next_miss = first future nonresident expert after i

    ensure current is resident or copied to current transient slot

    if next_miss exists:
        enqueue H2D(next_miss, other transient slot, copy_stream)

    wait current copy event on compute_stream
    launch current expert MLP on compute_stream

    record current compute-done event
    rotate transient slot roles
```

Resident Permanent/Quota hit의 compute 시간도 다음 miss를 prefetch하는 overlap window로 사용할 수 있다.

## 13.4 측정할 prefetch 지표

```text
total H2D time
exposed H2D stall
overlapped H2D time
overlap ratio
first-miss stall per layer
copy engine utilization
```

정의:

```text
overlap ratio
= 1 - exposed_H2D_stall / total_H2D_duration
```

단, 타임라인 중첩 계산은 Nsight Systems 또는 CUDA event 기반 interval 분석으로 검증한다.

---

# 14. Serial Expert execution 명세

한 layer의 token을 Expert별로 모은 뒤 다음을 수행한다.

```python
for expert_id in sorted(active_expert_ids):
    token_indices = routed_tokens[expert_id]
    x_e = gather(hidden_states, token_indices)

    # Individual Expert MLP
    gate = x_e @ W_gate[expert_id]
    up = x_e @ W_up[expert_id]
    hidden = silu(gate) * up
    y_e = hidden @ W_down[expert_id]

    scatter_add(output, token_indices, y_e)
```

실제 구현에서 gate/up이 fused GEMM이면 그대로 사용해도 된다.  
단, 서로 다른 Expert를 하나의 grouped GEMM으로 묶지 않는다.

기록:

```text
active Expert count/layer-step
tokens per active Expert
Expert별 M_e
Expert GEMM kernel count
Expert compute time
host enqueue time
```

---

# 15. Grouped GEMM 후속 실험

Grouped GEMM은 본 핵심 operating curve 완료 후 수행한다.

## 15.1 질문

> Layer-local Expert residency의 최적점이 작은 individual GEMM overhead 때문에 왜곡되어 있는가?

## 15.2 대표 지점

```text
k=0
1차 실험의 최적 k
k=128
```

## 15.3 Group size sweep

```python
GROUP_SIZES = [2, 4, 8, 16]
```

## 15.4 공정한 transient capacity

Group size가 `g`일 때:

### Single-buffer group execution

```text
현재 group의 g Experts resident
→ g transient slots 필요
```

### Double-buffer group execution

```text
현재 group g Experts 계산
동시에 다음 group g Experts fetch
→ 2g transient slots 필요
```

따라서 grouped GEMM 실험에서는 다음을 명시적으로 비교한다.

```text
serial_stream2
grouped_gemm_g_single_buffer
grouped_gemm_g_double_buffer
```

모든 configuration에서 그에 맞는 transient HBM을 차감하고 Bmax를 다시 계산한다.

## 15.5 Grouped GEMM에서 측정할 값

```text
group size
number of grouped launches
tokens per Expert
grouped GEMM duration
individual GEMM total duration
achieved TFLOP/s
kernel launch count
host enqueue time
additional transient HBM
Bmax reduction
final throughput
```

## 15.6 해석

### Grouped GEMM이 compute만 줄이고 throughput이 거의 안 오름

- workload가 H2D fetch-bound일 가능성이 큼

### Grouped GEMM으로 작은-k가 크게 개선됨

- 기존 small-batch penalty가 cache 문제가 아니라 작은 individual GEMM/launch overhead의 영향이었을 수 있음

### Grouped GEMM이 좋지만 Bmax 감소로 E2E가 악화됨

- compute efficiency와 KV capacity의 새로운 trade-off가 발생함

---

# 16. 측정 지표

## 16.1 Memory

```text
Bmax
peak KV bytes
persistent Expert bytes
transient Expert bytes
workspace bytes
peak allocated HBM
peak reserved HBM
```

## 16.2 Expert traffic

```text
Expert H2D fetch count
Expert H2D bytes
Expert H2D bytes/generated token
compulsory load count
refetch count
refetch ratio
```

Refetch 정의:

> 동일한 `(layer_id, expert_id)`가 과거 GPU에 적재된 적이 있지만 discard/eviction 이후 다시 H2D된 경우

## 16.3 Cache behavior

Permanent:

```text
permanent hit count
permanent coverage
per-layer permanent utilization
```

Quota-LRU:

```text
local hit count
local miss count
local hit rate
per-layer hit rate
per-layer eviction count
quota utilization
resident lifetime
```

## 16.4 Performance

```text
decode throughput, generated tokens/s
fixed-workload decode makespan
decode ms/generated token
attention time
router time
Expert compute time
total H2D duration
exposed H2D stall
host/idle time
```

## 16.5 Derived metrics

```text
Tokens per Expert Fetch
= total Expert-token assignments / Expert H2D fetch count
```

```text
Refetch Savings per Reserved GB
= (stream2 refetch bytes - policy refetch bytes)
  / persistent Expert capacity in GB
```

```text
Batch Cost
= Bmax(k=0) - Bmax(k)
```

```text
Throughput Gain
= throughput(policy, k) / throughput(stream2)
```

---

# 17. 두 종류의 throughput 평가

## 17.1 Maximum-feasible-batch

각 configuration에서 자신이 허용하는 Bmax를 사용한다.

이 결과가 실제 HBM trade-off의 primary result다.

## 17.2 Fixed-batch control

비교 대상들이 공통으로 실행 가능한 batch에서 측정한다.

예:

```text
B_common = min(Bmax of compared configurations) = 40
```

이 결과는 batch 차이를 제거하고 cache/residency 자체의 효과를 보여준다.

## 17.3 Fixed-workload makespan

고정된 Evaluation 1200-request prefix 전체를 각 configuration의 Bmax로 처리한다.

```text
waves = ceil(1200 / Bmax)
```

마지막 partial wave도 포함한다.

보고:

```text
steady-state full-batch throughput
fixed 1200-request makespan
cold-start 포함/제외
```

---

# 18. Cold-start와 warm-state

## Permanent-k

두 결과를 모두 기록한다.

```text
cold-start:
permanent Expert preload 시간 포함

steady-state:
permanent preload 완료 후 decode만 측정
```

## Quota-LRU-k

```text
cold-start:
모든 local quota empty

warm-state:
이전 wave의 quota state 유지
```

## Stream2

Persistent state가 없으므로 cold/warm 차이가 거의 없어야 한다.

---

# 19. 필수 그래프

## Graph 1. HBM operating curve

```text
x-axis: k, resident Experts per layer
y-axis: maximum-feasible-batch decode throughput
lines:
  Permanent-k
  Quota-LRU-k
points:
  k=0 stream2
  k=128 full-resident
```

## Graph 2. Bmax curve

```text
x-axis: k
y-axis: measured maximum feasible batch
```

## Graph 3. Expert traffic

```text
x-axis: k
y-axis: Expert H2D GB/generated token
```

## Graph 4. Refetch decomposition

```text
x-axis: k
stack:
  compulsory load
  refetch
```

## Graph 5. Runtime breakdown

```text
Attention
Router
Expert compute
Exposed H2D stall
Host/idle
```

## Graph 6. Per-layer result

```text
x-axis: layer index
y-axis:
  refetch count
  hit rate
  traffic reduction
```

---

# 20. Correctness 및 invariant

모든 policy에서 동일 request에 대해 다음이 동일해야 한다.

```text
forced output token IDs
forced routing Expert IDs
token/layer별 top-8 routing
Expert execution order
```

## Stream2 invariant

```text
persistent Expert count == 0
transient slot count == 2
Expert는 layer/step 종료 후 cache hit으로 재사용되지 않음
```

## Permanent invariant

```text
layer별 pinned count == k
pinned Expert eviction == 0
non-pinned Expert persistent admission == 0
```

## Quota-LRU invariant

```text
layer별 resident count <= k
cross-layer quota borrowing == 0
global LRU structure == 없음
admission을 위한 D2D weight copy == 0
```

## Full-resident invariant

```text
초기 loading 이후 Expert H2D fetch == 0
```

---

# 21. 권장 코드 구조

```text
experiments/
├── configs/
│   ├── h100_lmsys_4k256.yaml
│   ├── h100_lmsys_8k256.yaml
│   └── policy_sweep.yaml
├── data/
│   ├── prepare_lmsys.py
│   └── validate_fixed_lengths.py
├── trace/
│   ├── collect_forced_routing_trace.py
│   └── trace_schema.py
├── runtime/
│   ├── expert_slot.py
│   ├── transient_double_buffer.py
│   ├── stream2_policy.py
│   ├── permanent_policy.py
│   ├── quota_lru_policy.py
│   ├── prefetch_scheduler.py
│   └── serial_expert_executor.py
├── benchmark/
│   ├── characterize_h100.py
│   ├── measure_expert_bytes.py
│   ├── measure_kv_bytes.py
│   ├── measure_h2d.py
│   ├── find_max_batch.py
│   └── run_residency_sweep.py
├── grouped_gemm_followup/
│   ├── grouped_executor.py
│   └── run_group_size_sweep.py
├── analysis/
│   ├── aggregate.py
│   └── plot.py
└── results/
    ├── environment.json
    ├── memory_breakdown.csv
    ├── bmax.csv
    ├── runtime.csv
    ├── cache_events.parquet
    ├── per_layer.csv
    └── summary.md
```

`global_lru_policy.py`는 만들지 않는다.

---

# 22. 구현 순서

## Phase 0. H100 characterization

- H100 variant/HBM 확인
- 실제 Expert bytes 측정
- 실제 KV bytes/token 측정
- pinned H2D bandwidth 측정
- 1 Expert copy와 연속 다수 Expert copy bandwidth 측정
- compute/copy overlap 가능성 확인

## Phase 1. Dataset

- LMSYS 4K/256 calibration 256개
- LMSYS 4K/256 evaluation 고정 prefix 1200개
- exact token length 검증

## Phase 2. Trace

- forced token trace 저장
- token/layer/top-8 routing 저장
- 모든 policy에서 replay 가능한 schema 구축

## Phase 3. Stream2 baseline

- `stream1_no_prefetch` microbaseline
- `stream2_one_ahead` main baseline
- correctness 및 overlap 검증

## Phase 4. Cache policies

- Permanent-k 구현
- Quota-LRU-k 구현
- Global LRU는 구현하지 않음

## Phase 5. Trace sweep

```text
k = 0,1,2,4,8,16,32,48,64,96,128
```

- hit/miss/refetch/H2D 예측
- 유망 k 확인

## Phase 6. H100 runtime sweep

```text
k = 0,2,4,8,16,32,64,96,128
```

- 각 configuration Bmax 탐색
- decode-only throughput
- fixed-workload makespan

## Phase 7. E2E validation

- 대표 k에서 prefill + decode 전체 실행
- decode-only 결론이 전체 workload에서도 유지되는지 확인

## Phase 8. Grouped GEMM follow-up

- `k=0`, 최적 k, `k=128`
- group size `2,4,8,16`
- transient capacity와 Bmax 재계산
- serial individual GEMM과 비교

---

# 23. 1차 완료 기준

다음이 모두 충족되면 핵심 실험 완료로 간주한다.

- Global LRU 없이 3개 policy가 정상 동작
- 모든 policy에서 forced token/routing 동일
- `k=0`부터 `k=128`까지 실제 HBM accounting 완료
- configuration별 measured Bmax 확보
- Permanent/Quota의 refetch와 H2D bytes 측정
- exposed H2D stall과 decode throughput 측정
- maximum-batch 및 fixed-batch 결과 모두 확보
- Nsight Systems로 `k=0`, 최적 k, `k=128` 타임라인 검증
- 4K/256 primary 결과 완료

---

# 24. 최종 보고 문장

> Under a fixed H100 HBM budget, we sweep from two-slot Expert streaming to full per-layer Expert residency. All remaining HBM is assigned to GPU KV cache, and each configuration runs at its measured maximum feasible batch. We compare static per-layer permanent placement with adaptive per-layer LRU quotas, without using a global Expert cache, to determine whether cyclic refetch reduction outweighs the loss of KV-resident batch capacity.

한국어:

> 동일한 H100 HBM에서 2-slot Expert streaming부터 전체 Expert residency까지 layer별 Expert 공간을 변화시키고, 남는 HBM은 모두 GPU KV cache에 할당한다. Global Expert cache 없이 static permanent와 layer-local LRU를 비교하여, 반복 refetch 감소가 KV-resident batch 감소보다 큰 throughput 이득을 만드는지 확인한다.
