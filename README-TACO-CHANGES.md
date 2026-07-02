# TACO Changes

本文件记录 taco-sglang 的混合 main 相对社区上游 SGLang（`sgl-project/sglang`）追加的私有优化与特性，是一份随版本周期滚动更新的长期清单。

## 分支模型

我们以「日期周期」维护混合 main，每个周期产出一个 `main-<yyyymmdd>` 分支：

1. 从社区 `origin/main` 取当期最新代码，作为该周期的 rebase 基点。
2. 在 `mixed-main-<yyyymmdd>` 上，将我们的私有补丁栈 rebase 到该基点之上（线性历史，无 merge commit）。
3. `mixed-main-<yyyymmdd>` 合入 `main-<yyyymmdd>` 后，`main-<yyyymmdd>` 即为该周期正式的混合 main。

因此每个周期的 `main-<yyyymmdd>` = 某个社区基点 commit + 一组私有 commit。本文件的「当前周期」小节记录最新周期的基点与特性列表；进入新周期时更新该小节（历史周期可按需归档到文末）。

`mixed-main-<yyyymmdd>` 与合入后的 `main-<yyyymmdd>` 内容一致，下文对二者统称。

---

## 当前周期：`20260630`

### 社区基点（Rebase Base）

| 项 | 值 |
|---|---|
| 社区仓库 | `https://github.com/sgl-project/sglang.git` (`origin/main`) |
| 基点 commit id | `c6a7c98ae429760ed3b2df8d3a11600c3855d74a` |
| 基点日期 | 2026-06-30 |
| 基点标题 | `[NPU]GLM-4.7-Flash optimize with fused kernels (#29509)` |

即 `main-20260630` = 社区 `c6a7c98ae4` + 下列私有 commit。

### 相比社区追加的特性（按提交先后顺序）

以下按 commit 从旧到新排列，顺序即开发时间线。相邻且属于同一功能的 commit 合并描述；不相邻的即使功能相关也分开列出（例如末尾的 Perf CI 更新与前面的 Perf CI 初建并非同一批工作）。

#### 1. TACOps Fused MoE — `29b15f17ec73` → `232d2edaa8a6` → `7e47c07f821c`

接入 TACOps 融合 MoE 算子，并引入 `3rdparty/tacops` 子模块（pin 到 `68e4952`，移除 opforge 依赖）。

- `29b15f17ec73` add tacops fused moe — `layers/quantization/fp8.py`
- `232d2edaa8a6` chore: add 3rdparty/tacops submodule (68e4952, rm opforge deps) — `.gitmodules`, `3rdparty/tacops`
- `7e47c07f821c` fix codes — `layers/quantization/fp8.py`

#### 2. DSV4 fp8 paged MQA logits preshuffle decorator — `db165b2e782e`

新增 `fp8_paged_mqa_logits_preshuffle_decorator`（MR !21）。改动 `jit_kernel/dsv4/topk.py`、`layers/attention/dsv4/indexer.py`。

#### 3. OpForge MHC integration — `a6092d634c99`

DeepSeek-V4 的 MHC 集成（MR !22）。改动 `models/deepseek_v4.py`。

#### 4. OpForge → TACOps 改名 — `24443ac14955`

将 opforge 相关引用统一改名为 tacops（MR !32）。改动 `jit_kernel/dsv4/topk.py`、`layers/attention/dsv4/indexer.py`、`models/deepseek_v4.py`。

#### 5. Perf CI 初建 — `b87645c388ae` → `8070ffc2e15e`

搭建 main_amd 的 MR 性能流水线。

- `b87645c388ae` add mr perf ci to main_amd (MR !37) — `.ci/perf-config.yml`、`.ci/scripts/*.py`（cache_key / compare_report / decode_throughput / prefill_throughput）
- `8070ffc2e15e` main_amd mr ci add pipeline yml file (MR !39) — `.ci/auto-mr-perf.yml`、`.ci/perf-config.yml`

#### 6. AMD AITER FP4 MoE K-dim 对齐可配置 — `fa9648d214e2`

`[AMD] Make AITER FP4 MoE K-dim alignment configurable`。改动 `environ.py`、`layers/quantization/fp8.py`。

#### 7. AMD DSV4 indexer K-cache preshuffle 布局修复 — `1f02111de1a4`

`[AMD] Fix DSV4 indexer K-cache preshuffle layout`。改动 `jit_kernel/csrc/deepseek_v4/*.cuh`、`jit_kernel/dsv4/attn.py`、`jit_kernel/dsv4/compress.py`、`jit_kernel/triton_store_cache.py`、`layers/attention/dsa/index_buf_accessor.py`、`environ.py`。

#### 8. Decode server hybrid 模式（支持 raw 请求）— `f3baa6f5e115`

`feat(decode): hybrid mode for raw requests on decode server`。基于社区最新代码重新实现（源自 main_amd MR !19）。

- 开关：`--disaggregation-decode-accept-raw-requests`（bool，默认关闭）。
- 开启后，decode server 同时接受 disaggregated 请求（带 `bootstrap_room`）与 raw 请求（本地 prefill+decode），并镜像常规调度器的 chunked-prefill 记账，避免分块 raw EXTEND 被误当完成的 prefill 合并（静默数据损坏）及下一分块分配时 OOM。
- 改动 `disaggregation/decode.py`、`managers/scheduler.py`、`server_args.py`，新增 `test/registered/disaggregation/test_disaggregation_decode_hybrid.py`。

#### 9. Perf CI 更新 — `b4f615cb9a56`

`chore(ci): update perf CI for mixed-main-20260630`。由 9 个 CI 调优 commit 压缩而成（tacops 子模块处理改为可选、切换最新 AMD docker 镜像、更新机器 / env / server args）。改动 `.ci/auto-mr-perf.yml`、`.ci/perf-config.yml`。

### 子模块

| 子模块 | 路径 | Pin commit |
|---|---|---|
| tacops | `3rdparty/tacops` | `68e49529418b4ebd58d62e7fa0213930cf4fe065`（rm opforge deps, MR !5）|

### 快速核对命令

```bash
# 社区基点
git merge-base main-20260630 origin/main              # -> c6a7c98ae4

# 相比社区追加的全部 commit（按顺序）
git log --oneline --reverse c6a7c98ae4..main-20260630
```
