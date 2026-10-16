# sglang-flexkv rebase 到社区 main：冲突与解决方案

## 概要

| 项 | 值 |
|---|---|
| 操作 | squash FlexKV 独有改动后 `git rebase` 到社区 `main` |
| 社区 tip | `536cdb4b04`（2026-10-16） |
| FlexKV 原 tip | `702835134c`（`agent/flexkv-dsv4-main` / backup） |
| 分叉点 | `77983865d8`（2026-09-23） |
| 结果分支 | `rebase_main` → `32145dd5b8`（+ 测试对齐提交） |
| 备份分支 | `backup/flexkv-pre-rebase-20261016` |
| 策略 | **不以 70 个历史 commit 逐条 replay**；先 soft-reset 到 merge-base 压成 1 个 FlexKV commit，再 rebase 到社区 main，一次性解冲突 |

社区 `main` 已是 `rebase_main` 的祖先；FlexKV 能力以单 commit 叠在最新社区代码之上。

## 为何不用逐 commit rebase

- FlexKV 侧相对分叉约 **70** commits，其中大量 **merge commit**。
- 社区同期超前约 **1382** commits，且对 `scheduler` / `schedule_policy` / `mem_cache` 做了结构性重构。
- 逐条 rebase 会在同一批文件上反复冲突，且易丢失 merge 中已解决的语义。

## 冲突文件清单与决策

冲突共 **16** 个文件（`modify/delete` 计 1）。原则：**社区为底，迁回 FlexKV 钩子与适配层**。

### 1. 文档

| 文件 | 决策 | 说明 |
|---|---|---|
| `docs/docs.json` | **两边都留** | 保留社区 `layer_boundary`，并加入 `flexkv-hybrid-cache-eviction` |
| `docs/docs/developer_guide/overview.mdx` | **两边都留** | 两条 developer guide 链接并存 |
| `flexkv/README.md` | **采用 FlexKV** | 外部 connector / hybrid / restore ownership 文档 |

### 2. 与 FlexKV 无关或仅重命名

| 文件 | 决策 | 说明 |
|---|---|---|
| `flash_mla.py` | **社区** | docstring / `kv_format` 参数形态跟社区 |
| `deepseek_v4_backend.py` | **社区** | `get_swa_key_page_size()`（社区重命名，废弃 FlexKV 侧 `get_swa_kv_page_size`） |
| `deepseek_v4_memory_pool.py` | **社区** | 同上 API 命名 |
| `test_q8kv8_sparse_prefill_backend.py` | **社区** | 测试跟随 `get_swa_key_page_size` |
| `registry.py` | **社区** | 社区已有 `enable_flexkv → _flexkv_factory`；FlexKV 增强放在 `__init__.py` factory |

### 3. modify/delete

| 文件 | 决策 | 说明 |
|---|---|---|
| `flexkv_connector.py` | **删除** | 继续使用外部 `flexkv.integration.sglang.connector.FlexKVConnector` |
| `flexkv_comm.py` | **删除**（随 FlexKV 提交） | 同上，de-vendor |

### 4. 调度 / 缓存集成（高风险，手工合并）

| 文件 | 决策 | 说明 |
|---|---|---|
| `schedule_batch.py` | **社区 `prefix_len` + FlexKV `pending_restore_*`** | 社区已去掉 `prefix_indices`；保留 restore 世代/槽位字段 |
| `schedule_policy.py` | **社区逻辑 + FlexKV 钩子** | ① rematch 前 `has_uncommitted_restore` skip；② load-back 后按 `load_back_is_cache_owned` / `_flexkv_uncached_restore` 设置 `cache_protected_len`；③ 保留社区 `_kv_shard_reserve_scratch` |
| `scheduler.py` | **两边合并** | 保留社区 `budget_state()==OTHER → break`；加入 `has_uncommitted_restore → continue`；prefetch 条件加入 `enable_flexkv`；`batch_is_full` 保持社区 soft 语义（空闲可重试） |
| `common.py` | **社区门闩 + FlexKV SWA 参数** | `supports_prefix_sharing()` 前提下，若传入 `swa_num_tokens` 则转给 allocator |
| `unified_tree_core.py` | **社区结构 + FlexKV logical_progress** | 保留社区 `InternalStateBackup` / `finish_*`；`made_progress` 增加逻辑进度与 frees 判定 |
| `flexkv_radix_cache.py` | **FlexKV 实现 + API 适配** | 见下一节 |
| `test_prefill_adder.py` | **两边测试都留** | 社区 shard 测试 + FlexKV host-load admission 回归 |

### 5. 自动合并、仍需保留的 FlexKV 文件

无文本冲突，但属于 FlexKV 功能面：

- 新增：`flexkv_cache_lifecycle.py`、`flexkv_hybrid_radix_cache.py`、相关单测与 eviction 文档
- 增强：`storage/flexkv/__init__.py`（hybrid factory、外部 connector、EAGLE/streaming 校验）
- 其它 auto-merge：`environ.py`、`tokenizer_manager.py`、`metrics_reporter.py`、`allocator/swa.py`、`components/swa.py`、`base_prefix_cache.has_uncommitted_restore` 等

## 社区 API 迁移（合入后必须做的适配）

社区自 9 月底以来对 mem_cache 做了破坏性整理，FlexKV 适配层不能原样粘贴：

| 旧（FlexKV 分支） | 新（社区 main） | 处理 |
|---|---|---|
| `req.prefix_indices: Tensor` | `req.prefix_len: int` | 长度用 `prefix_len`；槽位用 `path_device_indices` / `prefix_device_indices` |
| `MatchResult.device_indices` | `MatchResult.device_prefix_len` | `_match_device_indices(match)` 从 path 重建 |
| `cache_finished_req` / `cache_unfinished_req` | `checkpoint(req, up_to=...)` | FlexKV 在 `checkpoint` 内 commit restore；仅 `req.finished()` 时 store |
| `init_load_back` → `(Tensor, node)` | → `(int, node)` | 返回 loaded token 数 |
| in-tree `FlexKVConnector` | 外部包 connector | 删除 in-tree connector/comm |

额外：

- `FlexKVRadixCache.load_back_is_cache_owned()`：IP 模式返回 `False`
- `FlexKVHybridRadixCache.load_back_is_cache_owned()`：恒 `False`
- Hybrid / Radix 的 store、restore ownership、prefetch、lifecycle 行为保留

## 验证建议

1. **静态**：相关模块已 `ast.parse` 通过。
2. **单测**（需 GPU/依赖环境）：
   - `test/registered/unit/mem_cache/test_flexkv_*.py`
   - `test/registered/unit/managers/test_flexkv_*.py`
   - `test_prefill_adder.py` 中 host-load / shard 用例
3. **冒烟**：`--enable-flexkv` / `--radix-cache-backend=flexkv` 启动；lookup → prefetch → restore → store 路径。
4. **注意**：FlexKV 单测已做机械 API 对齐（`checkpoint(up_to=...)`、`prefix_len` 等），部分断言（原 `device_indices` 张量比较、`is_insert=` 语义）可能仍需按新 API 精修；生产路径以 `checkpoint` / `prefix_len` 为准。

## 回滚

```bash
git checkout backup/flexkv-pre-rebase-20261016
# 或
git reset --hard backup/flexkv-pre-rebase-20261016
```

## 后续建议

1. 在目标硬件上跑通 FlexKV 单测并修齐剩余断言。
2. 将 `rebase_main` 推到 fork（需 force-with-lease，因历史已改写）。
3. 定期用「squash FlexKV delta → rebase 社区 main」方式同步，避免再积压上千社区提交。
