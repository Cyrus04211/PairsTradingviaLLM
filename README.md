# 美股配对中性策略 Pipeline

基于 SEC 年报文本分析、收益率中性化、LLM 主观审查的全流程美股配对交易策略。

## 策略概述

在业务高度重合的两只股票之间构建多空组合，做多相对更优的公司、做空相对较弱的公司。通过对冲系统性风险，提取纯粹的相对价值 Alpha。

## Pipeline 流程

```
全美股 → 市值筛选(≥$5B) → 富途行业划分 → 行业内配对
    → 截至 2026-05-08 的 252/504/1000 收益率中性化相关筛选
    → 三个窗口都进入前50%的 pair 才保留
    → 四维度文本Embedding → 文本相似度筛选(保留30%)
    → 集中度控制(单家≤2次)
    → LLM两阶段主观分析(每对最多2次调用) → 闭式解权重优化
    → 最终持仓组合
```

## 环境配置

### 前置要求

- Python 3.11+
- 默认使用项目内置的本地行业池（`data/sec_companies_gt_5B_marketcap_by_industry_plate_grouped/`）
- 若 `universe.local_grouped_industry_dir` 留空，Step 1 会退回到 SEC + 东方财富市值快照，Step 2 会退回到 FutuOpenD（默认 127.0.0.1:11111）
- 环境变量 `CLOSEAI_API_KEY` 设置为 OpenAI 兼容 API 的密钥
- Step 5 默认使用仓库内自带的 SEC 在线抽取 + `gpt-5.4` 摘要；只有在 `embedding.double_tower_root` 与 `embedding.reuse_text_artifact_roots` 被显式配置时，才会复用外部本地缓存加速

### 安装

```bash
cd pairs_trading_pipeline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 配置

编辑 `config.yaml` 中的参数，主要需要确认：
- `embedding.api_key_env`: API 密钥的环境变量名
- `universe.local_grouped_industry_dir`: 本地行业池目录；留空时才会走 SEC/Futu
- `industry.futu_host/port`: FutuOpenD 网关地址（仅在未使用本地行业池时需要）
- `embedding.sec_user_agent`: SEC 抽取时使用的 User-Agent，建议改成你自己的姓名和邮箱
- `embedding.double_tower_root` / `embedding.reuse_text_artifact_roots`: 可选的本地缓存复用路径；默认留空即可
- 各步骤的阈值参数

## 运行

### 完整 Pipeline

```bash
python run_pipeline.py
```

### 从指定步骤开始

```bash
python run_pipeline.py --start 4        # 从 step4 开始
python run_pipeline.py --start 4 --end 7 # 只跑 step4-7
python run_pipeline.py --step 8          # 只跑 step8
```

### 单步运行

```bash
python -m pipeline.step1_universe --config config.yaml --output-dir outputs
python -m pipeline.step2_industry --config config.yaml --output-dir outputs
# ... 以此类推
```

## 各步骤说明

| Step | 文件 | 功能 | 输出 |
|------|------|------|------|
| 1 | `step1_universe.py` | 本地行业池导入（默认）或 SEC EDGAR 获取全美股 + 市值筛选 | `universe.csv` |
| 2 | `step2_industry.py` | 复用预载行业板块（默认）或富途 OpenAPI 行业分类 | `universe_with_industry.csv` |
| 3 | `step3_pairing.py` | 行业内生成所有配对 | `all_pairs.csv` |
| 4 | `step4_curve_filter.py` | 按截至 `2026-05-08` 的 `252/504/1000` 收益率中性化相关系数打分，且三个窗口都进入前 50% 才保留 | `return_correlation.csv` |
| 5 | `step5_text_embedding.py` | 仅对 Step 4 保留下来的 ticker 做年报业务文本抽取 + LLM 精炼 + 四维度 Embedding；若配置了本地缓存则优先复用 | `embeddings_*.npz/json` |
| 6 | `step6_text_similarity.py` | 对 Step 4 保留的 pairs 计算加权余弦相似度，并保留前 30% | `text_similarity.csv` |
| 7 | `step7_candidate_selection.py` | 只做集中度控制（单家公司最多出现 2 次） | `candidates_for_llm.csv` |
| 8 | `step8_llm_review.py` | 两阶段 LLM 审查：第 1 次调用审查业务重叠度，第 2 次调用一次性返回四个方向维度 | `llm_review_results.json` |
| 9 | `step9_weight_optimization.py` | 闭式解风险中性权重优化 | `final_portfolio.csv` |

## 核心方法论

### 文本四维度相似性（Step 5-6）

将每家公司的 SEC 年报业务描述精炼为四个竞争维度：
- 核心业务 (Core Business) - 权重 30%
- 主要产品/服务 (Main Product/Service) - 权重 30%
- 客户与渠道 (Customer/Channel) - 权重 25%
- 地理区域 (Geography) - 权重 15%

### 收益率中性化（Step 4）

先对收益率做滚动 OLS 中性化，再在截至 `2026-05-08` 的 `252/504/1000` 三个窗口上分别按相关系数排序；只有在三个窗口里都进入前 50% 的 pair 才会保留。中性化时剥离三类系统性因子：
- 行业因子：市值加权行业收益（排除自身）
- 规模因子 (SMB)：小盘 vs 大盘收益差
- 波动率暴露：滚动标准差

### LLM 两阶段审查（Step 8）

- **第一阶段**：业务重叠度复审，分类为 Type 1（同一竞技场）或 Type 2（结构性分化）
- **第二阶段**：四维度方向性审查（业务分化、产品周期、财务状况、近期新闻）在一次调用中统一返回
- **共识合成**：维度间方向一致 → 高置信度；方向冲突 → 排除

### 权重优化（Step 9）

闭式解最小化风险因子暴露的加权平方和：
```
x* = Σ(λ_f · a_f · b_f) / Σ(λ_f · a_f²)
W_long = x*, W_short = -(1-x*), x* ∈ [0.2, 0.8]
```

## 输出示例

`final_portfolio.csv` 包含：
```
long_ticker, short_ticker, industry_plate, pair_type, weight_long, weight_short, review_confidence
MRK, PFE, Drug Manufacturers, type_1, 0.55, -0.45, 0.87
NOW, DOCU, Software - Application, type_2, 0.48, -0.52, 0.82
...
```

## 注意事项

- 使用项目内置行业池时，Step 1/2 不需要调用东方财富/Futu，启动会快很多
- 若切回 SEC/Futu 模式，Step 1 需要抓取 SEC ticker 和东方财富市值快照，Step 2 需要 FutuOpenD 网关运行且频率限制约 5 秒/请求
- Step 4 会使用 `akshare.stock_us_daily` 下载并缓存收盘价到 `outputs/price_cache.parquet`
- Step 5 在默认配置下不依赖外部仓库；若你本地有 `double_tower/strategy_model` 的摘要与 embedding 缓存，可以在 `config.yaml` 中填入路径加速
- Step 8 对每个 pair 最多发起 2 次 LLM 调用，且整体并发默认 10 个 worker
- 所有中间结果保存在 `outputs/` 目录，支持断点续跑

## 仓库内容边界

- 仓库默认保留 `data/sec_companies_gt_5B_marketcap_by_industry_plate_grouped/` 这份已经整理好的行业分组结果
- 本地运行生成的价格缓存、曲线筛选结果、SEC 抽取文本、业务摘要、embedding、LLM 审查结果等均不建议提交到仓库
