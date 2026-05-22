# 美股配对中性策略 Pipeline

基于 SEC 年报文本分析、收益率中性化、LLM 主观审查的全流程美股配对交易策略。

## 策略概述

在业务高度重合的两只股票之间构建多空组合，做多相对更优的公司、做空相对较弱的公司。通过对冲系统性风险，提取纯粹的相对价值 Alpha。

## Pipeline 流程

```
全美股 → 市值筛选(≥$5B) → 富途行业划分 → 行业内配对
    → 四维度文本Embedding → 文本相似度筛选
    → 收益率中性化 + 量价相关性筛选
    → 综合排序 + 集中度控制(单家≤2次)
    → LLM两阶段主观分析 → 闭式解权重优化
    → 最终持仓组合
```

## 环境配置

### 前置要求

- Python 3.11+
- FutuOpenD 网关运行中（用于行业分类，默认 127.0.0.1:11111）
- 环境变量 `CLOSEAI_API_KEY` 设置为 OpenAI 兼容 API 的密钥

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
- `industry.futu_host/port`: FutuOpenD 网关地址
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
| 1 | `step1_universe.py` | SEC EDGAR 获取全美股 + 市值筛选 | `universe.csv` |
| 2 | `step2_industry.py` | 富途 OpenAPI 行业板块分类 | `universe_with_industry.csv` |
| 3 | `step3_pairing.py` | 行业内生成所有配对 | `all_pairs.csv` |
| 4 | `step4_text_embedding.py` | 10-K 提取 + LLM 精炼 + 四维度 Embedding | `embeddings_*.npz/json` |
| 5 | `step5_text_similarity.py` | 加权余弦相似度计算与筛选 | `text_similarity.csv` |
| 6 | `step6_return_correlation.py` | 收益率中性化 + 多窗口相关性 | `return_correlation.csv` |
| 7 | `step7_candidate_selection.py` | 综合排序 + 集中度控制 | `candidates_for_llm.csv` |
| 8 | `step8_llm_review.py` | 两阶段 LLM 审查（重叠度 + 方向） | `llm_review_results.json` |
| 9 | `step9_weight_optimization.py` | 闭式解风险中性权重优化 | `final_portfolio.csv` |

## 核心方法论

### 文本四维度相似性（Step 4-5）

将每家公司的 SEC 年报业务描述精炼为四个竞争维度：
- 核心业务 (Core Business) - 权重 30%
- 主要产品/服务 (Main Product/Service) - 权重 30%
- 客户与渠道 (Customer/Channel) - 权重 25%
- 地理区域 (Geography) - 权重 15%

### 收益率中性化（Step 6）

滚动窗口 OLS 剥离三类系统性因子：
- 行业因子：市值加权行业收益（排除自身）
- 规模因子 (SMB)：小盘 vs 大盘收益差
- 波动率暴露：滚动标准差

### LLM 两阶段审查（Step 8）

- **第一阶段**：业务重叠度复审，分类为 Type 1（同一竞技场）或 Type 2（结构性分化）
- **第二阶段**：四维度方向性审查（业务分化、产品周期、财务状况、近期新闻）
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

- Step 1 的市值获取较慢（逐个 ticker 查询），首次运行预计 1-2 小时
- Step 2 需要 FutuOpenD 网关运行，频率限制约 5 秒/请求
- Step 4 的 SEC 10-K 提取和 LLM 精炼是最耗时的步骤
- Step 6 会缓存价格数据到 `outputs/price_cache.parquet`
- Step 8 的 LLM 调用有并发控制，默认 10 个 worker
- 所有中间结果保存在 `outputs/` 目录，支持断点续跑
