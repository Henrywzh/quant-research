# A股行业/主题 ETF 轮动：Universe 确定（第一阶段工作流程）

目标：从你已整理的 ETF 列表与初筛字典（`sw_sector_etf` + `other_sector_etf`）出发，构建一个**可回测、可扩展、可维护**的 ETF Universe，并能稳定拉取历史行情（ETF/LOF）用于后续轮动信号与回测。

---

## 0. 明确本阶段输出（Deliverables）

本阶段结束，你应得到以下 3 个产物：

1) **Universe 配置文件（建议 YAML/JSON）**
- `universe_industry`: 申万一级行业 ETF（单一行业）映射表：`行业 -> 代码`
- `universe_theme`: 主题/大类 ETF 映射表：`大类 -> 代码`
- 每个标的附带 `instrument_type`：`ETF` 或 `LOF`（用于选择 akshare 接口）

2) **标准化行情数据（OHLCV）**
- 对每只标的拉取日频：`open/high/low/close/volume/amount` + `pct_chg/turnover`
- 输出统一列名与时间索引，形成一个 `MarketData` 风格的数据结构（例如 `close[date, ticker]`）

3) **Universe 质量报告（用于验证与迭代）**
- 覆盖行业数量、标的数量
- 每只标的：可用历史长度、缺失率、最近更新日期
- 流动性概览：近 60 日日均成交额（从东财接口可得）

---

## 1. Universe 分层：行业池 vs 主题池（先拆开）

你已给出两类初筛：

### 1.1 行业池（申万一级行业） `sw_sector_etf`
- 用于“行业轮动”的核心 universe
- 原则：每个行业尽量只保留 1 只“代表性、流动性更好”的 ETF（或 LOF）

### 1.2 主题/大类池 `other_sector_etf`
- 用于“主题轮动”或作为“行业轮动的对照/扩展实验”
- 原则：不要与行业池混用，否则风格漂移、回测解释性变差

本阶段建议：**先跑行业池**，主题池作为第二条研究线保留。

---

## 2. 标的类型识别：ETF vs LOF（决定用哪个接口）

你给出的代码里混有 ETF 与 LOF（例如 `161037`, `161724`, `502023` 这类更像 LOF）。

### 2.1 为什么必须识别类型
- ETF 用：`ak.fund_etf_hist_em`
- LOF 用：`ak.fund_lof_hist_em`
- 两者输出字段一致，但入口不同

### 2.2 识别策略（建议的优先级）
1) **优先用 akshare spot 列表做 membership 判断**
   - `ak.fund_etf_spot_em()` 是否包含该代码 → ETF
   - `ak.fund_lof_spot_em()` 是否包含该代码 → LOF
2) 若两者都不包含：标记为 `UNKNOWN`，先从 universe 中剔除（或另行处理）

输出：为每个代码打上 `instrument_type in {ETF, LOF}`。

---

## 3. 数据拉取规范：统一日频、统一复权口径

### 3.1 复权选择（研究默认）
- **优先 `hfq`（后复权）** 做回测收益计算与财富曲线
- 原因：后复权更接近长期总回报口径（你引用的文档也强调量化研究普遍用后复权）
- 注意：如果你后续要做“可交易价格”的滑点模拟，可同时拉取不复权 `adjust=""` 作为交易价参考

### 3.2 拉取参数（建议默认）
- `period="daily"`
- `start_date="20150101"`（让接口返回尽可能长，后面统一截断）
- `end_date=today`（收盘后更新；盘中不取当日 close）

---

## 4. 行情数据标准化：统一字段名与数据结构

将东财返回的中文列标准化为固定 schema：

- `date`（datetime）
- `open, high, low, close`（float）
- `volume`（int）
- `amount`（float）
- `pct_chg`（float）
- `turnover`（float）

并强制：
- 以 `date` 为索引升序排列
- 去重（同一日期多条取最后一条）
- 所有数值列转为 numeric（errors->NaN）

输出建议结构：
- `close_df`: shape = (dates, tickers)
- `open_df`, `high_df`, `low_df`, `volume_df`, `amount_df` 同理
- 或者长表：`df_long` columns = `[date, ticker, open, high, low, close, volume, amount, pct_chg, turnover]`

---

## 5. Universe 质量检查（必须做，否则回测会“假稳健”）

对每只标的做检查并产出报告：

1) **历史长度**：最早日期、有效交易日数
2) **缺失率**：close 缺失的比例（对齐到全体交易日）
3) **最新日期**：是否更新到最近交易日（避免数据断更）
4) **流动性**：近 60 日日均成交额 `amount_mean_60`
5) **异常检测**：
   - close <= 0
   - 日涨跌幅极端（可设阈值，例如 |pct_chg| > 20% 做提示）
   - 成交额长期为 0（可能停牌或数据异常）

根据检查结果做第一轮剔除规则：
- 历史长度 < 2 年：剔除（或降级为候选）
- 近 60 日日均成交额太低：剔除（阈值你后续定）
- 最新日期明显滞后：剔除（数据源问题）

---

## 6. 定稿 Universe（第一版 freeze）

完成检查后，输出第一版 universe：

### 6.1 行业池（建议 10–16 个行业）
- 每行业 1 只代表 ETF/LOF
- 允许少数行业缺失（宁缺毋滥）

### 6.2 主题池（建议 5–8 个主题）
- 作为后续扩展实验，不与行业池混跑

并保存为配置文件（建议）：
- `config/universe_cn_industry.yaml`
- `config/universe_cn_theme.yaml`

---

## 7. 进入下一阶段的前置条件（Gate）

只有当以下条件满足，才进入“轮动信号 + 回测”阶段：

- Universe 中每只标的均已成功拉到日频行情
- 数据结构统一、无明显缺失/断更
- 质量报告可解释，剔除规则清晰
- 行业池规模稳定在你预期范围（例如 12 或 16 个）

---

## 附：你当前的初筛输入（作为 Universe v0）

### 行业池（申万一级行业）
- 来自 `sw_sector_etf`（行业 -> 代码）

### 主题/大类池
- 来自 `other_sector_etf`（大类 -> 代码）

后续所有研究迭代都应基于这两个池的版本化配置（避免“回测结果不可复现”）。
