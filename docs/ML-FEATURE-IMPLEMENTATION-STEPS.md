# 预测类 (ML) 功能 — 具体落实步骤

## 一、功能目标

用现有订单/配送数据做预测，在 Streamlit 中展示，包含三类方向：

| 方向 | 说明 |
|------|------|
| **延迟预测** | 用订单/地区/品类/支付等特征，预测「预计延迟天数」 |
| **异常检测** | 找出异常延迟订单、异常地区或异常品类 |
| **需求/流量预测** | 按时间序列预测未来订单量或配送压力 |

---

## 二、前置准备

### 2.1 依赖

- 在 `requirements.txt` 中增加：`scikit-learn` 和（可选）`lightgbm`。
- 若做时间序列预测，可增加 `statsmodels` 或继续用 scikit-learn 做简单时序特征。

### 2.2 数据与目录

- 数据源：沿用现有 `get_fact_orders()` 得到的 `fact_orders` 表（已有 `delay_days`、`customer_state`、`product_category_mode`、`payment_type_mode`、`order_purchase_timestamp` 等）。
- 新建目录（可选）：`data/models/` 用于存放训练好的模型与特征元数据（如编码映射、特征列表）。

---

## 三、分步落实

### 阶段 1：特征工程模块 `src/features.py`

| 步骤 | 内容 | 说明 |
|------|------|------|
| 1.1 | 定义「延迟预测」特征 | 从 fact_orders 抽取：地区、品类、支付方式、运费、订单金额、购买月份/星期等；对分类变量做 LabelEncoder 或 OneHot，数值做标准化（可选）。 |
| 1.2 | 定义「异常检测」特征 | 可复用延迟预测特征；或按订单/地区/品类聚合（如平均延迟、订单数），得到用于异常打分的输入。 |
| 1.3 | 定义「需求/流量」特征 | 按日/周聚合订单量；构造时间特征：星期、是否节假日、滞后项、滑动均值等，供时序或回归模型使用。 |
| 1.4 | 统一接口 | 提供如 `build_delay_features(df)`、`build_anomaly_features(df)`、`build_demand_features(df)`，返回只含特征列 + 目标列（若有）的 DataFrame，便于 `src/models.py` 调用。 |

### 阶段 2：模型训练与推理模块 `src/models.py`

| 步骤 | 内容 | 说明 |
|------|------|------|
| 2.1 | 延迟预测模型 | 目标变量：`delay_days`（或二分类 `on_time`）。用 `sklearn` 的 `RandomForestRegressor`/`GradientBoostingRegressor` 或 LightGBM；实现 `train_delay_model(df)` 与 `predict_delay(features)`，并保存/加载模型与编码器。 |
| 2.2 | 异常检测 | 可选方案：基于延迟的统计阈值（如延迟 > 某分位数）；或 Isolation Forest / One-Class SVM 对「延迟+地区+品类」等特征做无监督异常打分；提供 `detect_anomalies(df)` 返回带异常标签或得分的 DataFrame。 |
| 2.3 | 需求/流量预测 | 按日聚合订单量，用简单模型（如 SARIMAX、线性回归+时序特征，或 sklearn 的 Ridge + 滞后/滑动特征）；实现 `train_demand_model(series_or_df)` 与 `predict_demand(horizon_days)`，返回未来若干天的预测值。 |
| 2.4 | 模型持久化 | 将训练好的模型、LabelEncoder/OneHot 的映射、特征名列表保存到 `data/models/`（如 joblib 或 pickle），便于 app 启动时加载，避免每次请求都训练。 |
| 2.5 | 可解释性 | 对树模型提供 `get_feature_importance(model)`，返回特征重要性 DataFrame，供 Streamlit「预测/模型」Tab 展示。 |

### 阶段 3：Streamlit 集成 — 新 Tab「预测/模型」

| 步骤 | 内容 | 说明 |
|------|------|------|
| 3.1 | 新增 Tab | 在 `app.py` 中与现有「Root Causes」「Data Table」并列，增加第三个 Tab：「预测/模型」。 |
| 3.2 | 子选项或单选 | 在「预测/模型」内用 `st.radio` 或 `st.selectbox` 切换：① 延迟预测 ② 异常检测 ③ 需求/流量预测。 |
| 3.3 | 延迟预测 UI | 支持两种输入方式：**筛选条件**（复用侧边栏或本 Tab 内日期/州/品类/支付方式）或**单笔订单**（从表格选 order_id 或手动选特征）；展示预测的延迟天数 + 特征重要性（柱状图或表格）。 |
| 3.4 | 异常检测 UI | 根据当前筛选条件对数据做异常检测；展示异常订单/地区/品类的列表或简单图表（如 Top 异常订单、按州/品类的异常数量）。 |
| 3.5 | 需求/流量预测 UI | 展示历史订单量曲线 + 未来 N 天的预测曲线；可选：输入「预测未来天数」的滑块。 |
| 3.6 | 模型加载与缓存 | 应用启动时用 `@st.cache_resource` 加载已保存的模型与编码器；若未训练过，提示「请先运行训练脚本」或在本 Tab 提供「训练模型」按钮，调用 `src.models` 的训练函数并写回 `data/models/`。 |

### 阶段 4：训练入口与文档

| 步骤 | 内容 | 说明 |
|------|------|------|
| 4.1 | 训练脚本 | 新建 `scripts/train_models.py`（或类似）：加载 fact_orders → 调用 `src.features` 构建特征 → 调用 `src.models` 训练三个模型并保存到 `data/models/`；可由命令行执行，便于定期重训。 |
| 4.2 | README 更新 | 在 README 中增加「预测类 (ML)」说明：如何安装依赖、如何运行训练、如何在 App 中使用「预测/模型」Tab。 |

---

## 四、建议实施顺序

1. **依赖 + 目录**：改 `requirements.txt`，建 `data/models/`。
2. **`src/features.py`**：先做延迟预测所需特征，再做异常与需求的特征。
3. **`src/models.py`**：先实现延迟预测（训练 + 推理 + 保存/加载 + 特征重要性），再异常检测，再需求预测。
4. **`scripts/train_models.py`**：端到端跑通一次训练与保存。
5. **`app.py`**：加「预测/模型」Tab，先只做延迟预测的筛选 + 单笔预测 + 特征重要性；再补异常检测与需求预测的 UI。
6. **文档**：更新 README。

---

## 五、文件清单（新增/修改）

| 文件 | 操作 |
|------|------|
| `requirements.txt` | 修改：增加 scikit-learn、lightgbm（可选）、joblib（若未随 sklearn 安装） |
| `src/features.py` | 新建 |
| `src/models.py` | 新建 |
| `data/models/` | 新建目录（可 .gitignore 大文件，仅提交占位或示例） |
| `scripts/train_models.py` | 新建 |
| `app.py` | 修改：新增「预测/模型」Tab 及内部逻辑 |
| `README.md` | 修改：补充 ML 功能说明与使用步骤 |
| `docs/ML-FEATURE-IMPLEMENTATION-STEPS.md` | 本文件（已创建） |

---

按上述步骤逐项落实，即可在现有 Delivery Performance Dashboard 上完整接入「预测类 (ML)」功能，并在 Streamlit 中以「预测/模型」Tab 展示预测结果与简单可解释性（如特征重要性）。
