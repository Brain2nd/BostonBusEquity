# 演讲用图片说明：Realtime Inference April Check-in

这些图片位于 `reports/figures/presentation/`，建议按 1 到 5 的顺序讲。核心信息是：我们不只是做离线分析，还完成了一个可运行的本地实时推理链路，并用 MBTA live API 做了官方预测 vs 本地模型预测的对比。

## 60 秒讲法

我们使用 MBTA arrival/departure 真实数据，下载了 2024、2025、2026 年数据并转换成 parquet。为了避免未来信息泄漏，realtime bundle 只用 2024 年拟合历史统计和 scaler，2025/2026 只补 route、stop、direction 的词表。模型入口选择 V2 MLP，因为它的 18 个特征都能从单条计划到站记录和训练期统计实时构造；V3 wavelet/sequence 模型虽然是离线研究模型，但不适合直接无状态实时推理。最后我们接入 MBTA V3 live prediction API，对 route 111 / stop 5547 做了 3 次轮询，得到 15 条可比较记录，模式是 `official_vs_model`，平均绝对差约 4.53 分钟。这个结果证明链路跑通并可评估，但还不能当作全局精度结论。

## Figure 1: `presentation_01_realtime_pipeline.png`

主张：我们完成了端到端实时推理链路。

怎么讲：从 MBTA 原始 CSV 到清洗后的 parquet，再到 V2 realtime bundle，最后本地模型推理并和 MBTA 官方 live prediction 对比。图中每个方框都是一个实际产物或脚本输出，不是概念图。

答辩点：这张图不是用来证明模型最准，而是证明数据收集、处理、bundle 构建、推理、live compare 这条工程链路已经跑通。

## Figure 2: `presentation_02_data_processing_summary.png`

主张：数据来源明确，清理和 split 决策接近最终，并且有防泄漏逻辑。

怎么讲：真实数据规模是 27 个 CSV、约 6.75 GB，转换后 parquet 有 6230 万行。2024 用于训练期统计，2025/2026 只用于补充类别词表。这保证实时推理时不会用未来年份的延误统计“偷看答案”。

答辩点：如果被问为什么这样切分，回答是实时系统只能使用过去统计；未来数据只能帮助系统知道新 route/stop 类别，不能参与 scaler 或历史延误统计拟合。

## Figure 3: `presentation_03_model_method_and_latency.png`

主张：至少实现并测试了一种建模方法，而且选择模型有实时性理由。

怎么讲：输入是实时请求中能拿到的字段：route、stop、scheduled time、headway、direction。它们会被转换成 18 个 V2 因果特征，送入 MLP。延迟测试显示本地 `runtime.predict` 平均 0.150 ms，p95 0.258 ms，说明推理本身不是实时系统瓶颈。

答辩点：为什么不用 V3 GRU？因为 V3 的 wavelet/sequence 特征更适合离线研究，不能自然地从单条实时记录无状态构造。我们选择 V2 是为了正确性和可上线性，而不是只追求离线分数。

## Figure 4: `presentation_04_live_official_vs_model_result.png`

主张：模型输出已经能和 MBTA 官方实时预测在同一时间轴上对比。

怎么讲：蓝线是 MBTA 官方预测延误，红线是我们的本地 V2 模型预测延误。左图展示同一次 live snapshot 里接下来几班车的延误对比；右图展示同一条 prediction 在连续轮询中的变化。底部指标显示本次样例有 3 次轮询、15/15 条可比较记录，平均绝对差 4.53 分钟。

图表选择理由：这里用折线图而不是饼图或箱线图，因为问题本身有时间顺序和班次顺序；我们关心的是预测值如何随轮询时间和即将到来的车辆序列变化。饼图会丢失时间顺序，箱线图更适合大样本分布而不是这次 live smoke test。

限制：这不是最终全局模型性能，只是一个 route-stop 的实时样例。正确表述是“实时链路和对比评估机制已经跑通”，不要说“模型已经超过 MBTA 官方预测”。

## Figure 5: `presentation_05_april_rubric_evidence_map.png`

主张：四月 check-in 评分项都有对应证据。

怎么讲：数据可视化有 live compare 和 pipeline/data/model 图；数据处理有 MBTA 数据源、清理、parquet、无泄漏 split；建模方法有 V2 MLP、18 个特征、bundle 和 latency baseline；结果解释有 `official_vs_model`、15 条可比较记录和 4.53 分钟平均差，同时说明样本限制。

## 被问到可视化决策时的回答

- 为什么用 pipeline 图：因为助教需要快速知道“我们到底做了什么”，pipeline 能把数据、模型、API、结果串起来。
- 为什么用数据 summary 卡片：因为本次重点之一是数据是否真实、规模是否足够、处理是否接近最终，卡片比长表更容易现场读。
- 为什么用 latency bar chart：因为建模方法不仅要有模型，还要有性能测试；条形图适合比较 min/avg/p95/max 延迟。
- 为什么 live compare 用折线图：因为延误预测是随时间和班次顺序变化的数值，折线能展示趋势和差距。
- 会不会误导：会，如果把 15 条 live 样例当成全局准确率。因此演讲时必须明确这是 realtime integration smoke test，不是完整泛化性能评估。

