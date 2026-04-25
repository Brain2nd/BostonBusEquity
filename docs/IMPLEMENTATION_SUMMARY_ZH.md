# Boston Bus Equity 改进实现说明（中文）

## 1. 任务完成情况总览

基于你这次提出的要求，目前已经完成的内容如下：

### 已完成

1. 用你指定的远端仓库 `git@github.com:lvzhuojun/BostonBusEquity.git` 覆盖并同步了本地仓库状态。
2. 新建并切换到了你的命名分支：
   - `ZhuojunLyu/realtime-inference-baseline`
3. 按标准 Git 流程完成了本地提交：
   - commit: `38ea99d`
   - message: `Add realtime inference pipeline and delay baselines`
4. 补充了项目内的本地实时推理能力。
5. 补充了 baseline 延迟测试脚本。
6. 补充了相应的测试文件、依赖说明和 README 文档说明。
7. 在本地 `D:\Anaconda3\envs\506-final` 环境中安装了缺失依赖，并完成了本地验证。

### 未完成或按你的要求暂未执行

1. 没有推送任何远端分支。
2. 没有创建远端 PR。
3. 没有把实时服务做成前端页面。
4. 没有做长期运行的 MBTA 实时轮询服务。

### 受限项

1. 工作区内遗留了几个 `pytest-cache-files-*` 临时目录。
2. 这些目录是测试运行过程中由环境生成的权限异常缓存目录，不属于代码改动，也没有被提交。
3. 它们会导致 `git status` 出现权限 warning，但不影响代码、commit、分支和后续继续开发。

## 2. 本次改动对应你的原始需求

你提出的核心需求有两项：

1. 补实时推理功能。
2. 补 baseline 的延迟测试。

这两项都已经做了。

### 关于“实时推理功能”

已经实现了项目内可运行的本地 HTTP 推理链路，并且支持两种使用方式：

1. 直接传入请求字段做预测：
   - `route_id`
   - `stop_id`
   - `scheduled_time`
   - `scheduled_headway`
   - `direction_id`
2. 通过 MBTA V3 API 先拉取实时/近期调度信息，再将其转换为模型输入做预测。

这里我采用的是 `V2` 的 MLP 模型作为实时模型，而不是 `V3`。

原因是：

1. `V2` 的特征是因果可在线构造的。
2. `V3` 使用 wavelet 和 lag 等离线研究型特征，不适合在无历史窗口的 stateless 在线请求中直接复用。

### 关于“baseline 的延迟测试”

已经新增 baseline 评估脚本，可以对 `2025+` 时间切分测试集进行多个简单基线的误差比较。

当前新增的 baseline 包括：

1. `zero_delay`
2. `global_mean`
3. `route_mean`
4. `stop_mean`
5. `hour_mean`
6. `route_hour_mean`

## 3. 主要新增和修改的文件

### 新增：实时推理相关

1. [src/inference/__init__.py](E:\Learn\CLASS\2\506\final\f1\BostonBusEquity\src\inference\__init__.py)
2. [src/inference/build_bundle.py](E:\Learn\CLASS\2\506\final\f1\BostonBusEquity\src\inference\build_bundle.py)
3. [src/inference/bundle_utils.py](E:\Learn\CLASS\2\506\final\f1\BostonBusEquity\src\inference\bundle_utils.py)
4. [src/inference/runtime.py](E:\Learn\CLASS\2\506\final\f1\BostonBusEquity\src\inference\runtime.py)
5. [src/inference/api.py](E:\Learn\CLASS\2\506\final\f1\BostonBusEquity\src\inference\api.py)
6. [src/inference/serve.py](E:\Learn\CLASS\2\506\final\f1\BostonBusEquity\src\inference\serve.py)
7. [src/inference/mbta_realtime.py](E:\Learn\CLASS\2\506\final\f1\BostonBusEquity\src\inference\mbta_realtime.py)

### 新增：模型和 baseline 评估

1. [src/models/v2_mlp.py](E:\Learn\CLASS\2\506\final\f1\BostonBusEquity\src\models\v2_mlp.py)
2. [src/models/evaluate_delay_baselines.py](E:\Learn\CLASS\2\506\final\f1\BostonBusEquity\src\models\evaluate_delay_baselines.py)

### 修改：现有训练脚本和文档

1. [src/models/train_delay_predictor_v2.py](E:\Learn\CLASS\2\506\final\f1\BostonBusEquity\src\models\train_delay_predictor_v2.py)
2. [README.md](E:\Learn\CLASS\2\506\final\f1\BostonBusEquity\README.md)
3. [requirements.txt](E:\Learn\CLASS\2\506\final\f1\BostonBusEquity\requirements.txt)
4. [pytest.ini](E:\Learn\CLASS\2\506\final\f1\BostonBusEquity\pytest.ini)

### 新增：测试

1. [tests/test_inference_runtime.py](E:\Learn\CLASS\2\506\final\f1\BostonBusEquity\tests\test_inference_runtime.py)
2. [tests/test_inference_api.py](E:\Learn\CLASS\2\506\final\f1\BostonBusEquity\tests\test_inference_api.py)

## 4. 这次实现的设计思路

## 4.1 为什么实时推理选择 V2

`V2` 的特征来自以下几类：

1. 时间特征：小时、星期、月份的周期编码。
2. 请求自身特征：`route_id`、`stop_id`、`direction_id`、`scheduled_headway`。
3. 历史统计特征：训练期内的 route/stop/hour/route_hour 聚合统计。

这意味着模型可以在“只给当前一条计划到站记录”的情况下完成推理，不需要依赖在线服务端维护历史窗口。

而 `V3` 的 wavelet、lag、rolling 特征更适合研究型离线实验，不适合作为当前项目里的 stateless online inference 默认实现。

## 4.2 realtime bundle 的作用

新增的 bundle 机制解决了一个关键问题：

训练脚本里用到的编码映射、统计量、标准化参数，原来都只存在训练过程里，不适合直接在线加载。

所以我把在线推理所必需的内容统一打包成一个单文件 bundle，里面包括：

1. `model_state_dict`
2. `model_config`
3. `feature_columns`
4. `scaler_X`
5. `scaler_y`
6. route / stop / direction 编码映射
7. route / stop / hour / route_hour 历史统计
8. 全局均值、标准差和默认 headway 中位数

这样实时服务只需要加载一个 `.pt` 文件就能工作。

## 4.3 MBTA 实时接口的接入方式

我没有把它做成长期轮询任务，而是做成了“按请求调用 MBTA V3 API”。

当前实现方式是：

1. 调用 MBTA V3 的 `schedules` 接口，拿到 route/stop 对应的近期计划发车记录。
2. 调用 `predictions` 接口，尝试获取 MBTA 当前实时预测时间。
3. 从 schedule 中提取下一班车的 `scheduled_time`。
4. 用相邻班次估计 `scheduled_headway`。
5. 把这些字段转成项目自己的模型输入。
6. 同时在返回结果中保留 MBTA 官方当前预测延误，便于对比。

## 5. 对外接口说明

## 5.1 bundle 构建

命令：

```bash
python -m src.inference.build_bundle
```

默认输出：

```text
models/delay_predictor_mlp_v2_lag_features_temporal_realtime_bundle.pt
```

## 5.2 启动本地服务

命令：

```bash
python -m src.inference.serve --bundle models/delay_predictor_mlp_v2_lag_features_temporal_realtime_bundle.pt --host 127.0.0.1 --port 8000
```

## 5.3 API 接口

### `GET /health`

返回：

1. bundle 是否已加载
2. 模型名称
3. experiment
4. feature version

### `POST /predict`

请求字段：

1. `route_id`
2. `stop_id`
3. `scheduled_time`
4. `scheduled_headway` 可选
5. `direction_id` 可选

返回字段：

1. `predicted_delay_minutes`
2. `model`
3. `experiment`
4. `used_defaults`

### `POST /predict/mbta`

请求字段：

1. `route_id`
2. `stop_id`
3. `direction_id` 可选
4. `api_key` 可选

返回字段在本地预测结果基础上额外包含：

1. `source`
2. `schedule_id`
3. `scheduled_time`
4. `scheduled_headway`
5. `mbta_prediction_departure_time`
6. `mbta_prediction_delay_minutes`

## 6. baseline 评估说明

命令：

```bash
python -m src.models.evaluate_delay_baselines
```

输出文件：

```text
reports/delay_prediction_baselines_temporal.csv
```

baseline 评估采用与你模型实验一致的时间切分逻辑：

1. 训练集：`< 2025`
2. 测试集：`>= 2025`

这样 baseline 与模型结果是可直接对比的。

## 7. 本地验证情况

本次我实际完成了以下验证：

### 已验证

1. 新增模块可成功导入：
   - `src.inference.build_bundle`
   - `src.inference.runtime`
   - `src.inference.api`
   - `src.inference.serve`
   - `src.inference.mbta_realtime`
   - `src.models.evaluate_delay_baselines`
2. 小样本 smoke test 可以成功构建 realtime bundle。
3. `runtime` 相关测试通过。
4. `api` 相关测试通过。
5. 全部测试通过：
   - `3 passed`
6. 小样本 baseline smoke test 能正常输出 RMSE 比较结果。

### 验证环境

使用环境：

```text
D:\Anaconda3\envs\506-final
```

我在这个环境中补装了以下依赖：

1. `torch`
2. `scikit-learn`
3. `fastapi`
4. `uvicorn`
5. `pywavelets`
6. `pytest`
7. `pyarrow`

## 8. 当前还需要你知道的事

### 不是没做，而是按你要求没做

1. 没推远端。
2. 没开 PR。
3. 没合并回 `main`。

### 当前代码层面的结论

如果只按你这次明确提的开发目标来判断：

1. 实时推理功能：已完成项目内实现。
2. baseline 延迟测试：已完成。
3. 本地分支与 commit：已完成。

所以结论是：

**你之前明确让我做的核心开发项，已经做完了。**

## 9. 后续可选工作

如果你下一步继续让我推进，比较自然的后续是：

1. 补一个 benchmark 脚本，测实时接口 latency。
2. 补一个更完整的 API 调用示例文档。
3. 用真实本地全量 processed 数据重新生成 baseline 结果文件。
4. 继续处理 `pytest-cache-files-*` 的 Windows 权限异常目录。
5. 由我继续帮你整理 PR 描述，但仍然先不推送。
