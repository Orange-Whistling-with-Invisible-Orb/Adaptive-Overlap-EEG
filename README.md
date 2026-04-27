# 基于自适应重叠率滑动窗口的实时脑电预处理项目

本项目实现了论文《基于自适应重叠率滑动窗口的实时脑电信号预处理方法研究》的核心代码，目标是在流式 EEG 场景下兼顾低时延与高质量去伪迹。

核心思想是将连续脑电流经过固定窗口长度切分，在不同重叠率分支上并行预处理，再通过可学习权重进行融合重构，减少传统滑窗拼接时的边缘截断伪影。

## 方法流程

1. **流式接收与基础预处理**：接收 EEG 数据包，进行带通/标准化等基础处理。  
2. **多重叠率并行切分**：固定窗口长度 `L`，以 `N` 控制重叠率，形成多窗口并行分支。  
3. **STFNet 去伪迹**：每个窗口分支使用冻结的 STFNet 做去噪。  
4. **双尺度位置评估头打分**：学习不同窗口在不同位置的贡献度。  
5. **自适应加权融合**：按时刻融合多分支输出，得到连续重构信号。  
6. **效率-精度权衡**：通过不同 `N` 的实验选择最优重叠率配置。  

## 代码结构

- `EEG_Adaptive_Streaming_Project/core_algorithm/`：自适应窗口管理、双尺度评分头、在线训练逻辑。  
- `EEG_Adaptive_Streaming_Project/stream_receiver/`：TCP 流式接收与基础预处理。  
- `EEG_Adaptive_Streaming_Project/stfnet_module/`：STFNet 模型与训练脚本。  
- `EEG_Adaptive_Streaming_Project/legacy_contamination/`：伪迹注入与数据对构建工具。  
- `EEG_Adaptive_Streaming_Project/train.py`：双尺度位置评估头训练入口。  

## 环境依赖

建议 Python 3.10+，核心依赖包括：

- `numpy`
- `scipy`
- `torch`
- `tqdm`

可按需自行安装，例如：

```bash
pip install numpy scipy torch tqdm
```

## 使用说明（示例）

> 仓库默认不包含训练/测试数据与实验结果文件，请先准备配套 EEG 数据。

### 1) STFNet 训练（直接配对数据）

```bash
python EEG_Adaptive_Streaming_Project/stfnet_module/train_stfnet_direct_pairs.py --data_dir EEG_Adaptive_Streaming_Project/data/contaminated --combo hybrid
```

### 2) 融合权重网络训练（双尺度位置评估头）

```bash
python EEG_Adaptive_Streaming_Project/train.py --data_dir EEG_Adaptive_Streaming_Project/data/contaminated --combo hybrid --overlap_n 3
```

### 3) 流式接收与在线预处理

可参考 `EEG_Adaptive_Streaming_Project/stream_receiver/tcp_receiver.py` 与 `EEG_Adaptive_Streaming_Project/core_algorithm/adaptive_window_manager.py` 进行实时推理流程集成。

## 说明

- 本仓库优先保留核心算法与工程代码。  
- 实验中间产物（如大体积数据、模型权重、结果图表、notebook）不作为代码仓库主内容提交。  
