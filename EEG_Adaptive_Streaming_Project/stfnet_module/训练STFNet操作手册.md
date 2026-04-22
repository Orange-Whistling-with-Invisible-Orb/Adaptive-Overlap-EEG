# 训练 STFNet 操作手册（使用 data/contaminated 训练对）

## 一、目标

本手册用于：
1. 使用你刚生成的污染训练对（`Contaminated_*.npy` + `Pure_*.npy`）训练 STFNet。
2. 只依赖你自己的 `stfnet_module` 文件夹，不再依赖外部 `STFNet` 目录。

## 二、我已帮你整理好的必要文件

当前 `stfnet_module` 下用于训练的最小必需文件是：

1. `train_stfnet.py`：训练主入口。
2. `stfnet_model.py`：STFNet 模型结构。
3. `tools.py`：评价指标与滤波、标准化工具。
4. `preprocess/SemiMultichannel.py`：数据读取与划分逻辑。
5. `preprocess/__init__.py`：预处理模块导出。
6. `checkpoints/`：用于存放你后续挑选出的最佳权重（可继续保留）。

## 三、哪些文件是可选/可删

在 `stfnet_module` 内：
1. `训练STFNet操作手册.md` 是文档，可保留。
2. 除上面必要文件外，如果你后续添加了临时脚本、旧日志、临时结果，可删。

说明：
1. 这版已经是最小训练集，不建议再删 `train_stfnet.py`、`stfnet_model.py`、`tools.py`、`preprocess/`。

## 四、训练输入数据要求（重点）

你现在污染流程生成的是：

1. `data/contaminated/Contaminated_<combo>_<sid>.npy`
2. `data/contaminated/Pure_<sid>.npy`

格式要求：
1. 两个文件 shape 必须一致。
2. 推荐 shape 为 `(S, C, T)`，通常当前是 `(1, 19, T)`。
3. 采样率默认按你污染脚本是 200 Hz。

本训练脚本会：
1. 先对每个样本做滤波与标准化。
2. 训练阶段从长序列随机裁 500 点窗口。
3. 验证/测试阶段按 500 点窗口评估指标。

## 五、快速检查（训练前必做）

在项目根目录 `EEG_Adaptive_Streaming_Project` 执行：

```powershell
python -c "import numpy as np; x=np.load('data/contaminated/Contaminated_hybrid_01_f0.npy'); y=np.load('data/contaminated/Pure_01_f0.npy'); print('x',x.shape,'y',y.shape,'same',x.shape==y.shape)"
```

## 六、开始训练

在项目根目录执行（推荐）：

```powershell
python stfnet_module/train_stfnet.py `
  --device cuda:0 `
  --EEG_path "data/contaminated/Pure_01_f0.npy" `
  --NOS_path "data/contaminated/Contaminated_hybrid_01_f0.npy" `
  --save_dir "stfnet_module/checkpoints/STFNet_01_f0_hybrid" `
  --log_dir "stfnet_module/checkpoints/json_file" `
  --epochs 50 `
  --batch_size 16 `
  --folds 10 `
  --seed 42 `
  --depth 3
```

如果没有 GPU，可改：

```powershell
python stfnet_module/train_stfnet.py --device cpu --EEG_path "data/contaminated/Pure_01_f0.npy" --NOS_path "data/contaminated/Contaminated_hybrid_01_f0.npy"
```

## 七、结果输出位置

1. 模型权重：
	`stfnet_module/checkpoints/STFNet_01_f0_hybrid/STFNet/STFNet_3/STFNet_3_50_<fold>/best.pth`
2. 每折训练日志：
	`.../result.txt`
3. fold 汇总日志：
	`stfnet_module/checkpoints/json_file/STFNet_3_50.log`

## 八、常见问题

1. 报 `No module named einops`：
	安装依赖 `pip install einops`。
2. 报 shape 不一致：
	先检查 `Contaminated_*` 与 `Pure_*` 的 shape 是否完全一致。
3. 训练很慢：
	降低 `--folds`（如先设 1）和 `--epochs`（如先设 5）做冒烟测试。

## 九、你问的重点：是不是一次只能训练一个数据集？

结论：
1. 不是只能一个。
2. 当前命令里传的是单对文件，所以看起来像“单数据集训练”。
3. 你完全可以把多个 `sid` 先合并成一个大的 `(S, C, T)` 数据集再训练。

建议规模（经验值）：
1. 最少：5 个以上 `sid`。
2. 推荐：8 到 20 个 `sid`。
3. 如果你坚持 `--folds 10`，最好 `S >= 10`，否则交叉验证意义会变弱。

## 十、长度是否太短？怎么理解当前长度

你现在最终文件通常是 `(1, 19, T_full)`：
1. `T_full` 是整段长度（取决于原始 EDF 时长），不是 500。
2. 训练脚本内部会从长序列随机裁 500 点窗口训练。
3. **每个训练样本的时间跨度**：500个样本点 / 200Hz = **2.5秒**
4. 所以“短不短”主要看 `T_full` 是否至少大于 5500（训练流程里有这个基线）。

常见情况：
1. 如果每个 EDF 是几分钟，`T_full` 通常远大于 5500，长度是够的。
2. 真正限制泛化的往往不是单条长度，而是 `sid` 数量太少。

## 十一、多数据集合并训练（命令行）

关键注意：
1. 不同 `sid` 的 `T_full` 往往不一样。
2. 直接 `np.concatenate` 会报错。
3. 必须先统一长度（推荐统一裁剪到所有样本的最小长度 `min_t`）。

### 11.1 自动收集 hybrid 对并统一长度后合并

在项目根目录执行：

```powershell
python -c "import os,glob,numpy as np; nos=sorted(glob.glob('data/contaminated/Contaminated_hybrid_*.npy')); pairs=[(n,n.replace('Contaminated_hybrid_','Pure_')) for n in nos if os.path.exists(n.replace('Contaminated_hybrid_','Pure_'))]; assert pairs,'no pairs found'; xs=[]; ys=[]; lens=[]; [lens.append(min(np.load(n).shape[-1],np.load(p).shape[-1])) for n,p in pairs]; min_t=min(lens); [xs.append(np.load(n)[..., :min_t]) or ys.append(np.load(p)[..., :min_t]) for n,p in pairs]; X=np.concatenate(xs,axis=0).astype(np.float32); Y=np.concatenate(ys,axis=0).astype(np.float32); os.makedirs('data/contaminated/merged',exist_ok=True); np.save('data/contaminated/merged/Contaminated_hybrid_merged.npy',X); np.save('data/contaminated/merged/Pure_merged.npy',Y); print('pairs',len(pairs),'min_t',min_t,'X',X.shape,'Y',Y.shape)"
```

### 11.2 用合并后的数据训练（推荐）

```powershell
python stfnet_module/train_stfnet.py `
  --device cuda:0 `
  --EEG_path "data/contaminated/merged/Pure_merged.npy" `
  --NOS_path "data/contaminated/merged/Contaminated_hybrid_merged.npy" `
  --save_dir "stfnet_module/checkpoints/STFNet_hybrid_merged" `
  --log_dir "stfnet_module/checkpoints/json_file" `
  --epochs 50 `
  --batch_size 16 `
  --folds 10 `
  --seed 42 `
  --depth 3
```

## 十二、你当前这条单数据集命令怎么理解

你这条命令本身是正确可跑的，但它只用了一个 `sid`：
1. 适合做流程验证。
2. 不适合做最终结论实验。

建议：
1. 先单数据集跑 5 epoch 验证代码和显存。
2. 再按第十一章做多数据集合并后跑正式实验。

## 十三、你要的新脚本：随机抽 10 个数据集 + 自动训练

已新增脚本：
1. `stfnet_module/train_stfnet_random10.py`

功能：
1. 从 `data/contaminated` 中按 `combo` 自动匹配 `Contaminated_<combo>_<sid>.npy` 与 `Pure_<sid>.npy`。
2. 随机抽取 10 对（默认）。
3. 自动按最短长度和通道对齐后合并。
4. 合并数据文件按时间戳命名。
5. 自动调用 `train_stfnet.py` 训练，训练输出目录也按时间戳命名。

### 13.1 一条命令直接跑

在项目根目录执行：

```powershell
python stfnet_module/train_stfnet_random10.py `
  --device cuda:0 `
  --combo hybrid `
  --epochs 50 `
  --batch_size 16 `
  --folds 10 `
  --seed 42 `
  --depth 3
```

### 13.2 参数说明（训练参数与原 STFNet 对齐）

和原训练参数保持一致：
1. `--device`
2. `--epochs`
3. `--batch_size`
4. `--folds`
5. `--seed`
6. `--depth`
7. `--save_dir`
8. `--log_dir`

新增控制参数：
1. `--combo`：`eog|emg|hybrid|mixed`，默认 `hybrid`
2. `--merge_count`：默认 `10`
3. `--contaminated_dir`：默认 `data/contaminated`

### 13.3 输出命名规则（时间戳）

每次运行会生成类似：
1. 合并数据：
  - `data/contaminated/merged/Contaminated_hybrid_merged_20260422_173015.npy`
  - `data/contaminated/merged/Pure_merged_20260422_173015.npy`
2. 训练输出目录：
  - `stfnet_module/checkpoints/run_20260422_173015/...`

## 十四、不拼接版本（你现在要的方案）

你现在的需求是：
1. 不做拼接。
2. 只从“长度足够长”的样本里挑。
3. 只在长度前 15% 的池子里随机抽 1 个数据集训练。

已新增脚本：
1. `stfnet_module/train_stfnet_top15_single.py`

### 14.1 一条命令直接跑

在项目根目录执行：

```powershell
python stfnet_module/train_stfnet_top15_single.py `
  --device cuda:0 `
  --combo hybrid `
  --top_ratio 0.15 `
  --min_len 5000 `
  --epochs 50 `
  --batch_size 16 `
  --folds 10 `
  --seed 42 `
  --depth 3
```

### 14.2 它实际做了什么

1. 扫描 `data/contaminated` 下匹配对：
  - `Contaminated_<combo>_<sid>.npy`
  - `Pure_<sid>.npy`
2. 过滤长度 `< min_len` 的样本（默认 5000）。
3. 按长度从大到小排序，取前 `top_ratio`（默认 15%）作为候选池。
4. 从候选池随机抽 1 个 `sid`。
5. 直接调用 `train_stfnet.py` 用该单数据集训练。
6. 输出目录按时间戳和 sid 自动命名：
  - `stfnet_module/checkpoints/run_YYYYMMDD_HHMMSS_<sid>/...`

### 14.3 参数说明

与原训练参数一致：
1. `--device --epochs --batch_size --folds --seed --depth --save_dir --log_dir`

新增筛选参数：
1. `--combo`：污染模式（默认 `hybrid`）
2. `--top_ratio`：长度前多少比例参与随机抽样（默认 `0.15`）
3. `--min_len`：最短长度阈值（默认 `5000`）