# 膝关节康复角度波形分类系统模型架构与工作原理

## 1. 系统整体架构

```mermaid
graph TD
    A[原始波形数据] --> B[数据预处理]
    B --> C[特征增强]
    C --> D[数据标准化]
    D --> E[MG-Transformer模型]
    E --> F[概率输出]
    F --> G[角度分类结果]
    E --> H[AI智能分析引擎]
    H --> I[专业分析报告]
    
    subgraph 数据预处理阶段
        B
        C
        D
    end
    
    subgraph 模型推理阶段
        E
        F
    end
    
    subgraph 结果输出阶段
        G
        H
        I
    end
```

## 2. MG-Transformer 模型架构详解

```mermaid
graph LR
    A[输入波形序列<br/>[Batch, SeqLen, Features]] --> B[CNN特征提取层]
    B --> C[位置编码]
    C --> D[Transformer编码器]
    D --> E[多尺度池化]
    E --> F[全连接分类器]
    F --> G[角度分类概率]
    
    subgraph MG-Transformer模型
        B
        C
        D
        E
        F
    end
```

### 2.1 CNN特征提取层

```mermaid
graph TB
    A[输入特征<br/>7维增强特征] --> B[Conv1D Layer 1<br/>Kernel=5, Channels=64]
    B --> C[ReLU激活]
    C --> D[BatchNorm]
    D --> E[Conv1D Layer 2<br/>Kernel=3, Channels=128]
    E --> F[ReLU激活]
    F --> G[BatchNorm]
    G --> H[Conv1D Layer 3<br/>Kernel=3, Channels=256]
    H --> I[输出特征<br/>用于Transformer]
```

### 2.2 特征增强模块

```mermaid
graph TB
    A[原始电流信号] --> B[一阶差分<br/>变化率特征]
    A --> C[二阶差分<br/>加速度特征]
    A --> D[滑动均值<br/>趋势特征]
    A --> E[峰值检测<br/>最大值特征]
    A --> F[谷值检测<br/>最小值特征]
    A --> G[峰值位置<br/>时间特征]
    
    B --> H[7维增强特征]
    C --> H
    D --> H
    E --> H
    F --> H
    G --> H
```

### 2.3 Transformer编码器

```mermaid
graph TB
    A[位置编码输入] --> B[Multi-Head Attention<br/>8个注意力头]
    B --> C[Layer Normalization]
    C --> D[残差连接]
    D --> E[前馈神经网络<br/>1024维隐藏层]
    E --> F[Layer Normalization]
    F --> G[残差连接]
    G --> H[输出特征]
    
    subgraph Transformer Layer
        B
        C
        D
        E
        F
        G
    end
```

### 2.4 多尺度池化策略

```mermaid
graph TB
    A[Transformer输出<br/>[Batch, SeqLen, Features]] --> B[平均池化]
    A --> C[最大池化]
    A --> D[最小池化]
    B --> E[特征拼接]
    C --> E
    D --> E
    E --> F[全连接分类器输入]
```

## 3. 模型训练流程

```mermaid
graph TB
    A[训练数据准备] --> B[数据增强与预处理]
    B --> C[批次化处理]
    C --> D[前向传播]
    D --> E[损失计算<br/>交叉熵损失]
    E --> F[反向传播]
    F --> G[参数更新<br/>AdamW优化器]
    G --> H{训练完成?}
    H -- 否 --> I[下一个Epoch]
    H -- 是 --> J[模型保存]
    I --> C
```

## 4. 损失函数与优化器

### 4.1 损失函数：交叉熵损失

$$\text{Loss} = -\sum_{i=1}^{N} y_i \log(\hat{y}_i)$$

其中：
- $N$ 是类别数量
- $y_i$ 是真实标签（one-hot编码）
- $\hat{y}_i$ 是预测概率

### 4.2 优化器：AdamW

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

其中：
- $\theta_t$ 是模型参数
- $\eta$ 是学习率
- $\hat{m}_t$ 和 $\hat{v}_t$ 分别是梯度的一阶矩和二阶矩估计

## 5. 注意力机制详解

```mermaid
graph TB
    A[查询 Q] --> B[注意力权重计算]
    C[键 K] --> B
    D[值 V] --> E[加权求和]
    B --> E
    E --> F[输出]
    
    subgraph Self-Attention计算
        B
        E
    end
```

### 5.1 注意力权重计算公式

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：
- $Q$ 是查询矩阵
- $K$ 是键矩阵
- $V$ 是值矩阵
- $d_k$ 是键向量的维度

### 5.2 多头注意力机制

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

## 6. 位置编码

```mermaid
graph TB
    A[序列位置索引] --> B[正弦函数编码]
    A --> C[余弦函数编码]
    B --> D[位置编码向量]
    C --> D
```

### 6.1 位置编码公式

$$PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

其中：
- $pos$ 是位置
- $i$ 是维度
- $d_{model}$ 是模型维度

## 7. 产品功能逻辑图

```mermaid
graph TB
    A[用户界面] --> B[数据上传]
    A --> C[模型训练]
    A --> D[波形预测]
    A --> E[数据可视化]
    A --> F[AI智能分析]
    
    B --> G[数据预处理模块]
    C --> H[训练模块]
    D --> I[预测模块]
    E --> J[可视化模块]
    F --> K[AI分析模块]
    
    G --> L[特征工程]
    L --> M[标准化处理]
    M --> N[数据集构建]
    N --> H
    
    H --> O[MG-Transformer模型]
    O --> P[模型保存]
    
    I --> Q[模型加载]
    Q --> R[数据预处理]
    R --> S[模型推理]
    S --> T[概率输出]
    T --> U[结果展示]
    
    J --> V[Matplotlib图表]
    V --> W[波形可视化]
    
    K --> X[分析报告生成]
    X --> Y[专业建议输出]
    
    subgraph 后端处理模块
        G
        H
        I
        J
        K
        L
        M
        N
        O
        P
        Q
        R
        S
        T
        V
        X
    end
    
    subgraph 输出结果
        U
        W
        Y
    end
```

## 8. 科研级可视化示例

### 8.1 混淆矩阵示意图

```mermaid
graph TB
    A[真实标签] --> B[混淆矩阵]
    C[预测标签] --> B
    B --> D[对角线元素<br/>正确分类数]
    B --> E[非对角线元素<br/>误分类数]
    D --> F[准确率计算]
    E --> G[错误分析]
```

### 8.2 训练过程监控

```mermaid
graph TB
    A[训练过程] --> B[损失曲线监控]
    A --> C[准确率曲线监控]
    A --> D[梯度范数监控]
    A --> E[学习率调度]
    
    B --> F[收敛性分析]
    C --> G[过拟合检测]
    D --> H[训练稳定性]
    E --> I[优化策略调整]
```

## 9. 性能优化策略

### 9.1 数据增强技术

```mermaid
graph TB
    A[原始波形数据] --> B[噪声注入]
    A --> C[时间轴扰动]
    A --> D[幅度缩放]
    A --> E[片段裁剪]
    
    B --> F[增强数据集]
    C --> F
    D --> F
    E --> F
```

### 9.2 正则化技术

```mermaid
graph TB
    A[模型训练] --> B[Dropout层]
    A --> C[权重衰减]
    A --> D[梯度裁剪]
    A --> E[早停机制]
    
    B --> F[泛化能力提升]
    C --> F
    D --> F
    E --> F
```

以上架构图展示了膝关节康复角度波形分类系统的完整技术架构和工作原理，从数据预处理到模型推理再到结果输出，涵盖了整个产品的工作流程。