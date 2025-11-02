# survey: Hallucination of MLLM

## 基本信息

- 题目：Hallucination of Multimodal Large Language Models: A Survey
- 作者：Zechen Bai, Pichao Wang, Tianjun Xiao, Tong He, Zongbo Han, Zheng Zhang, Mike Zheng Shou*（通讯作者）
- 单位：Show Lab, National University of Singapore；Amazon AGI, USA；AWS Shanghai AI Lab, China
- 引用：Bai, Z., Wang, P., Xiao, T., He, T., Han, Z., Zhang, Z., \& Shou, M. Z. (2024). Hallucination of multimodal large language models: A survey. arXiv preprint arXiv:2404.18930. [https://arxiv.org/abs/2404.18930]()
- 期刊及影响因子：ACM Preprint（暂未正式刊登），出版机构为 Association for Computing Machinery（ACM）；尚未确定正式期刊与影响因子（参考同类综述期刊《ACM Computing Surveys》影响因子约23.7，JCR Q1，SCI一区）

## **幻觉的定义与类型（Definition and Types of Hallucination）**

**一、定义**
多模态大语言模型（MLLM）的幻觉指生成的文本与输入的视觉内容不一致的现象，即模型在跨模态理解中出现偏差或虚构。与传统语言模型的幻觉不同，MLLM 的幻觉是“跨模态错配”（cross-modal inconsistency），而不仅仅是语言层面的事实错误。

---

**二、幻觉的三种主要类型**

1. **Category Hallucination（类别幻觉）**
    * 含义：模型识别出图像中并不存在的对象，或错误地将某一对象识别为其他类别。
    * 示例：图中没有“长椅”或“围栏”，但描述中却出现这些词。
    * 原因：视觉特征模糊、数据偏差（某些物体出现频率高）、语言模型倾向补全场景。
2. **Attribute Hallucination（属性幻觉）**
    * 含义：模型识别到的对象是正确的，但其描述的属性错误，包括颜色、形状、数量、材质或状态等。
    * 示例：图像中的花是白色的，模型却描述为“粉色花朵”。
    * 原因：视觉细节丢失、分辨率不足、语言模型根据常识补全属性。
3. **Relation Hallucination（关系幻觉）**
    * 含义：对象及其属性正确，但对象之间的关系描述错误，如空间位置、互动方式、动作方向等。
    * 示例：模型描述“其他人围着她站着”，但图像中并不存在这种互动。
    * 原因：空间推理能力弱、缺乏交互关系建模、语言模板影响生成。

---

**三、分类扩展说明**

* 对象计数（object counting）被视为属性幻觉的一种，因为数量属于对象属性。
* 动作动词幻觉（verb hallucination）涉及对象交互，属性或关系，在这里不考虑。
* 该三分法旨在保持分类体系的统一与可评测性，避免过度细分带来的复杂度。

---

**四、不同幻觉类型的风险与特征对比**


| 幻觉类型 | 错误性质 | 典型影响场景 | 主要风险 | 检测难度 |
| :-- | :-- | :-- | :-- | :-- |
| Category | 错认或虚构对象 | 自动驾驶、安防 | 高（安全风险） | 中 |
| Attribute | 属性错误 | 医疗诊断、科研 | 中（误导性高） | 中 |
| Relation | 对象关系错误 | 视觉问答、描述生成 | 中（语义失真） | 高 |


---

## **幻觉成因（Hallucination Causes）**

### **一、总体概念（General Concept）**

幻觉（*Hallucination*）的产生是多因素共同作用的结果，贯穿模型的整个生命周期。作者将其分为四个层面：**数据（Data）**、**模型（Model）**、**训练（Training）**、**推理（Inference）**。
这四类成因构成从输入到输出的完整因果链：
**Data Bias → Model Bias → Training Amplification → Inference Distortion**。

---

### **二、数据层幻觉（Hallucination from Data）**

幻觉的根源往往在于数据层（*Data-level causes*）。

1. **数据量不足（Data Quantity Insufficiency）**
    * 跨模态配对样本（*multimodal paired samples*）稀缺，模型难以学习稳定的跨模态映射（*cross-modal mapping*）。
    * 特别在领域数据（*domain-specific data*）中，幻觉更为明显。
2. **数据质量低（Data Quality Noise）**
    * 图文不对齐（*misaligned image-text pairs*）、标签错误（*label noise*）或自动生成文本描述误差（*caption errors*）。
    * 模型学到错误的语义对应（*false correspondence*）。
3. **统计偏差（Statistical Bias）**
    * 高频共现（*co-occurrence bias*）导致模型根据语言共现而非视觉内容生成。
    * 例：训练集中“dog + grass”频繁共现，模型看到草地就自动生成“dog”。
4. **描述粒度不均衡（Description Granularity Imbalance）**
    * 样本中细节描述（*detail level*）不统一，模型生成风格不稳定。

数据问题决定了模型学习的真实性边界（*fidelity boundary*）。

---

### **三、模型层幻觉（Hallucination from Model）**

模型架构与模态融合方式决定幻觉的结构性偏差（*structural bias*）。

1. **视觉信息丢失（Visual Information Loss）**
    * 视觉编码器（*vision encoder*, e.g., CLIP, ViT）在下采样（*downsampling*）中丢失局部特征（*local features*），小物体难以识别。
2. **特征偏差（Feature Bias / Modality Dominance）**
    * 在特征融合（*feature fusion*）中，语言模态（*language modality*）占主导，视觉信号（*visual signal*）被弱化。
3. **语言知识主导（Parametric Knowledge Dominance）**
    * 语言模型（*Language Model, LM*）内部知识（*parametric world knowledge*）覆盖视觉证据（*visual evidence*），生成时依赖常识补全。
4. **跨模态接口对齐不足（Inferior Cross-modal Interface Alignment）**
    * 图像与语言特征的投影层（*projection layer*）未良好对齐（*alignment*），语义漂移（*semantic drift*）严重。

模型结构决定幻觉的偏向来源，是视觉—语言权衡（*vision-language balance*）的关键环节。

---

### **四、训练层幻觉（Hallucination from Training）**

训练过程（*training procedure*）可能放大幻觉倾向。

1. **序列级监督缺失（Lack of Sequence-level Supervision）**
    * 仅基于逐词预测（*token-level loss*）优化，缺乏整体语义一致性（*semantic consistency*）约束。
2. **视觉监督不足（Weak Visual Supervision）**
    * 缺乏针对视觉特征（*visual features*）的显式损失项（*explicit loss function*），语言部分被过度优化。
3. **人类反馈稀缺（Lack of Human Feedback for Vision）**
    * 强化学习（*Reinforcement Learning from Human Feedback, RLHF*）主要用于文本任务，缺少针对视觉一致性（*visual alignment*）的反馈信号。

训练目标设计不当会强化语言偏向（*language dominance*），使幻觉倾向持续放大。

---

### **五、推理层幻觉（Hallucination from Inference）**

推理阶段（*inference stage*）的注意力与采样机制（*attention and decoding mechanisms*）会触发幻觉。

1. **视觉注意力衰减（Visual Attention Decay）**
    * 随着序列生成（*sequence generation*）的推进，模型注意力逐渐偏向文本历史（*textual context*），忽视视觉输入。
2. **视觉 Token 误导（Trap Visual Tokens）**
    * 某些高权重视觉 token（*visual tokens with high attention weights*）主导生成，出现“以偏概全”的描述（*overgeneralization*）。
3. **解码策略偏差（Decoding Bias）**
    * 高温采样（*high-temperature sampling*）或宽 top-k 策略（*wide top-k sampling*）会增加随机生成（*stochastic hallucination*）的概率。

推理阶段的问题是幻觉的直接触发器（*immediate trigger*）。

---

### **六、启示（Summary and Implications）**

* 幻觉是系统性问题（*systemic issue*），不是单一环节错误。
* 四层成因之间存在因果传递（*causal propagation*），早期偏差会在后续被放大。
* 有效缓解需多层协同（*multi-level mitigation*）：数据增强（*data augmentation*）、架构优化（*architecture refinement*）、训练策略（*training objective design*）与推理控制（*inference control*）并行。
* “视觉-语言对齐”（*vision-language alignment*）是幻觉问题的核心矛盾与突破关键。


## **幻觉评估（Metrics \& Benchmarks）**


---

### 一、总体概念（General Concept）

幻觉评估（*Hallucination Evaluation*）旨在衡量多模态大语言模型（*Multimodal Large Language Models, MLLMs*）生成结果与视觉输入之间的一致性（*cross-modal consistency*）。
由于幻觉类型多样，作者将现有研究的评估方法划分为两大类：

1. **评测指标（Metrics）**：用于定量衡量幻觉程度的计算方法。
2. **基准测试（Benchmarks）**：包含标注样本与任务的评测数据集，用于比较不同模型的幻觉表现。

论文系统梳理了从早期的图像描述（*image captioning*）指标到近期针对 MLLM 幻觉的新型评测体系，构建了完整的评价框架。

---

### 二、评测指标（Metrics）

评测指标的核心目标是**度量生成文本与视觉内容的语义一致性（*semantic consistency*）与忠实性（*faithfulness*）**。
这些指标可分为**基于语言的（text-based）**、**基于视觉的（vision-based**和**基于模型对齐的（alignment-based**三类。

#### 1. CHAIR (Caption Hallucination Assessment with Image Relevance)

* **来源**：Rohrbach et al., 2018
* **类型**：基于图像描述任务（*image captioning*）的早期幻觉指标。
* **原理**：检测生成描述中的对象（*object mentions*）是否出现在图像的检测结果中。
* **度量**：
    * *CHAIRs*: 幻觉对象占描述对象的比例；
    * *CHAIRi*: 含有幻觉对象的描述占总描述比例。
* **局限性**：依赖目标检测器（*object detector*）的性能，对属性或关系类幻觉不敏感。


#### 2. POPE (Perceptual Object-Predicate Evaluation)

* **来源**：Li et al., 2023
* **任务类型**：基于问答（*VQA-style yes/no questions*）。
* **设计思路**：通过问模型“图片中是否有X”来检测对象幻觉。
* **优势**：操作简单、适用广泛，可评估生成式或判别式模型。
* **局限**：仅覆盖对象存在性幻觉（*category hallucination*），语义粒度有限。


#### 3. AMBER (Automatic Multimodal Benchmark for Evaluation of Reliability)

* **来源**：Qi et al., 2024
* **特征**：评估对象（*object*）、属性（*attribute*）和关系（*relation*）三类幻觉。
* **机制**：综合模型生成、视觉检测与语言匹配模块实现自动化评估。
* **优点**：覆盖面广，自动化程度高，能够综合反映模型多维度幻觉表现。


#### 4. FaithScore / HalluScore

* **核心思想**：基于语义相似度（*semantic similarity*）与事实一致性（*factual consistency*）计算模型输出的忠实度。
* **典型实现**：采用 CLIP 或大型语言模型（如 GPT-4）作为评估器，对文本描述与图像的语义对齐程度打分。
* **优势**：不依赖人工标注，可自动化评估。
* **问题**：评估结果受评估器性能影响较大。


#### 5. M-HalDetect / HaELM

* **类型**：基于判别器（*discriminator-based*）或探测器（*hallucination detector*）的学习型评测方法。
* **原理**：训练分类器预测生成文本是否出现幻觉。
* **优点**：灵活、可针对特定任务优化。
* **局限**：需要额外标注或训练，通用性较差。

---

### 三、评测基准（Benchmarks）

评测基准是评估幻觉表现的重要数据基础，通常包含人工标注的图像-文本对或问答样本。论文总结了多种典型数据集：


| 名称 | 任务类型 | 特点 | 评估维度 |
| :-- | :-- | :-- | :-- |
| **CHAIR** | 图像描述 (Image Captioning) | 最早的幻觉检测基准 | 对象幻觉（Category） |
| **POPE** | 视觉问答 (VQA Yes/No) | 检测对象存在性幻觉 | Category |
| **AMBER** | 综合基准 (Comprehensive Benchmark) | 自动化检测对象、属性、关系 | Category / Attribute / Relation |
| **MMHal-Bench** | 多模态综合评估 (Multimodal Benchmark) | 涵盖文本生成、问答、对话 | 全维幻觉评测 |
| **FaithScore Dataset** | 图文对齐数据集 | 评估视觉与文本的忠实性 | Cross-modal Faithfulness |
| **HaELM Benchmark** | 视觉问答扩展集 | 标注模型错误类型 | Error Type Classification |

这些基准共同构成了 MLLM 幻觉研究的标准测试框架，为不同模型提供可复现的比较依据。

---

### 四、评测方法的比较与局限性

1. **主观评估（Human Evaluation）**
    * 最可靠但成本高、可扩展性差。
    * 常用于验证自动化指标的有效性。
2. **自动评估（Automatic Evaluation）**
    * 依赖模型或嵌入空间（如 CLIP）度量语义一致性。
    * 适合大规模比较，但存在模型偏差问题。
3. **混合评估（Hybrid Evaluation）**
    * 结合人类评估与自动评测，提高客观性与效率。

主要挑战：

* 幻觉定义仍不统一，指标难以全面覆盖。
* 自动评估依赖模型性能，易产生评估偏差。
* 多模态任务复杂，标准化尚未形成。

---

## 幻觉缓解（Hallucination Mitigation）


---

### 一、核心概念（Core Idea）

幻觉（*hallucination*）在多模态大语言模型（*Multimodal Large Language Models, MLLMs*）中普遍存在。
作者指出：

> *“Mitigation is not elimination, but continuous control.”*
> （缓解不是消除，而是持续控制。）

因此，缓解的目标是让模型在事实约束内生成内容，而非凭语言先验“想象”。

作者将缓解策略按产生原因分为四个层面：
**数据（Data）→ 模型（Model）→ 训练（Training）→ 推理（Inference）**。
每一层面对应不同的干预方式。

---

### 二、数据层缓解（Data-level Mitigation）

**1. 思想**

幻觉的源头往往在数据。
数据层缓解旨在清洗噪声、增强对齐、平衡分布，让模型“学真话”。

---

**2. 关键方法与案例**


| 方法 | 机制 | 案例与说明 |
| :-- | :-- | :-- |
| **数据清洗与过滤（Data Cleaning \& Filtering）** | 利用 CLIP 或语言模型过滤图文不对齐样本。 | **ReCaption (2024)**：用 BLIP2 重新生成图像描述（re-captioning），再按对齐得分筛样本。噪声比例降至原先的三分之一。 |
| **反事实数据生成（Counterfactual Data Generation）** | 构造“有幻觉 vs 无幻觉”样本，训练模型识别错误。 | **HalluciDoctor (2024)**：自动替换对象/属性（如 “dog” → “cat”），让模型学习区分真假图文关系。幻觉检测准确率 +10%。 |
| **分布平衡（Distribution Debiasing）** | 调整训练集中对象共现频率，弱化语言先验。 | **LRV-Instruction (2023)**：增加稀有对象样本，减少高频“共现错觉”。 |


---

**3. 小结**

* 核心目标：提升图文对齐质量（*image-text alignment quality*）
* 优点：简单有效，直接作用于模型输入。
* 局限：成本高、需依赖高质量检测模型。

**一句话总结：**
干净的数据，是让模型“看对图”的第一步。

---

### 三、模型层缓解（Model-level Mitigation）

**1. 思想**

幻觉往往源于模型架构的模态不平衡：语言通道主导、视觉通道弱化。
因此，改进模型结构与对齐机制是缓解的关键。

---

**2. 关键方法与案例**


| 方法 | 机制 | 案例与说明 |
| :-- | :-- | :-- |
| **增强视觉编码（Enhanced Visual Encoding）** | 提高分辨率、保留更多视觉 token。 | **LLaVA-1.5 (2023)**：使用 ViT-Large 编码器，视觉 token 数量翻倍。POPE 幻觉率从 23% 降至 14%。 |
| **动态模态平衡（Dynamic Modality Gating）** | 根据生成阶段自动调整视觉/语言权重。 | **HallE-Switch (2024)**：门控模块（*gating module*）在生成前半段强化视觉注意力，后半段强化语言连贯性，减少对象遗漏。 |
| **跨模态对齐（Cross-modal Alignment）** | 引入投影共享或对齐层以减少语义漂移。 | **Flamingo / Qwen-VL**：融合注意力（*cross-attention fusion*）提升视觉-语言一致性。 |


---

**3. 小结**

* 核心目标：防止语言特征覆盖视觉特征。
* 优点：直接影响模型感知能力。
* 局限：结构复杂、计算成本高。

**一句话总结：**
让模型真正“看图说话”，而不是“凭经验猜图”。

---

### 四、训练层缓解（Training-level Mitigation）

**1. 思想**

训练目标（*objective function*）决定模型行为。
通过设计新的监督信号或优化目标，可以在学习阶段直接约束幻觉。

---

**2. 关键方法与案例**


| 方法 | 机制 | 案例与说明 |
| :-- | :-- | :-- |
| **对比学习（Contrastive Learning）** | 区分匹配与不匹配样本，强化语义边界。 | **MOCHa (2024)**：正负样本对比训练，让模型“意识到什么是错误”。幻觉率下降约 15%。 |
| **视觉一致性监督（Visual Consistency Loss）** | 在损失函数中加入跨模态一致性项。 | **Chen et al. (2024)**：在语言损失外加视觉约束，减少视觉漂移。 |
| **视觉强化学习（Visual Reinforcement Learning, RLHF-V）** | 使用视觉反馈作为奖励信号优化生成。 | **HA-DPO (2024)**：利用 CLIP 或 GPT-4V 打分“图文一致性”，奖励忠实生成。 |


---

**3. 小结**

* 核心目标：让模型在学习时“学会诚实”。
* 优点：直接作用于参数更新，通用性强。
* 局限：训练成本高、需额外反馈模型。

**一句话总结：**
训练不是“记住句子”，而是“学习守真”。

---

### 五、推理层缓解（Inference-level Mitigation）

**1. 思想**

幻觉往往在生成时被“放大”。
推理层缓解通过**控制生成过程**来抑制错误输出。

---

**2. 关键方法与案例**


| 方法 | 机制 | 案例与说明 |
| :-- | :-- | :-- |
| **对比解码（Contrastive Decoding）** | 比较“保守型”与“创造型”生成结果，拒绝虚构内容。 | **OPERA (2024)**：利用双模型输出对比，高温输出仅在视觉证据支持下保留，幻觉率降 30%。 |
| **视觉引导生成（Visual-guided Decoding）** | 在生成中动态强化视觉注意力。 | **Woodpecker (2023)**：根据显著性图（*visual saliency map*）调整注意力焦点，防止模型忽略关键物体。 |
| **后验修正（Post-hoc Correction）** | 生成后检测并修正幻觉句子。 | **Volcano (2024)**：用 GPT-4 审查生成输出，检测并改写幻觉段落。 |


---

**3. 小结**

* 核心目标：在生成阶段“看住模型的嘴”。
* 优点：可作为插件应用于已有模型。
* 局限：推理延迟、计算成本高。

**一句话总结：**
让模型“说得多”可以，但要“说得准”。

---

### 六、多层协同与未来趋势（Multi-level Synergy \& Future Directions）

**1. 协同机制**

多层方法往往协同效果更好：


| 组合策略 | 示例 | 优点 |
| :-- | :-- | :-- |
| 数据 + 训练 | ReCaption + MOCHa | 兼顾质量与目标约束 |
| 模型 + 推理 | HallE-Switch + OPERA | 架构优化 + 解码控制 |
| 端到端闭环 | RLHF-V + 后验修正 | 形成“检测→修正→再训练”的闭环 |


---

**2. 未来方向**

1. **视觉主导架构（Vision-dominant Architecture）**
强化视觉分支的表达与控制能力。
2. **自校正反馈系统（Self-corrective Feedback Loop）**
将幻觉检测模块融入生成循环，实现动态自监督。
3. **开放评测标准（Open Evaluation Protocol）**
统一幻觉定义、建立持续更新的 benchmark。
