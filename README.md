# CS610-Product-Image-Text-Consistency-Detection-System-for-E-commerce

## 0.商业问题与价值

### 商业问题

1. **图文不匹配问题**：电商平台上商家上传的产品图片与文字描述不一致，误导消费者做出错误购买决策
2. **人工审核效率低**：传统人工审核方式难以应对海量商品上新，导致审核滞后和漏检
3. **用户体验损害**：图文不匹配导致用户期望与实际收到商品差异大，增加退货率与投诉

### 商业价值

1. **提升用户信任**：自动检测并过滤图文不匹配商品，增强平台可信度，提高用户留存率
2. **降低运营成本**：将人工审核流程自动化，减少人力资源投入，加速商品上架流程
3. **减少售后纠纷**：预先拦截不匹配商品，大幅降低因图文不符导致的退货率和客服压力

## 1.数据集

### 核心字段利用策略

| 字段名称      | 基础模型用途                                                 | 改进模型用途                                                 | 创新模型用途                                                 |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `images`      | HOG特征提取/颜色直方图                                       | ResNet特征蒸馏                                               | ViT稀疏注意力输入                                            |
| `title`       | TF-IDF关键词匹配                                             | BERT文本编码                                                 | 滑动窗口注意力输入                                           |
| `features`    | 规格参数结构化解析                                           | 多模态门控融合辅助特征                                       | 属性级对齐监督信号                                           |
| `categories`  | 类目权重计算                                                 | 渐进解冻策略引导                                             | 对比学习负样本筛选                                           |
| `details`     | 商品物理属性特征工程                                         | 多模态对抗训练约束                                           | 稀疏注意力初始化参数                                         |
| `description` | 1. 长文本关键词提取  <br>2. 补充标题未提及属性  <br>3. 计算与标题的TF-IDF相似度 | 1. 作为BERT附加输入序列  <br>2. 跨模态注意力交互  <br>3. 检测语义矛盾点 | 1. 属性-视觉区域弱监督  <br>2. 图文推理链构建  <br>3. 文本蕴含关系验证 |

### 选用数据集

-  [Amazon Review Dataset 2023](https://amazon-reviews-2023.github.io/)
- 备用（目前不用）：
	- [Shopee Dataset](https://www.kaggle.com/competitions/shopee-product-matching)：本地图片
	- [Fashion200k Dataset](https://huggingface.co/datasets/Marqo/fashion200k)
	- [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset/data)

- `main_category`: 25660 rows 
	- Fashion: `raw_meta_Clothing_Shoes_and_Jewelry`: 10000 -> 5039 rows
	- Grocery: `raw_meta_Grocery_and_Gourmet_Food`: 10500 -> 5102 rows
	- Electronics: `raw_meta_Electronics`: 13000 -> 5192 rows
	- Home:  `raw_meta_Home_and_Kitchen`: 12000 -> 5127 rows
	- Beauty: `raw_meta_Beauty_and_Personal_Care`: 13000 -> 5200 rows

## 2. 数据预处理

- 拆分前预处理：
	- 数据清洗：去除空值
	- 特征选择：去除与目标变量无关的特征
	- 数据集平衡：保证每个 `main_category` 数据数量差不多（如过采样、欠采样）
	- 数据整合：处理分层采样的需求
- 拆分后预处理：
	- 特征缩放
	    - 标准化（StandardScaler）
	    - 归一化（MinMaxScaler）
	    - 对数变换等
	- 特征编码
		- 独热编码（One-Hot Encoding）
		- 标签编码（Label Encoding）
		- 频率编码等

### 原始数据

- `main_category` (str)
	- 构建同类和不同类负样本的判断依据，**不参与训练**
- `title` (str)
- `category` (list)
- `image` (json)
	- `hi_res` 的第一个 url: `image_url` -> `image_path`
- `features` (list)
- `description` (list)
- `details` (json)
- `is_match`
	- 一致（原始配对）= 1
    - 不一致（构造的负样本）= 0

### 数据清洗和数据集构建

- 去空去重复值
- 合并文本：将 `title`、`features`、`categories`、`description` 和 `details` 合并成统一文本，保留字段标识
    - 例如：`[TITLE] Ce-H22B12-S1 4Kx2K Hdmi 4Port [FEATURES] UPC: 662774021904, Weight: 0.600 lbs [DESC] HDMI In - HDMI Out...`
- 统一图片格式：调整图片大小 (224, 224) 并转化为 RGB 然后保存图片
- 构建负样本 *正负样本 1: 1*  `mismatch_type`
	- 跨类别替换：`cross_category`  20%
	- 同类别替换：`within_category`  30%
	- 颜色替换：`color_mismatch`  30%
	- 材料替换：`material_mismatch`  20%
		- **所有类别的语义分组**：
		    - 为每个类别（Fashion, Home, Electronics, Beauty, Grocery）都创建了详细的语义分组
		    - 每个分组包含类似功能或属性的材料，例如"natural_fabrics"、"metals"、"active_ingredients"等
		- **多材料一致处理**：
		    - 添加了`replacement_map`字典来跟踪已替换的材料
		    - 确保同一材料在文本中多次出现时，替换为相同的新材料，保持描述一致性
		    - 例如：如果"cotton"替换为"plastic"，则文本中所有"cotton"都会变成"plastic"
		- **跨语义组替换**：
		    - 算法会识别材料所属的语义组
		    - 选择替换材料时会优先从不同语义组中选择
		    - 确保了替换材料与原材料在功能和特性上有明显差异
		- **每个类别的特殊处理**：
		    - Fashion：区分天然面料、合成面料、动物材料等
		    - Home：区分金属、木材、石材等
		    - Electronics：区分普通塑料和高级材料
		    - Beauty：区分包装材料、活性成分和天然成分
		    - Grocery：区分包装材料、谷物和蛋白质来源
	- 每类负样本数量
		- `Beauty`: 4800
		- `Electronics`: 5357
		- `Fashion`: 4715
		- `Grocery`: 4993
		- `Home`: 5795

### 划分数据集前预处理

- **数据集合并**
    - 处理mismatch_type列
    - 检查缺失值和数据质量
- **文本预处理**
    - 标准化空白和Unicode字符
    - 结构化内容，便于模型特定处理
    - 多种格式的预处理文本(传统ML和深度学习)
- **图片预处理**
	- 调整图片大小 (224, 224) 并转化为 RGB 
	- 每类图片数量
		- Fashion: 5038 files 
		- Grocery: 5089 files 
		- Electronics: 5154 files 
		- Home: 5122 files 
		- Beauty: 5195 files
- **创建分层标记**：`strat_label` = `main_category` + '\_' + `is_match'
- **数据集分析**

**Dataset Distribution Overview:**
Total samples: 51320
Positive samples: 25660 (50.0%)
Negative samples: 25660 (50.0%)
Number of product categories: 5

**Product category distribution:**
Main_category
Home           21.3%
Electronics    20.6%
Grocery        19.7%
Beauty         19.5%
Fashion        19.0%
Name: proportion, dtype: object

**Mismatch type distribution (for negative samples):**
Mismatch_type
Color_mismatch       30.0%
Within_category      30.0%
Material_mismatch    20.0%
Cross_category       20.0%
Name: proportion, dtype: object
### 划分数据集

- 训练集/验证集/测试集 = 70% / 10% / 20%
	- Total dataset size: 51320 
	- Training set size: 35923 (70.0%) 
	- Validation set size: 5132 (10.0%) 
	- Test set size: 10265 (20.0%)
- 使用分层采样保持类别平衡，确保五个大类别在各个集合中的分布一致
	- For train set / validation set / test set
		- Home: 21.3%
		- Electronics: 20.6%
		- Grocery: 19.7%
		- Beauty: 19.5%
		- Fashion: 19.0%

### 划分数据集后预处理（模型特定预处理）

- 决策树集成模型需要TF-IDF和HOG等特征提取
- 预训练特征蒸馏网络需要BERT风格的标记化和特定的图像归一化
- 动态稀疏跨模态注意力网络需要保留结构的标记化
- CLIP有其特定的预处理流程
#### 决策树集成模型 (Decision Tree Ensemble Model)

- **文本特征提取**：
    - TF-IDF向量化文本并控制特征维度
    - 单词和双词组合(1-gram和2-gram)捕捉更多语义
- **图像特征提取**：
    - 提取HOG特征捕捉形状信息
    - 提取颜色直方图捕捉颜色分布
    - 使用并行处理加速特征提取
- **特征融合与降维**：
    - 使用PCA降低图像特征维度，避免维度灾难
    - 将文本和图像特征拼接成统一特征向量
	    - Fusion of features (simple concatenation)
	    - `xxx_combined`
- **数据保存**：
    - 保存处理后的特征和标签为NumPy数组
    - 保存TF-IDF向量器和PCA模型供后续使用
    - 保存有效样本信息便于追踪
#### 预训练特征蒸馏网络 (Pre-trained Feature Distillation Network)

- **BERT风格的文本标记化**：
    - 使用`BertTokenizer`将文本转换为模型可接受的标记序列
    - 添加适当的填充和截断，固定长度为`max_length`
    - 生成注意力掩码(attention mask)区分实际内容和填充
- **ResNet图像归一化**：
    - 调整所有图像到224×224像素（与ResNet标准输入尺寸匹配）
    - 应用ImageNet均值和标准差归一化
    - 将图像张量值范围从[0,1]调整到模型期望的标准化范围
- **PyTorch数据集和加载器**：
    - 创建专用的`DistillationDataset`类处理多模态输入
    - 自动验证和过滤无效图像路径
    - 提供高效的批处理支持，适用于蒸馏网络训练
- **信息保存和组织**：
    - 保存有效样本信息和预处理配置
    - 生成标准化的数据加载器，方便模型训练

#### 动态稀疏跨模态注意力网络 (Dynamic Sparse Cross-modal Attention Network)

- **结构化文本处理**：
    - 添加特殊标记到标记器词汇表
    - 分段处理不同文本字段(标题、类别、特征等)
    - 为每个字段生成独立的token_type_ids
    - 记录字段边界，支持局部注意力机制
- **Vision Transformer图像预处理**：
    - 使用ViT标准归一化参数(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    - 为patch-based处理准备图像
    - 维持224×224图像尺寸便于分割成固定大小的patch
- **复杂数据批处理**：
    - 自定义collate_fn处理结构化数据和字段边界
    - 支持动态稀疏注意力所需的层次结构
    - 保持字段级语义信息
- **增强的字段建模**：
    - 识别并存储各字段的开始和结束位置
    - 支持多头交叉注意力细粒度特征对齐
    - 保留字段间关系，便于自适应稀疏度设置
#### CLIP (Contrastive Language-Image Pretraining)

- CLIP预处理:
    - 使用CLIP提供的`preprocess`函数确保图像格式正确
    - 使用CLIP的`tokenize`函数处理文本
    - CLIP模型变体：`ViT-B/16`, `ViT-B/32`, `RN50`, `RN101`
	    - 使用 `ViT-B/16`
		    - **细粒度特征捕捉**：16×16像素的patch size(比32×32更小)能更好地捕捉产品细节
			- **性能和效率平衡**：ViT-B/16提供比ViT-B/32更高的精度，同时比RN101更高效，处于性能与计算资源的良好平衡点
		- **延迟标记化**:
		    - 在collate_fn中进行批量标记化，提高效率
		    - 保留原始文本用于调试和可视化

## 3. 模型选择 

### 基础模型：基于决策树集成的多模态特征融合模型

**架构设计**：
1. **特征工程**：
    - 图像特征：HOG（方向梯度直方图）+ 颜色直方图
    - 文本特征：TF-IDF + 商品类目关键词匹配度
2. **多模态融合**：  
    使用Stacking策略：
    - 第一层：3个基模型（随机森林/XGBoost/LightGBM）
    - 第二层：逻辑回归
    **优化空间**：
    - 引入卡方检验动态筛选TOP50%重要特征
    - 设计类目自适应权重：`w_category = 1 + log(类目商品数/100)`  

### 改进模型：基于预训练模型的特征蒸馏网络

**架构设计**：

1. **特征提取**：
    - 图像：蒸馏ResNet50（保留前10层）
    - 文本：蒸馏BERT的[CLS]向量（冻结前6层）
2. **跨模态交互**：  
    设计门控融合机制：
```python
    gate = σ(W_g * [v_feat; t_feat])  # σ为sigmoid
    fused_feat = gate * v_feat + (1-gate) * t_feat
```
**优化空间**：
- 采用课程中"Regularization"技术：在融合层添加DropPath正则化
- 设计渐进解冻策略：每3个epoch解冻一层预训练参数  

### 创新模型：动态稀疏跨模态注意力网络

**架构设计**：
1. **稀疏编码器**：
    - 图像：Vision Transformer with 4动态稀疏注意力头
    - 文本：滑动窗口局部注意力（窗口大小=5）
2. **对齐策略**：  
    设计商品类目引导的对比学习：
```python
    loss = -log[exp(s_pos)/（exp(s_pos) + ∑_{同品类负样本} exp(s_neg))]
```

**优化空间**：
- 引入课程中"Unsupervised Learning"技术：采用SimCLR进行预训练
- 动态调整稀疏度：从80%逐步降低到50%  

### 对比模型：CLIP（Contrastive Language-Image Pretraining）

- **双模态编码器**：
    - **图像编码器**：ViT-B/16（Vision Transformer Base，Patch Size=16）
        - 输入：224x224图像 → 拆分为16x16像素块
        - 输出：768维归一化特征向量
    - **文本编码器**：Transformer-12L（12层标准Transformer）
        - 输入：文本截断 → 词嵌入+位置编码
        - 输出：768维归一化特征向量

### 四种模型解释不匹配原因的思路

#### 1. 基于决策树集成的多模态特征融合模型

**可行性**: 很高，决策树天然具备可解释性

**思路**:

- 提取特征重要性分数，识别影响决策的关键特征
- 分析决策路径，确定是文本特征还是图像特征主导判断
- 对比不同颜色/材质特征的权重，确定不匹配类型
- 构建专门的不匹配分类器，在二分类基础上进一步分类
- 可视化关键决策节点，直观展示判断依据

#### 2. 基于预训练模型的特征蒸馏网络

**可行性**: 中等，需要额外技术辅助解释

**思路**:

- 使用Grad-CAM等技术可视化模型关注的图像区域
- 分析BERT特征中的关键词激活情况
- 设计多任务学习框架，同时预测是否匹配和不匹配类型
- 计算不同文本字段（标题、类别、特征）与图像的相似度差异
- 分析特征空间中不同类型不匹配样本的聚类模式

#### 3. 动态稀疏跨模态注意力网络

**可行性**: 很高，注意力机制本身可解释

**思路**:

- 直接可视化图像-文本注意力热图，找出不匹配区域
- 对比不同字段（如颜色描述、材质描述）的注意力分数
- 分析稀疏注意力的激活模式，高注意力区域通常指示不匹配位置
- 设计字段级交叉分析，如检测"红色"文本与蓝色区域图像的高注意力
- 根据注意力分布特征构建不匹配类型分类器

#### 4. CLIP模型

**可行性**: 中高，利用其对比学习机制

**思路**:

- 利用零样本能力测试特定属性匹配度（"这是红色产品"vs"这是蓝色产品"）
- 构建特定不匹配类型的文本模板库（如"这是一件{实际类别}产品"）
- 计算图像与不同类别文本描述的相似度，确定跨类别问题
- 使用特征归因技术定位导致低相似度的关键区域
- 分析不同属性描述（颜色、材质、尺寸）与图像的相似度差异

## 4. 评估指标

### 使用模型专属预处理

- **优点**：每个模型能发挥最佳性能
- **评估内容**：评估的是"完整解决方案"（预处理+模型），而非单纯模型架构
- **实施方法**：明确声明这是"端到端解决方案"的比较
- **适用场景**：更接近实际业务部署场景
- **主要评估**：使用每个模型的最优预处理方式进行比较，清晰说明这是"解决方案"的比较
- **在报告中明确说明**：
    - 预处理方法是模型设计的一部分
    - 每种预处理方法的选择理由
    - 预处理可能对性能产生的影响
- **强调系统设计思路**：
    - 决策树模型+TF-IDF：轻量级解决方案
    - 预训练蒸馏网络+专用tokenizer：平衡方案
    - 动态稀疏注意力网络+字段级处理：高级方案
#### 核心分类指标 

- **准确率(Accuracy)**
- **召回率(Recall)**
- **F2分数**：在电商场景中合理增加对召回率的权重

#### 业务相关指标 

- **误报率(FPR)**
- **类别平均性能**

#### 模型效率指标

- **单批次推理时间**：固定批次大小(如64或128)的平均处理时间
- **峰值内存占用**：模型运行时最大内存使用量
- **参数量**：模型可训练参数数量，反映模型复杂度
#### 可解释性与特征指标

- **注意力分布分析**：(保持不变，适用于注意力模型)
- **特征重要性**：(新增，适用于决策树模型)
    - 识别哪些文本特征对预测影响最大
    - 对比不同类别产品的关键特征差异

#### 测量建议

- 对测试样本，使用固定批次大小(如128)进行推理
- 每个模型测量3-5次并取平均值，消除随机波动
- 确保在测量前进行"预热运行"，避免首次运行初始化开销
- 使用相同硬件环境测试所有模型


示例：

| 模型           | 优势                 | 劣势         | 最适用场景             |
| -------------- | -------------------- | ------------ | ---------------------- |
| 决策树集成     | 速度快、可解释性强   | 精度略低     | 实时审核、资源受限环境 |
| 预训练特征蒸馏 | 平衡性能与资源       | 预处理较复杂 | 中等规模平台           |
| 稀疏注意力网络 | 高准确率、细粒度分析 | 资源需求高   | 高端平台、精确审核     |
| CLIP           | 零样本能力、通用性强 | 缺乏领域特化 | 新品类快速扩展         |
## 5. 实验设置（待补充）

- 构建平衡测试集：每个大类包含相等数量的匹配/不匹配样例
- 进行五折交叉验证，确保结果稳定可靠

## 6. 代码文件

- 数据预处理和探索分析 (1个文件)
	- `data_preprocessing.ipynb`
- 每个模型的训练与评估 (4个文件，每个模型一个)
	- `Decision Tree Ensemble.ipynb`
	- `Pre-trained Feature Distillation Network.ipynb`
	- `Dynamic Sparse Cross-modal Attention Network.ipynb`
	- `CLIP.ipynb`
- 模型比较与结果分析 (1个文件)
	- `Evaluation.ipynb`