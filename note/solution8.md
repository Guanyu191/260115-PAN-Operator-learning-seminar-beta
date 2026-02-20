## Big Question 8: 从 2D Cartesian domain 走向 arbitrary domains/unstructured grids 时，representation 与 model class 应该怎么选？

> **Evidence:** AISE25Lect10.pdf p.3-4，p.8，p.11，p.17-24，p.27-31; AISE25Lect9.pdf p.24，p.29.

**我们要回答什么：**之前的 neural operator 讨论默认隐含了一个前提，即 PDE 的定义域是 Cartesian domain，并且用 uniform grid 离散 (AISE25Lect10.pdf p.3). 但现实里很多 PDE 在 arbitrary domains 上，离散形式是 unstructured grids 或 point clouds (AISE25Lect10.pdf p.3). 因此我们要回答：当输入/输出不再是规则网格张量时，我们应该怎么选数据表示 (representation) 与模型类 (masking / DSE / GNN / GAOT/MAGNO)，以及这些路线在 accuracy / efficiency / scalability 上各自更合理的前提是什么.

**结论：**简单来说，slides 给的选型思路分三层：

1) **先选表示：**我们能不能把问题变回 Cartesian grid (masking)，或者我们必须面对 point cloud / mesh (DSE / GNN / GAOT).  
2) **再选交互结构：**我们想用 spectral 全局交互 (FNO-DSE) 还是图上的局部消息传递 (GNN / RIGNO).  
3) **最后看工程瓶颈：**纯 GNN 路线 (例如 RIGNO) 可以很准确，但 slides 明确提示它不高效，主要原因是 repeated sparse memory access (AISE25Lect9.pdf p.29; AISE25Lect10.pdf p.17). GAOT/MAGNO 用 encode-process-decode 把 point cloud 编码成 geometry-aware tokens，再用 transformer processor 做密集计算，目标是把 accuracy、efficiency、scalability 同时推上去 (AISE25Lect10.pdf p.18-24，p.27-31).

> **Note:** 这里的 "representation" 不是一个抽象概念，它直接决定了模型到底在做哪种计算：  
> - 若数据被表示成 $H\times W$ 的网格张量，那么 FFT/patching 等 dense 计算路线很自然.  
> - 若数据被表示成点集或 mesh，那么我们要么先把它变回某种 latent grid/token，要么就得接受 sparse neighbor aggregation 的代价.

**#1 核心限制是什么，为什么之前隐含了 "Cartesian domain + uniform grids".**

AISE25Lect10 开头把问题讲得很直接：之前的讨论 "only focussed on Cartesian Domains" 且 "Discretized with Uniform Grids" (AISE25Lect10.pdf p.3). 这对很多 neural operator (尤其是基于卷积或 Fourier 的家族) 是一个默认前提，因为它们的计算与参数共享方式都绑定了规则网格结构.

但现实 PDE 数据经常来自：
- **Arbitrary domains：**几何不规则，边界形状复杂.  
- **Unstructured grids / point clouds：**例如 FEM/FVM mesh 的节点与邻接关系，或者传感器/采样点云 (AISE25Lect10.pdf p.3).

<img src="../assets/solution8-fig1-任意域问题.png" style="zoom:25%;" />

所以从这里开始，"怎么把几何与邻接关系编码进模型" 就变成主问题，而不仅是 "换一个更强的 kernel".

**#2 masking / DSE / GNN 三条路线各自假设输入输出是什么表示.**

Slides 在一页里把可用方法总结为三类 (AISE25Lect10.pdf p.4)：

<img src="../assets/solution8-fig2-方法总结.png" style="zoom:25%;" />

1) **Masking (把 arbitrary domain 变回 Cartesian).**  
   - **表示假设：**我们仍然用 Cartesian uniform grid 存场值，同时用一个 mask 标注哪些网格点属于真实物理域.  
   - **对应模型类：**继续用基于规则网格的模型 (例如 FNO/CNO/ViT operator 等).  
   - **直观权衡：**简单，能复用成熟实现. 代价是几何信息被粗粒度地塞进 mask，且很多计算可能浪费在域外网格上.

2) **DSE (Direct Spectral Evaluations for FNO).**  
   - **表示假设：**输入是 point cloud / unstructured points，但希望仍然走 spectral (FNO) 路线.  
   - **对应模型类：**FNO-DSE (slides 在方法总结里点名，并在 benchmark 表里单列 FNO-DSE) (AISE25Lect10.pdf p.4，p.12-15).  
   - **(待定) 细节：**Lect10 这里没有给出 DSE 的公式细节，我们只把它当作 "把 spectral 计算从 uniform grid 推广到点云" 的思路信号，不在 note 里补充超出 slides 的实现细节.

3) **GNN (Graph Neural Networks).**  
   - **表示假设：**数据以图来表示，节点是 mesh/point cloud 的点，边是邻接关系 (或 radius graph).  
   - **对应模型类：**基于 message passing 的 GNN operator，例如 RIGNO (AISE25Lect10.pdf p.11; AISE25Lect9.pdf p.24).  
   - **直观权衡：**表示最贴合 unstructured mesh，但计算通常是 sparse neighbor aggregation，工程上可能受 sparse memory access 限制.

> **Note:** 这三条路线可以理解为三种 "把几何塞进模型" 的方式: masking 把几何塞进输入通道，DSE 把几何塞进谱计算的求积/采样，GNN 则把几何塞进图结构与邻域聚合.

**#3 message passing 的 generic form 是什么，$v_i,N_i$ 在 PDE mesh 语境下对应什么.**

Slides 给出 message passing 的 generic form (AISE25Lect10.pdf p.8)：

$$
h_i := f\left(v_i,\,\bigoplus_{j\in N_i}\Psi(v_i,v_j)\right),
$$

<img src="../assets/solution8-fig3-消息传递.png" style="zoom:25%;" />

其中 $f,\Psi$ 是 MLP，$\bigoplus$ 是聚合算子 (例如 sum / mean / max).

在 PDE mesh 的语境下，我们可以把符号对上：

- $i$ 是一个 mesh node / point.  
- $v_i$ 是节点特征，例如 $a(x_i)$ 或 $u(x_i)$ 的局部场值，以及点坐标 $(x_i,y_i,...)$ 或其他几何 embedding.  
- $N_i$ 是邻居集合，来自 mesh connectivity (FEM 邻接) 或半径图 (local neighborhood).  
- $\Psi(v_i,v_j)$ 是从邻居 $j$ 传来的 message (可以含相对位移 $x_j-x_i$ 等几何信息).  
- 聚合后的 $h_i$ 是更新后的节点表示，用于下一层传播或用于输出.

**#4 RIGNO 的目标性质与主要权衡是什么.**

Slides 在 Lect9/Lect10 都用同一套措辞描述 **RIGNO** (AISE25Lect9.pdf p.24; AISE25Lect10.pdf p.11)：

- 基于 general MPNNs.  
- 为了 operator learning 任务做了多处修改，以确保：  
  1) **Multiscale information processing.**  
  2) **Temporal continuity.**  
  3) **Resolution invariance.**

同时 slides 也给出一个非常明确的工程结论：**RIGNO is very accurate but not efficient**，训练时间超过 2 天，并把主要原因归结为 repeated sparse memory access (AISE25Lect9.pdf p.29; AISE25Lect10.pdf p.17).

<img src="../assets/solution8-fig5-稀疏瓶颈.png" style="zoom:25%;" />

> **Note:** 我们可以把这段话理解成一个很具体的提醒：当输入规模变大 (节点数上百万) 时，"用 GNN 在稀疏邻接上反复做 message passing" 可能把瓶颈推到 memory access 上，而不是 FLOPs 上. 这会迫使我们考虑把表示改造成更适合 dense kernel 的形式.

**#5 GAOT/MAGNO 的 encode-process-decode 流程是什么，它试图用 graph + transformer 同时解决什么瓶颈.**

Slides 给出的主线是：既然纯 GNN 路线的瓶颈在 sparse memory access，那么一个自然的改法是 **Use Graphs + Transformers** (AISE25Lect10.pdf p.18). 具体到 GAOT/MAGNO，slides 强调的是 **encode-process-decode strategy** (AISE25Lect10.pdf p.19).

<img src="../assets/solution8-fig4-编码流程.png" style="zoom:25%;" />

**(1) Encoder: Point cloud -> latent grid / geometry-aware tokens.**  
Slides 用一个局部聚合 + attention 加权的写法描述 encoder：对每个 latent 点 $y$，从半径邻域内的物理点 $x_k$ 聚合输入场 $a(x_k)$，并用 attention 选取 quadrature weights (AISE25Lect10.pdf p.20). 把关键结构抽出来，可以写成：

$$
w_e(y)=\sum_{|y-x_k|\le r}\alpha_k\,K(y,x_k,a(x_k))\,\phi(a(x_k)),
$$

其中 $K,\phi$ 是 MLP，$\alpha_k$ 由 attention 归一化得到. Slides 还写了 multi-scale aggregation：用多组半径 $r_m$ 聚合得到 $w_e^m(y)$，再用 softmax 权重 $\beta_m(y)$ 混合成最终 token (AISE25Lect10.pdf p.20，p.22).

几何信息的注入方式在 slides 里也被显式列出 (AISE25Lect10.pdf p.21): point coordinates，signed distance functions (SDFs)，以及一组 local statistical embeddings (邻居数，平均距离与方差，局部各向异性等). 这些信息共同服务于一个目标：让 encoder 输出的 tokens 对几何敏感，而不是把几何当作噪声.

**(2) Processor：在 latent 表示上用 transformer 做密集计算.**  
Slides 把 processor 画成一串 transformer blocks，并明确指出：  
- 若 latent grid 是 Cartesian 的，可以用 ViT 或 SWIN，从而允许 patching (AISE25Lect10.pdf p.23).  
- 若 latent grid 是 unstructured 的，则用标准 seq2seq transformer (AISE25Lect10.pdf p.23).

这一步的动机是把主要计算从 "稀疏邻域聚合" 转到 "dense token interaction"，从而更贴近现代硬件的实现优势.

**(3) Decoder: latent -> output field (可在任意 query point 评估).**  
Slides 强调 decoder "allows evaluation of output at any query point $y\in D$" 并把它称为 "Neural Field Property" (AISE25Lect10.pdf p.24). 这件事在选型上很关键：它把输出分辨率从训练时的 mesh 解绑出来，更像是在学一个连续场，而不是固定节点上的向量.

**(4) 结果与权衡: GAOT 的 accuracy / efficiency / scalability.**  
Slides 用 "GAOT combines best of both Worlds" 来概括整体结论，并明确写 "We can scale it to 3d Industrial Scale datasets" (AISE25Lect10.pdf p.28). 它还用三个 benchmark 强调输入规模与推理速度：

- DrivAerNet++: 0.5M nodes，CFD 375 node-hours vs. GAOT 0.36 seconds (AISE25Lect10.pdf p.29).  
- DrivAerML: 9M surface nodes，CFD 61K node-hours vs. GAOT 14 seconds (AISE25Lect10.pdf p.30).  
- NASA CRM aircraft: GAOT 在 surface pressure/skin friction 上与 GINO 对比给出更低的误差 (AISE25Lect10.pdf p.31).

<img src="../assets/solution8-fig6-车流基准.png" style="zoom:25%;" />

> **Note:** 这里我们不把这些数值当成 "方法论证明"，而把它当成 slides 的一个强信号：当问题进入 million-node 规模时，"先 tokenization，再 transformer" 这条路线被视为更可扩展的工程解法，而不仅是换一个更大的 GNN.
