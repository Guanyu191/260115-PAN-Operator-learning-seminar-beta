## Big Question 2: 什么叫 "genuine operator learning"，以及它和跨分辨率泛化的关系是什么？

> **Evidence:** AISE25Lect5.pdf p.26-31，p.40; AISE25Lect6.pdf p.4-6，p.20-25; AISE25Lect7.pdf p.2-8.

**我们要回答什么：**Slides 在 "Desiderata for Operator Learning" 里写 "Input + Output are functions" 与 "Learn underlying Operator, not just a discrete Representation" (AISE25Lect5.pdf p.31). 同时又把 **discretize-then-learn** 的流程标注为 **"Not possible to evaluate on a Different Resolution" 与 "Not genuine operator learning"** (AISE25Lect5.pdf p.28; AISE25Lect7.pdf p.2). 我们想把这两页放在一起读：什么叫 **"genuine operator learning"**，为什么它会自然牵扯到跨网格/跨分辨率的一致性，以及 slides 用 CDE/ReNO 给出的判定标准是什么.

**结论：**Slides 这里说 **"genuine operator learning"**，简单来说就是两点：**输入与输出是函数**，而且我们想学的是 **underlying operator**，不是某个固定网格上的离散表示 (AISE25Lect5.pdf p.31). 因为我们实际只能通过编码 $\mathcal{E}$ (discretization) 与重建 $\mathcal{R}$ (reconstruction) 来接触这些函数，所以跨分辨率 (换网格) 会变成一个很自然的测试：如果模型学到的是 $L:\mathbb{R}^N\to\mathbb{R}^N$ 这种固定维度的离散映射，那么换 $N$ 时要么直接无法评估，要么出现明显 discrepancy (AISE25Lect5.pdf p.28; AISE25Lect7.pdf p.2). Slides 用 **continuous-discrete equivalence (CDE)** 与 **Representation Equivalent ReNO** 把 "是否真的学到了算子" 写成一个可判定的条件：**在给定 $\mathcal{E}/\mathcal{R}$ 下，离散计算链与连续算子是否 representation equivalent (aliasing error 是否为 0)，并且这个条件要逐层检查** (AISE25Lect6.pdf p.20-21; AISE25Lect7.pdf p.3-4).

> **Note:** 如果一个模型的核心模块只能吃长度为 $N$ 的向量输入 (也就是某个固定网格上的采样值)，它很容易就学成了 "这个 $N$ 维向量到那个 $N$ 维向量" 的映射. 而 "genuine operator learning" 更希望网格只是表示方式，目标对象是函数到函数的 **operator**，所以换一种网格不应让模型的定义或主要结论失效.

> **Definition:** (Aliasing error，AISE25Lect6.pdf p.20-21 / AISE25Lect7.pdf p.3-4) 令 $\mathcal{E}$ 为编码 (discretization)，$\mathcal{R}$ 为重建 (reconstruction)，slides 写
> $$
> \epsilon(\mathcal{G}, G) = \mathcal{G} - \mathcal{R} \circ G \circ \mathcal{E}.
> $$
> 若 $\epsilon \equiv 0$，则称离散表示与连续算子在该表示下是一致的 (representation equivalent).

<img src="../assets/solution2-fig2-ReNO定义.png" style="zoom:25%;" />

**#1 为什么 discretize-then-learn 往往不是 genuine operator learning.**Slides 给出的 desiderata 是：输入与输出是函数，并且要 "learn underlying operator, not just a discrete representation" (AISE25Lect5.pdf p.31). 但在 discretize-then-learn 的链路里，核心学习对象是固定维度的离散映射 $L^{*}:\mathbb{R}^{N}\to\mathbb{R}^{N}$，因此它天然绑定某个分辨率 $N$，这也是 slides 用 "Not possible to evaluate on a Different Resolution" -> "Not genuine operator learning" 来总结它的原因 (AISE25Lect5.pdf p.28; AISE25Lect6.pdf p.6).

Slides 把 "discretize -> learn -> reconstruct" 写成

$$
\mathcal{G}^{*} = \mathcal{R} \circ L^{*} \circ \mathcal{E}
$$

(AISE25Lect5.pdf p.26; AISE25Lect6.pdf p.4). 其中 $L^{*}$ 的输入输出维度固定在 $\mathbb{R}^{N}$，当网格从 $N$ 变到 $N'$ 时，模型结构就会直接与分辨率绑定.

<img src="../assets/solution2-fig1-离散再学习.png" style="zoom:25%;" />

**#2 表示变化的两个组件 $\mathcal{E}$ 与 $\mathcal{R}$.**Slides 给出一个 1D regular grid 的具体例子来说明 $\mathcal{E}/\mathcal{R}$ 的含义 (AISE25Lect6.pdf p.22). 其中 encoding $\mathcal{E}(u)$ 是 pointwise evaluation，而 reconstruction $\mathcal{R}(v)$ 用 **sinc basis** 做重建；**Nyquist-Shannon** 在这里提供的是 **"bandlimited + sufficiently dense grid"** 下的 bijection (一一对应). 换句话说，在这些假设下，采样值 $\mathcal{E}(u)$ 可以唯一确定 $u$，并且可以用 $\mathcal{R}$ 从 $\mathcal{E}(u)$ 重建回 $u$.

<img src="../assets/solution2-fig3-sinc重建.png" style="zoom:25%;" />

> **Note:** (**bandlimited** function) Slides 在这个例子里把输入和输出的函数空间 $X,Y$ 限制为 bandlimited functions，并用 "supp $\hat u \subset [-\Omega,\Omega]$" 来刻画 (AISE25Lect6.pdf p.22). 简单来说，这等价于 **"函数的细节不含超过某个最高频率"**. 在这个假设下，**如果网格足够密，那么 pointwise sampling 不会丢信息**，sinc reconstruction 可以把离散样本无损重建回原函数 (这也是 slides 写 "Nyquist-Shannon -> **bijection (一一对应)**" 的原因). 反过来，如果不满足 bandlimited 或网格不够密，不同的连续函数可能在网格点上取到相同数值，从而出现 aliasing.

$\mathcal{E}$ 的作用是采样：把连续函数 $u(\cdot)$ 变成离散样本 $\mathcal{E}(u)=\{u(x_j)\}_{j=1}^n$. $\mathcal{R}$ 的作用是重建：把离散样本 $v=\{v_j\}$ 变回连续函数，例如用 sinc 基:

$$
\mathcal{R}(v)(x) = \sum_{j=1}^n v_j \, \mathrm{sinc}(x-x_j).
$$

> **Note:** (**sinc basis**) 这里的 $\mathrm{sinc}(x-x_j)$ 表示以采样点 $x_j$ 为中心的一族基函数. 常见约定有 $\mathrm{sinc}(t)=\sin(\pi t)/(\pi t)$ (normalized) 或 $\mathrm{sinc}(t)=\sin(t)/t$ (unnormalized)，并取 $\mathrm{sinc}(0)=1$. 这一页 slides 未显式写出归一化常数，但无论采用哪种约定，它都对应 **bandlimited** 情况下的理想插值核：$\mathcal{R}$ 把离散样本 $\{v_j\}$ 通过加权求和组合成连续函数，从而和后面的 Nyquist-Shannon 双射结论对齐.

**Nyquist-Shannon 的核心是：如果 $u$ 是 bandlimited 且网格足够密，那么采样与 sinc 重建在理论上可以做到一一对应 (双射)**. 这给了我们一个便于思考的理想化场景：同一函数在不同表示之间可以无损转换. 在这个场景下，我们再去讨论 CDE / ReNO 会更清楚.

**#3 aliasing error $\epsilon$ 在测什么.**$\epsilon$ 衡量 **"连续算子 $\mathcal{G}$" 与 "离散计算链 $\mathcal{R} \circ G \circ \mathcal{E}$" 之间的差异** (AISE25Lect6.pdf p.20-21; AISE25Lect7.pdf p.3-4). 若 $\epsilon \equiv 0$，则表示该计算链与 $\mathcal{G}$ 在该编码/重建下完全一致；slides 在下一页直接把它与 "discrepancies between resolutions" 联系起来 (AISE25Lect6.pdf p.21; AISE25Lect7.pdf p.4).

$\mathcal{R} \circ G \circ \mathcal{E}$ 表示 "先把输入函数离散化，再在离散空间里做运算，最后插值回函数" 的整条离散计算链. 如果把它看作对真实算子 $\mathcal{G}$ 的近似，那么 $\epsilon$ 就是在连续层面衡量 "这条离散链到底偏离了 $\mathcal{G}$ 多少". $\epsilon \equiv 0$ 是一个更严格的一致性要求：在该编码/重建下，离散实现与连续算子一致，因此按这个要求来看，更换表示 (例如网格分辨率变化) 不应额外引入误差来源.

**#4 ReNO 的判定为什么要 layerwise.**Slides 强调这个概念需要逐层 (layerwise) 满足 (AISE25Lect6.pdf p.20; AISE25Lect7.pdf p.3). 对应到 $\mathcal{G} = \mathcal{G}_L \circ \cdots \circ \mathcal{G}_1$ 的分解，直观动机是：端到端误差小并不意味着中间层与表示变换相容；跨网格时新增的 aliasing/discrepancy 往往来自**某一层的非等价实现**.

只看端到端误差时，内部的表示问题可能不太容易看出来：模型可能在某个固定分辨率下把误差做得很小，但中间层的算子并不与表示变换相容，一换网格就会出现新的 aliasing 误差. Layerwise 的好处是把 "是否与表示变化可交换" 这件事落实到具体模块上：哪一层破坏了 bandlimit，哪一层引入了分辨率相关的离散卷积. 在 ReNO 视角下，若每一层都满足 representation equivalence，那么组合后整体更容易保留这种性质，跨分辨率的不一致也更可控.

**#5 CNN / FNO 的关键限制各是什么.**Slides 的对比结论是: CNN 不是 ReNO (AISE25Lect6.pdf p.23; AISE25Lect7.pdf p.6)，FNO 的 kernel 部分在周期 bandlimited 假设下可能是 ReNO，但 **activation 会破坏 bandlimit**，因此 "not necessarily ReNO" (AISE25Lect6.pdf p.24-25; AISE25Lect7.pdf p.7-8).

CNN 的离散卷积基于固定网格索引与离散 kernel，因此分辨率一变就可能出现不一致 (AISE25Lect6.pdf p.23).

<img src="../assets/solution2-fig4-CNN非ReNO.png" style="zoom:25%;" />

FNO 的 Fourier 卷积在周期 bandlimited 假设下 kernel 部分可保持表示一致性，但 pointwise 非线性 $\sigma$ 会引入高频并打破 bandlimit，从而不一定是 ReNO (AISE25Lect6.pdf p.24-25).

<img src="../assets/solution2-fig5-FNO非必ReNO.png" style="zoom:25%;" />
