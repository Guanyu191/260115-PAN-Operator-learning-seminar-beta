## Big Question 7: time-dependent PDE 的 operator learning 目标应如何定义 (连续时间评估)，训练策略如何利用 semi-group 结构？

> **Evidence:** AISE25Lect9.pdf p.4-9，p.12; AISE25Lect12.pdf p.15.

**我们要回答什么：**time-dependent PDE 时，我们不是学一个静态的 $\mathcal{G}:X\to Y$，而是学一个随时间演化的 solution operator. Slides 的核心要求是：  
1) 把目标写成一个对任意 $t\in(0,T]$ 都能评估的算子 $\mathcal{S}(t,\bar u)$.  
2) 把训练策略与推理/评估对齐，尤其要搞清楚 "只在离散时间网格上做 rollout" 和 "连续时间评估 (含 OOD 时间点)" 的差别.

**结论：**Slides 给的框架很清楚：

- **定义：**对抽象 PDE $u_t+L(t,x,u)=0$，给定初值 $u(0)=\bar u$，solution operator 写成
  $$
  \mathcal{S}:(0,T)\times X\to X,\qquad \mathcal{S}(t,\bar u)=u(t).
  $$
- **半群性质：**对任意时间增量 $\Delta t$，
  $$
  \mathcal{S}(\Delta t,u(t))=u(t+\Delta t).
  $$
  这给了我们两件事：  
  (1) 轨迹数据的组织方式 (把一条 trajectory 拆成很多 "从当前态到未来态" 的子问题).  
  (2) all2all training 的样本构造 (用任意 $i<j$ 的 pair 来训练).
- **训练/推理三条路：**
  1) autoregressive rollout：定义一个固定步长的 next-step operator $NO_{\Delta t}$，反复迭代得到长时间预测，但会有误差累积且只在离散时间点 (time levels) 上评估.  
  2) time conditioning：把 lead time $\bar t$ 当作输入，直接学 $\mathcal{S}(\bar t,u(t))$，并用 conditional normalization 让网络参数显式依赖 $t$.  
  3) all2all training：用 $(u(t_i),u(t_j))$ 的所有 pair 训练 $\mathcal{S}(t_j-t_i,u(t_i))$，从而支持 "任意 $t>0$" 的评估，包含 OOD 时间点.

<img src="../assets/solution7-fig1-时变算子.png" style="zoom:25%;" />

> **Note:** slides 用的词是 "Continuous-in-Time evaluations"，这里我们把它理解成 "连续时间评估"：我们希望模型的输出不是只在 $t_k$ 这些离散点上好看，而是能把 $t$ 当作一个连续变量来对齐与测试，尤其要能回答 "在没见过的时间点 $t$ 上，它到底靠谱不靠谱".

**#1 solution operator $\mathcal{S}(t,\bar u)=u(t)$ 与 $\mathcal{S}(\Delta t,u(t))=u(t+\Delta t)$ 在这里提供了什么结构.**

Slides 先把 time-dependent PDE 写成抽象形式：

$$
u_t + L(t,x,u)=0,\qquad u(0)=\bar u.
$$

然后把 "解 PDE" 改写成算子学习问题：

$$
\mathcal{S}:(0,T)\times X\to X,\qquad \mathcal{S}(t,\bar u)=u(t).
$$

关键结构是对任意时间增量 $\Delta t$，有

$$
\mathcal{S}(\Delta t,u(t))=u(t+\Delta t).
$$

我们可以把它理解成半群性质在数据层面的直接后果：一条 trajectory 不再只是 "从 $\bar u$ 到 $u(t_k)$ 的若干点"，它也隐含了很多子映射，例如从 $u(t_i)$ 走到 $u(t_j)$ 的映射，lead time 是 $t_j-t_i$. 这为后面的训练策略铺路.

**#2 autoregressive rollout 的公式是什么，slides 列出的 issues 在说什么.**

Slides 的 autoregressive evaluation 假设 trajectory 在均匀时间点上采样：$u(t_k)=u(k\Delta t)$. 然后定义一个 next-step 的 neural operator：

$$
NO_{\Delta t}(u(t_\ell))\approx u(t_\ell+\Delta t).
$$

长时间预测通过 rollout 得到：

$$
u(t_k)\approx \underbrace{NO_{\Delta t}\circ\cdots\circ NO_{\Delta t}}_{k\ {\rm times}}(\bar u).
$$

Slides 列的 issues 可以翻译成 4 个风险点：

1) **需要 uniform spacing：**训练与推理强绑定一个固定 $\Delta t$.  
2) **长 rollout 会带来训练问题：**要么只训 one-step，要么要训多步稳定性，都会变得更难.  
3) **误差累积：**one-step 小误差在多步迭代下会放大.  
4) **只在离散时间点 (time levels) 上评估：**你最多在 $t=k\Delta t$ 上能跑出值，但这不等价于 "连续时间评估".

**#3 time conditioning 的核心做法是什么，lead time 与 conditional normalization 分别在做什么.**

Slides 给的 time conditioning 版本是：把 lead time $\bar t$ 当成输入通道，直接学

$$
{\rm CNO}(\bar t, u(t))\approx \mathcal{S}(\bar t,u(t))=u(t+\bar t).
$$

这件事的直觉是：模型不再只学 "固定步长的下一步"，而是学 "任意 lead time 的推进算子". 这样就天然更贴近 "任意 $t$ 可评估".

但如果只是把 $\bar t$ 拼进输入，网络内部未必真的学会利用时间信息. Slides 的另一个关键设计是 **conditional normalization**：在每一层后加一个依赖 $t$ 的归一化/仿射变换. Slides 写的形式是：

$$
\mathcal{N}(w)=g_N(t)\odot\frac{w-\mathbb{E}(w)}{\sqrt{{\rm Var}(w)+\varepsilon}}+h_N(t),
$$

其中 $g_N,h_N$ 是关于 $t$ 的函数 (slides 说一般用 MLP，但 linear 也足够)，并且可以选择 instance / batch / layer normalization. 我们可以把它理解成一种显式的 "让模型参数随时间变化" 的机制.

> **Note:** 这条式子可以直观读成：把 normalization 的 scale/shift 变成 time-dependent. 先用 $\frac{w-\mathbb{E}(w)}{\sqrt{{\rm Var}(w)+\varepsilon}}$ 把中间特征 $w$ 标准化到相近的数值尺度，再用 $g_N(t)$ 与 $h_N(t)$ 做逐通道的缩放与平移 (这里 $\odot$ 是逐元素乘). 这样时间 $t$ 会在每一层都直接调节 feature 的统计量，比只把 $t$ 拼进输入更容易让网络真的用上时间信息.

<img src="../assets/solution7-fig2-时间条件.png" style="zoom:25%;" />

**#4 all2all training 的 input-target pairs 怎么写，为什么每条 trajectory 有大约 $(K^2+K)/2$ 个样本.**

Slides 先给了一个 "one at a time" 的训练方式：用 pair $(\bar u, \mathcal{S}(t_k,\bar u)=u(t_k))$ 训练. 如果一条 trajectory 有 $K+1$ 个时间点 $t_0=0,t_1,\dots,t_K=T$，那么每条 trajectory 贡献 $K$ 个样本 (对应 $k=1,\dots,K$).

all2all training 则利用半群性质，把每条 trajectory 变成所有 $i<j$ 的 pair：

$$
u(t_i),\qquad \mathcal{S}(t_j-t_i,u(t_i))=u(t_j),\qquad \forall i<j.
$$

<img src="../assets/solution7-fig3-all2all训练.png" style="zoom:25%;" />

这会把每条 trajectory 的样本数从 $K$ 提升到大约

$$
\frac{K^2+K}{2}.
$$

> **Note:** 如果一条 trajectory 给了状态序列 $u(t_0),u(t_1),\dots,u(t_K)$ (共 $K+1$ 个时间点)，all2all 取所有 $i<j$ 的 pair. 计数时可以按 $j$ 来分组：对固定的 $j$，可选的 $i$ 有 $j$ 个，因此 pair 总数是 $\sum_{j=1}^K j = K(K+1)/2 = (K^2+K)/2$.

直觉上就是：$K$ 个时间点的两两组合数量是二次增长的. 这件事的关键收益是：模型在训练时就见过很多不同的 lead time，因此推理时更自然支持 "任意 $t>0$" 的评估.

**#5 什么叫连续时间评估与 OOD 时间点，它和离散时间网格评估的差别是什么.**

Slides 在开头就把 operator learning task 写成 "Continuous-in-Time evaluations"，并且写了目标表述：给定 $\bar u$ (加上边界条件)，生成整个轨迹 $u(t)$，对所有 $t\in(0,T]$.

这里的核心差别是：

- **离散时间网格评估：**只在 $t_k$ 这些训练/采样过的点上评估，模型看起来可能很好，但它可能依赖了时间离散化带来的偶然对齐.  
- **连续时间评估：**把 $t$ 当成连续变量，允许在任意 $t$ 评估，尤其包含 "没见过的时间点" (OOD 时间点). 这会更像一个真正的 solution operator 评估，而不是一个离散序列预测任务.

Slides 也提示 all2all training 的推理既可以 direct，也可以 autoregressive，并且允许 "evaluation at any time $t>0$ including out-of-distribution times". 这里我们可以把它理解成，all2all 给了更丰富的训练监督，从而让 OOD 时间点的评估变得更合理.

<img src="../assets/solution7-fig4-OOD时间.png" style="zoom:25%;" />

> **Note:** 这里说的 "时间对齐的训练样本"，指的是让每个样本都长成同一个输入-输出模板：输入是当前态 $u(t_i)$ 加上 lead time $\Delta t=t_j-t_i$，输出是未来态 $u(t_j)=\mathcal{S}(\Delta t,u(t_i))$. all2all training 就是在一条 trajectory 内把很多不同的 $(t_i,t_j)$ 都枚举出来，从而在训练阶段覆盖多种 $\Delta t$，更贴近我们要做的连续时间评估. Lect12 里 scOT 也强调 "with all2all training"，我们可以把它看作同一个信号：先把训练监督的时间结构组织对齐，再谈换更大的 backbone.
