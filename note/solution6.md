## Big Question 6: 我们如何评价 "operator model 做得好"，以及 sample complexity 为什么是主瓶颈 (我们有什么路线提升 data efficiency)？

> **Evidence:** AISE25Lect7.pdf p.19-25，p.30-32; AISE25Lect8.pdf p.5-9，p.14-17，p.24-31，p.33-39; AISE25Lect12.pdf p.3-5，p.14-18，p.39-42，p.45.

**我们要回答什么：**Slides 强调评估结果更像一条 curve 或一个 histogram，而不是一个 point. 我们要把这句话变成可执行的评估协议，并说明为什么 sample complexity 是 operator learning 的硬瓶颈，以及 slides 给出的两条提升 data efficiency 的路线分别是什么、证据是什么.

**结论：**简单来说就三句话：

1) **评估不是一个数字，而是一组多方面的测试.** 至少要同时看：In/Out-of-distribution (zero-shot) 测试、resolution sweep (跨网格)、spectral behavior (log spectra).  
2) **sample complexity 是主瓶颈，因为 scaling 太慢且每个任务需要的大样本太贵.** Slides 用 $E\sim N^{-\alpha}$ (且 $\alpha$ 很小) 来概括误差随样本增长的缓慢下降，并给出每个任务通常需要 $O(10^3)$ 到 $O(10^4)$ 条训练样本的量级，这对 PDE 数据生成是很不友好的.  
3) **提升 data efficiency 的两条路线：**
   1) **physics-informed neural operators**. 把 PDE residual 加进 loss，但 slides 给出负面证据 "not even for simple problems".  
   2) **foundation model / pretrain + finetune**. 以 Poseidon 为例，用 scOT backbone 做大规模预训练，再迁移到下游任务；transfer latents 与 scaling 曲线给出 "预训练确实学到可迁移结构" 的证据.

**#1 In/Out-of-distribution testing：用 Poisson 例子写一条可复现协议.**

我们选 slides 里的 Poisson 例子把 protocol 写清楚：

- **PDE：**$-\Delta u=f$.  
- **目标算子：**$\mathcal{S}:f\mapsto u$.  
  - **数据分布：**Slides 里把 $f$ 写成一组正弦基的随机线性组合，并用一个频率截断参数 $K$ 控制 "最高频"：
  $$
  f \sim \sum_{i,j=1}^K \frac{a_{ij}}{(i^2+j^2)^\alpha}\sin(i\pi x)\sin(j\pi y),\qquad a_{ij}\sim U[-1,1].
  $$
  训练时固定一个 $K_{\rm train}$，测试时用 $K_{\rm test}$ 来定义 in-distribution 或 out-of-distribution.  
- **In-distribution testing：**$K_{\rm test}=K_{\rm train}$.  
- **Out-of-distribution (zero-shot) testing：**用更高的频率截断 $K_{\rm test}>K_{\rm train}$，也就是让测试集包含更高频的 source，从而逼迫模型面对更细的结构.  
- **指标：**对测试集计算相对误差 (slides 用百分比展示)，并且最好同时给出误差分布 (histogram) 或随样本量变化的 curve，而不是只报一个均值.

这个例子里 slides 还把 $K$ 具体化为: in-distribution 用 $K=16$，out-of-distribution 用 $K=20$. 对应的 test errors (相对误差) 是：

- **In-distribution：**FFNN 5.74%，UNet 0.71%，DeepONet 12.92%，FNO 4.78%，CNO 0.23%.  
- **Out-of-distribution：**FFNN 5.35%，UNet 1.27%，DeepONet 9.15%，FNO 8.89%，CNO 0.27%.  

直觉上，这件事在回答：模型到底是在 "背训练分布的频率范围"，还是在学更接近算子层面的规律.

<img src="../assets/solution6-fig1-Poisson协议.png" style="zoom:25%;" />

**#2 "Success is a histogram / curve rather than a point"：这句话在评估里怎么落地.**

如果我们只报一个 test error，很容易把不稳定性与长尾掩盖掉. 更具体地说：

- **用 histogram 看长尾：**同一个模型在不同样本/不同输入上可能差异巨大，median 可能好看但 worst-case 很差.  
- **用 curve 看 scaling：**把误差画成随训练样本数 $N$ 变化的曲线，才知道模型是否真的在 "继续变好"，还是早早进入平台期.  

这两件事一旦落到 operator learning，会自然逼出下面两类测试，分别是 resolution sweep 与 spectral behavior.

<img src="../assets/solution6-fig2-直方图评估.png" style="zoom:25%;" />

**#3 Resolution dependence + log spectra：能直接读出的结论与不能直接推出的结论.**

Slides 在 "Further Results" 里把两件事并列：

1) **Resolution dependence：**同一个输入在不同网格分辨率上评估，误差是否显著变化.  
2) **Spectral behavior (log spectra)：**把预测解与真值的频谱 (能量随频率的分布) 放在一起看，检查模型是否系统性丢失高频 (常见表现是过度平滑).

<img src="../assets/solution6-fig3-分辨率频谱.png" style="zoom:25%;" />

我们可以把这两种图的读法总结成：

- **能直接读出的：**有没有明显的 resolution sensitivity；有没有明显的频谱偏差 (例如高频能量被系统性压掉).  
- **不能直接推出的：**不能仅凭频谱或分辨率曲线就断言 "模型一定更稳定/更物理"；也不能直接把原因归咎为 aliasing、训练不足或网络结构，需要结合 controlled experiments 才能定位.

**#4 sample complexity：为什么说它是主瓶颈.**

Slides 给了一个明确的结论链：

- **Scaling 很慢：**$E\sim N^{-\alpha}$，并且 $\alpha$ 很小.  
- **每个任务要大样本：**每个 PDE operator learning task 往往需要 $O(10^3)$ 到 $O(10^4)$ 条训练样本.  
- **PDE 数据很难搞：**生成一条高质量轨迹/样本的成本不低，因此 "多收集数据" 不是一个轻松选项.

<img src="../assets/solution6-fig4-Scaling曲线.png" style="zoom:25%;" />

这就把问题推到一个更现实的目标：我们能不能让模型用更少的数据学到更稳的东西.

**#5 Attention as a Neural Operator：为什么它是 nonlinear kernel neural operator，它和 FNO/CNO 的差别是什么.**

Slides 把 self-attention 从 token 版本推到连续极限，得到一个 operator 形式：

$$
u(x)=A(v)(x)=W\int_D
\frac{\exp\left(\frac{\langle Qv(x),Kv(y)\rangle}{\sqrt{m}}\right)}
{\int_D \exp\left(\frac{\langle Qv(z),Kv(y)\rangle}{\sqrt{m}}\right){\rm d}z}
Vv(y)\,{\rm d}y.
$$

<img src="../assets/solution6-fig8-注意力算子.png" style="zoom:25%;" />

它可以被解释成

$$
A(v)(x)=\int_D K(v(x),v(y))v(y)\,{\rm d}y,
$$

也就是一个 **nonlinear kernel neural operator**: kernel 依赖于输入函数在 $x,y$ 处的取值.

对比之下，FNO/CNO 这类更接近

$$
C(v)(x)=\int_D K(x,y)v(y)\,{\rm d}y
$$

的 **linear kernel** 形式 (kernel 不依赖 $v$，只依赖坐标/相对坐标). 这不是说 attention 一定更好，而是在提醒我们，attention 的表达能力更强，但复杂度与可扩展性也会变得更尖锐.

**#6 为什么 vanilla transformer 对 2D/3D 不可行，ViT patching 与 windowed attention 分别在复杂度上解决了什么.**

Slides 对 vanilla transformer 的诊断是：计算成本对 token 数 $K$ 是二次的，粗略写成 $O(mnK^2)$. 2D/3D 输入一旦把每个网格点都当 token，$K$ 会爆炸，因此 "infeasible for 2 or 3-d inputs".

两条结构化改造路径分别是：

1) **ViT patching (patchification)：**把高分辨率图像切成 $p\times p$ 的 patch，每个 patch 变成一个 token，从而把 token 数从 $HW$ 级别降到 $N\approx (HW)/p^2$ 级别. 对应复杂度从 $O((HW)^2)$ 降到 $O((HW)^2/p^4)$.  
2) **Windowed attention：**把 attention 限制在局部 window 内，计算量不再按全局 token 两两交互增长；同时通过跨层 window shift 来让信息在更大范围传播.

<img src="../assets/solution6-fig9-窗口注意力.png" style="zoom:25%;" />

Slides 还给了一个例子：scOT (scalable Operator Transformer) 基于 Swin attention，并且宣称是 continuous operators 的 universal approximator. 我们可以把它理解成：它把 "attention 的表达力" 和 "operator learning 的可扩展性" 这两件事尽量对齐，但它并没有让 sample complexity 问题消失，只是把模型类换到另一条可能更有效的曲线.

**#7 两条 data efficiency 路线: physics-informed NO vs foundation model (Poseidon).**

路线 I: **Physics-informed neural operators (把 PDE residual 加进 loss).** Slides 写了一个联合目标：

$$
\sum_{i=1}^N\Big[\|\mathcal{S}(f_i)-\mathcal{S}_\theta(f_i)\|_Y+\lambda\|L(\mathcal{S}_\theta(f_i))-f_i\|_Z\Big].
$$

直觉上这在做两件事：既拟合数据 $(f_i,\mathcal{S}(f_i))$ (也就是 $\{(f_i,\mathcal{S}(f_i))\}_{i=1}^N$ 这样的监督对)，又用 residual 强迫输出满足 PDE. 但 slides 紧接着给了一个负面证据，即 **"Not even for Simple problems"**，并用线性平流方程举例. 这里的主要风险是，PDE residual 并不会自动带来更好的 data efficiency，甚至可能引入优化与泛化层面的新问题.

<img src="../assets/solution6-fig5-残差负证据.png" style="zoom:25%;" />

路线 II: **Foundation model，pretrain + finetune (Poseidon).** Slides 给出的证据链主要落在两类图上：

1) **Transfer latents vs frozen latents：**在相同的少样本条件下，transfer latents 的误差显著更低，这支持 "预训练学到的表征是可迁移的，而且允许迁移会更有收益".  
2) **Scaling curves：**随着 trajectory 数增加，Poseidon 系列模型的 error curve 与基线模型的差距提供了 "更 data efficient" 的证据.

<img src="../assets/solution6-fig6-迁移latent.png" style="zoom:25%;" />

为了进一步支撑我们的结论，这里把 slides 的关键信息摘出来：

- **Transfer vs frozen 的对比：**在一个下游设定里，32 trajectories 时的误差对比是: FNO 3.69%，frozen latents 3.1%，transfer latents 1.35%. 这支持的结论是，预训练 latent 不是摆设，允许迁移能显著降低误差.  
- **Scaling 的对比：**Poseidon 的 scaling 曲线把它和 FNO、CNO、scOT 等 baselines 放在一起，展示 "更 data efficient" 的趋势 (尤其在小样本区间差距更明显).  
- **成本对比：**推理成本很低，但 PDEgym 数据生成可能要 $10^{-1}$ 到 $10^2$ 秒量级. 这会让 "用更强的预训练模型去省数据" 更有吸引力，因为数据才是贵的.

<img src="../assets/solution6-fig7-推理成本.png" style="zoom:25%;" />
