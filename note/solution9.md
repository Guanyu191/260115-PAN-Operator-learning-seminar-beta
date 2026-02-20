## Big Question 9: 对 chaotic multiscale PDEs，为什么 "回归一个确定性解算子" 可能目标就不太合适 (应该学 conditional distribution)？

> **Evidence:** AISE25Lect11.pdf p.4-6，p.10-12，p.15-20.

**我们要回答什么：**Slides 在 Lect11 里强调一个 caveat：当 PDE 的解是 chaotic multiscale 的时候，直接用监督回归去学一个确定性的解算子 (给定 $u_0$ 输出一个 $u(t)$) 会出现典型失效模式，slides 把它称为 "collapse to mean" (AISE25Lect11.pdf p.5). 我们要把 slides 给的原因链条复述清楚，并解释为什么学习目标会从 deterministic operator 推向 conditional distribution $P(u(t)\mid u_0)$ (AISE25Lect11.pdf p.6). 最后，我们需要把 GenCFD 的训练目标写出来，并说明 slides 用哪些指标验证 "分布学对了" 与推理速度 (AISE25Lect11.pdf p.10-12，p.15-20).

**结论：**简单来说就两句话：

1) **对 chaotic multiscale PDE，确定性回归模型倾向于输出过度平滑的平均态 (collapse to mean).** Slides 给出一条解释链：神经网络的 insensitivity，训练中倾向 edge of chaos (Lip $\sim O(1)$)，再叠加 spectral bias 与 bounded gradients 的约束，会让模型很难表达 "小扰动 -> 大变化" 的混沌放大机制，最终预测会向平均态收缩 (AISE25Lect11.pdf p.5).  
2) **因此更合理的目标是直接学习条件分布 $P(u(t)\mid u_0)$，并用生成模型去采样这个分布.** Slides 以 statistical solutions 的观点出发，认为 "Only Statistical Computation is feasible"，并提出用 conditional score-based diffusion (GenCFD) 去逼近条件分布，结果用 mean/variance、point pdfs、spectra、Wasserstein distance 以及 runtime 对比来验证 (AISE25Lect11.pdf p.6，p.10-12，p.15-20).

> **Note:** 这里我们可以把 "collapse to mean" 当成一个非常具体的现象描述：模型的输出看起来像很多真实样本的平均，细节被抹平，高频能量偏少，同时预测分布的方差会被显著低估.

**#1 "Collapse to mean" 在预测输出里最直接的表现是什么.**

Slides 在 Taylor-Green vortex 的例子里用一组图来表达 "真实解是多样的" 与 "我们需要评估分布层面的性质"：

- **Samples:** kinetic energy / vorticity 的样本展示了小尺度结构随样本变化而变化 (AISE25Lect11.pdf p.13-14).  
- **Mean / Variance：**分别展示了条件分布的均值与方差场 (AISE25Lect11.pdf p.15-16).  
- **Point pdfs / Spectra：**在点位与频域上直接对比分布 (AISE25Lect11.pdf p.17-18).

下面两张图分别对应 Samples 与 Spectra.

<img src="../assets/solution9-fig2-能量样本.png" style="zoom:25%;" />

<img src="../assets/solution9-fig3-频谱对比.png" style="zoom:25%;" />

在这个语境下，"collapse to mean" 最直观的表现是：

1) **样本多样性消失：**预测更像某个单一的平均态，而不是不同 realization.  
2) **方差被低估：**如果模型本质上只输出一个 mean-like 的场，那么对应的 variance 近似会偏小 (甚至接近 0).  
3) **频谱过度衰减：**小尺度结构缺失会体现为高频能量被系统性压低 (AISE25Lect11.pdf p.18).

**#2 Slides 列出的 4 个点如何导向 "collapse to mean" (推理链).**

Slides 在一页里把原因写成 4 个点 (AISE25Lect11.pdf p.5)：

<img src="../assets/solution9-fig1-均值塌缩.png" style="zoom:25%;" />

1) **Insensitivity：**$\Psi_\theta(u+\delta u)\approx \Psi_\theta(u)$，当 $\delta u\ll 1$.  
2) **Edge of chaos：**训练得到的 DNN 往往处于 "edge of chaos"，其 Lipschitz 常数 ${\rm Lip}(\Psi_\theta)\sim O(1)$.  
3) **Spectral bias：**DNN 更容易拟合低频成分.  
4) **Bounded gradients：**用 GD 训练时需要梯度有界.

我们把它们串起来，可以这样理解：

- chaotic PDE 的一个核心困难是：**微小初值扰动 $\delta u$ 会在时间演化后被放大成宏观差异**.  
- 但 (1)(2)(4) 共同把网络推向一个更 "不放大扰动" 的函数类：网络对小扰动不敏感，且 Lipschitz 常数被训练稳定性约束在 $O(1)$ 的量级.  
- 叠加 (3) 的 spectral bias，模型更倾向于用平滑的低频结构去解释数据.  

因此，在需要表达 "强敏感 + 多尺度" 的任务上，确定性 DNN 回归更容易收缩到一个更稳定的平均态，这就是 slides 用一句话概括的 "Implication $\Rightarrow$ DNNs will Collapse to Mean" (AISE25Lect11.pdf p.5).

> **Note:** 上面这段推理的重点不是某个定理，而是一个对齐：混沌系统需要放大微小差异，而可训练的神经网络倾向于不放大差异. 当两者对不上时，回归模型就会把不确定性 "平均掉".

**#3 为什么要把目标改成 conditional distribution $P(u(t)\mid u_0)$，它与 Big Question 1 的 $\mathcal{G}_{\#}\mu$ 视角如何对齐.**

Slides 引用 statistical solutions 的观点，给出结论："Only Statistical Computation is feasible"，并把学习目标写成：直接近似条件分布 $P(u(t)\mid u_0)$ (AISE25Lect11.pdf p.6).

我们可以把这件事和 Big Question 1 的语言对齐起来：

- 在 Big Question 1 里，我们写过 $a\sim\mu$ 时输出分布是 $\mathcal{G}_{\#}\mu$. 这里强调的是：在数据驱动场景下，我们关心的是 **输出分布**，而不仅是某个点值预测.  
- chaotic multiscale 的场景进一步强化了这一点：即使给定同一个宏观初值 $u_0$，微观扰动或未解析尺度也会让 $u(t)$ 呈现一个分布，而不是一个单点. Slides 在 "Why should this Work" 页用
  $$
  {\rm Law}_{\delta \bar u}\big(\mathcal{S}(\bar u^\ast+\delta \bar u)\big)
  $$
  来提示这种 "由扰动推动的条件分布" 视角 (AISE25Lect11.pdf p.12).

  > **Note:** slides 这里的 ${\rm Law}_{\delta \bar u}(\cdot)$ 可以理解成：当微扰 $\delta \bar u$ 按某个分布抽样时，括号里这个随机变量在输出空间里诱导出的概率分布. 这句话想强调的不是某个具体分布形式，而是 "同一个宏观初值下，微观扰动会把输出推成一个分布" 这一层目标变化.

因此，把目标改成 $P(u(t)\mid u_0)$ 本质上是在说：我们不再要求模型给出唯一的 $u(t)$，而是要求它生成与真实动力系统一致的统计行为 (均值、方差、频谱、点分布等).

**#4 GenCFD 的训练目标是什么 (conditional score-based diffusion + denoiser objective)，conditioning 里包含哪些变量.**

Slides 给出的 GenCFD 关键描述是两句话 (AISE25Lect11.pdf p.10-11)：

- GenCFD 基于 **Conditional Score Based Diffusion Models**.  
- 用 reverse SDE 进行 denoising，训练一个 denoiser $D_\theta$，使其最小化
  $$
  \mathbb{E}\left\|u(t_n,\bar u)-D_\theta\big(u(t_n,\bar u)+\eta,\,\bar u,\,\sigma\big)\right\|.
  $$

我们把它翻译成更直白的描述就是：

1) 取一个时间 level $t_n$ 上的真实场 $u(t_n,\bar u)$.  
2) 给它加噪声：$u(t_n,\bar u)+\eta$ (噪声强度由 $\sigma$ 控制).  
3) 训练 $D_\theta(\cdot,\bar u,\sigma)$ 去把 noisy field denoise 回真实 field.

因此 slides 在符号层面明确的 conditioning 至少包含两项：**初值/条件 $\bar u$** 与 **噪声尺度 $\sigma$**. 时间 $t_n$ 在 slides 的写法里是通过 "我们正在 denoise 哪个 time slice" 来出现的；是否把 $t_n$ 显式作为额外输入，在 Lect11 的文字页里没有展开，这里不额外补充超出 slides 的设定.

**#5 Slides 用哪些指标验证生成分布的质量与推理速度，它们各自验证了什么.**

Slides 把验证分成两类：分布质量与推理速度.

**(1) 分布质量：先看结构性诊断，再看数值指标.**

- **Mean / Variance：**检查生成分布的一阶与二阶统计量是否匹配 (AISE25Lect11.pdf p.15-16).  
- **Point pdfs：**在固定点位上对比边缘分布的形状，检查 "是不是只学到均值附近" (AISE25Lect11.pdf p.17).  
- **Spectra：**在频域检查高频能量是否被系统性抹平 (AISE25Lect11.pdf p.18).

Slides 还给出一个数值指标面板，在 Taylor-Green 的 micro-macro setup 下比较 GenCFD、C-FNO 与 UViT 的指标 (AISE25Lect11.pdf p.19)：

- **Mean Error：**GenCFD 0.154，C-FNO 0.210，UViT 0.883.  
- **Std Error：**GenCFD 0.056，C-FNO 1.000，UViT 0.813.  
- **Wasserstein distance：**GenCFD 0.017，C-FNO 0.117，UViT 0.130.

<img src="../assets/solution9-fig4-指标面板.png" style="zoom:25%;" />

我们可以把这些指标各自对应到一个 "它在验证什么" 的解释上：

- mean error / std error：分别对应分布的一阶与二阶统计是否对齐.  
- Wasserstein distance：更像是一个整体的 distribution distance，用来衡量生成分布与真实分布的差距.

**(2) 推理速度：与数值模拟对比.**

Slides 在真实流动例子里直接给出 "simulator hours vs GenCFD seconds" 的对比 (AISE25Lect11.pdf p.20)：

- Nozzle Jet: 3.5 hrs (LBM) vs GenCFD 1.45s.  
- Cloud-Shock: 5 hrs (FVM) vs GenCFD 0.45s.  
- Conv. Boundary Layer: 13.3 hrs (FDM) vs GenCFD 3.8s.

<img src="../assets/solution9-fig5-推理加速.png" style="zoom:25%;" />

> **Note:** 这里的逻辑收束点是：对 chaotic multiscale PDE，"学分布" 不是为了更花哨，而是为了让学习目标与可计算性对齐. 当我们接受 "只能做统计预测" 这个前提时，生成式模型给了一个可行的近似路径，同时把推理速度拉到了和 surrogate 模型同一量级.
