## Big Question 5: CNO 为什么是 ReNO，它具体解决了 FNO/CNN 的哪类 aliasing 问题？

> **Evidence:** AISE25Lect7.pdf p.11-16; AISE25Lect8.pdf p.4.

**我们要回答什么：**我们想要的是 "跨分辨率一致" 这件事不仅是经验现象，而是被结构保证 (ReNO by construction). 用 slides 的话说，CNO 把一个 neural operator 的两块高风险源头拆开处理：
1) 卷积算子本身如何做到 representation equivalent.
2) pointwise 非线性为什么会打破 band limit，引入 aliasing (以及 CNO 怎么处理).

**结论：**CNO，简单来说就是做两件事：

<img src="../assets/solution5-fig1-CNO概览.png" style="zoom:25%;" />

1) **Continuous convolution (在 bandlimited 函数空间上定义).** 卷积核在实现上仍然是离散参数，但卷积算子被定义成对 bandlimited 函数的连续算子，因此它可以被构造成 ReNO，而不是像 CNN 那样被网格索引绑定.  
2) **Alias-aware activation (上采样 -> 非线性 -> 下采样).** pointwise 非线性会产生高频并打破 band limit. CNO 用
$$
\Sigma = D_{\bar w,w}\circ\sigma\circ U_{w,\bar w}
$$
把激活变成一个 "先扩带宽、再做非线性、再做抗混叠下采样" 的算子，从而把非线性带来的 aliasing 风险变成可控条件 (在 $\bar w\gg w$ 的设定下，activation 也可以满足 ReNO 的一致性要求).

> **Note:** 直觉上，CNO 的路线是：先把讨论限定在一个 "表示可控" 的函数类里 (bandlimited)，再分别把线性算子与非线性算子做成和表示变化相容的形式. 这样跨分辨率 discrepancy 不再是 "希望它别发生"，而是 "结构上尽量不允许它发生".

**#1 为什么 CNO 要强调 "operator between band-limited functions"，这个假设在 CDE/ReNO 语境下解决了什么问题.**

在 CDE/ReNO 的语境下，我们永远绕不开 $\mathcal{E}/\mathcal{R}$：连续对象必须被采样 (encoding) 才能进计算机，离散结果也必须被重建 (reconstruction) 才能和连续算子对齐比较. Bandlimited 的假设提供了一个关键便利：在理想化条件下，采样与重建可以被当作 "信息不丢失" 的表示变化，这样我们才能更认真地讨论 "离散实现是否真的在学算子，而不是学某个固定网格上的离散映射".

因此，CNO 把工作空间限定在 bandlimited 函数类上，可以理解为先把 "表示变化本身就会丢信息" 这种不可控因素移出讨论范围，把注意力集中到：模型内部的卷积与非线性是否引入了额外的 aliasing.

**#2 Continuous convolution vs discrete convolution：为什么 CNN 被判定为非 ReNO，CNO 如何避免 resolution dependence.**

Slides 对 CNN 的批评是：CNN 用的是 **离散卷积**，卷积核 $K_c[m]$ 与网格索引绑定；当我们改变表示 (换一种 grid / 分辨率) 时，这个离散算子通常不与 $\mathcal{E}/\mathcal{R}$ 可交换，因此会出现 resolution dependence.

> **Note:** 这里的 "可交换"，指的是 "换表示" 与 "做卷积" 的先后顺序不应该影响结果：如果我们有两套表示 (两张网格) 之间的转换 $\mathcal{E},\mathcal{R}$，那么在新网格上做离散卷积 $\mathcal{G}'$，应该等价于 "先转换到旧网格 -> 用旧网格的离散卷积 $\mathcal{G}$ -> 再转换回新网格". 如果这两条路径对不上，就意味着卷积算子依赖分辨率.

Slides 用一个不等式把这种不相容性写得很具体 (AISE25Lect7.pdf p.13)：

$$
\mathcal{G}' \neq \mathcal{E}' \circ \mathcal{R} \circ \mathcal{G} \circ \mathcal{E} \circ \mathcal{R}'.
$$

<img src="../assets/solution5-fig4-CNN不等式.png" style="zoom:25%;" />

CNO 的做法是把卷积算子提升回连续层面：我们仍然可以用离散参数表示 kernel，但卷积被定义为对 bandlimited 函数的 **continuous convolution**. 在这种构造下，卷积算子本身被宣称为 ReNO，这就从结构上把 "卷积导致的跨分辨率不一致" 这个风险压下去.

**#3 Activation operator $\Sigma$ 在做什么，上采样/下采样各自为了解决什么问题，为什么需要 $\bar w \gg w$.**

非线性 $\sigma$ 的问题是：即使输入是 bandlimited 的，$\sigma(f)$ 也可能不再 bandlimited. 这会把 aliasing 风险引入到后续的离散表示里.

CNO 的策略是把激活改写成一个算子：

$$
\Sigma = D_{\bar w,w}\circ\sigma\circ U_{w,\bar w}.
$$

<img src="../assets/solution5-fig2-抗混叠激活.png" style="zoom:25%;" />

> **Note:** Slides 还给了一个非常具体的 downsampling 形式 (用 sinc 做抗混叠投影)：
> $$
> D_{\bar w,w}f(x)=\left(\frac{\bar w}{w}\right)^d\int_D \mathrm{sinc}(2\bar w(x-y))f(y)\,{\rm d}y.
> $$
> 这里我们可以把它理解成：先在更高带宽里做非线性，再用一个 "带限滤波 + 投影" 把结果压回目标带宽.

对应的直觉分解是：

1) **Upsampling $U_{w,\bar w}$：**先把带宽从 $w$ 提到更高的 $\bar w$，给非线性 "预留频谱空间".  
2) **Apply nonlinearity $\sigma$：**在更高带宽的表示里做 pointwise 非线性.  
3) **Downsampling $D_{\bar w,w}$：**再把结果带宽压回 $w$，并且用带有 sinc 核的方式做一个抗混叠的投影/滤波.

为什么需要 $\bar w \gg w$：因为非线性会产生更高频成分. 如果 $\bar w$ 不够大，那么这些高频会立刻折叠回低频 (aliasing). 当 $\bar w$ 足够大时，非线性产生的高频更可能被 "暂时容纳"，再通过下采样时的滤波更可控地回到 $B_w$. Slides 用一句话把它总结为：在 $\bar w \gg w$ 的设定下，activation 可以被视为一个 ReNO.

**#4 为什么 CNO 用 UNet 结构实例化 (把动机压缩成 1 句话).**

Slides 的动机词是 "Built for multiscale information processing". UNet 的编码器-解码器结构天然在做多尺度信息的提取与融合，因此它适合承载 CNO 这种以 "频谱/分辨率一致性" 为核心约束的 operator 架构.

<img src="../assets/solution5-fig3-UNet架构.png" style="zoom:25%;" />

**#5 CNO 的两条性质分别在什么前提下陈述.**

Slides 给出的两个性质是：

1) **ReNO by construction：**在它的构造假设下 (工作在 bandlimited 函数类上，并用 continuous convolution + alias-aware activation)，CNO 被设计成 representation equivalent 的 neural operator.  
2) **Universal approximation：**CNO 给出一个 universal approximation 方向的表述：它可以逼近某一类连续算子 $\mathcal{G}:H^r\to H^s$. Slides 还提示证明的关键是把目标算子在 bandlimited 空间之间做近似构造 (例如 $\mathcal{G}\approx \mathcal{G}^*:B_w\to B_{w'}$)，从而把逼近问题落到更可控的函数类上.
