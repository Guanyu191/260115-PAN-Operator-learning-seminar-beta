## Big Question 4: FNO 的 universal approximation 与跨分辨率表现变差为什么不矛盾？

> **Evidence:** AISE25Lect5.pdf p.35-39; AISE25Lect6.pdf p.13-15，p.24-25; AISE25Lect7.pdf p.2，p.7-8，p.17.

**我们要回答什么：**Slides 一方面说 FNO 有 universal approximation 的理论结果，另一方面又强调 FNO 依然可能不跨分辨率泛化. 这两句话听起来像冲突，但其实中间缺了一个关键概念：**理论的误差定义与假设范围，并不自动包含 "跨表示 (换 grid) 的一致性"**. 我们要把这块缺口补上.

**结论：**Universal approximation theorem 讲的是一个 **存在性** 命题：在给定输入分布 $\mu$ 与目标算子 $\mathcal{G}$ 的前提下，存在某个 FNO 结构 $\mathcal{N}$ 可以把一个定义好的误差 $\hat{\mathcal{E}}$ 做到任意小. 但跨分辨率泛化讨论的是另一件事：当输入/输出函数只能通过编码 $\mathcal{E}$ 与重建 $\mathcal{R}$ 接触时，模型的离散计算链是否与连续算子 **representation equivalent** (CDE/ReNO 视角). FNO 的卷积 kernel 在某些理想假设下可能满足 ReNO，但**pointwise 非线性会破坏 bandlimit**，从而使得 "FNOs are not necessarily ReNOs"，因此换 grid 时出现 discrepancies 并不矛盾.

> **Note:** 我们可以把这件事分成两层问题：  
> (1) 表达能力：有没有能力逼近 $\mathcal{G}$ (universal approximation).  
> (2) 表示一致性：逼近的方式是否与 "换网格/换表示" 可交换 (ReNO).  
> 第 (1) 不自动推出第 (2).

**#1 FNO 为什么要假设 translation-invariant kernel，它如何导向 Fourier space 的实现.**

FNO 从 operator layer 的 kernel integral operator 出发，但额外加了一个结构假设：**平移不变**，

$$
K(x,y)=K(x-y).
$$

这会把核积分

$$
\int_D K(x,y)v(y){\rm d}y
$$

变成卷积 $K\ast v$. 卷积的直接收益是：它可以在 Fourier domain 里变成逐点乘法，因此可以用 FFT 做快速实现. Slides 的表达是把核积分写成

$$
K\ast v = \mathcal{F}^{-1}(\mathcal{F}(K)\mathcal{F}(v)),
$$

并且在 Fourier space 里参数化 kernel，同时截断到固定数量的 Fourier modes，得到可计算的实现.

**#2 Universal approximation theorem 在说什么，它的误差 $\hat{\mathcal{E}}$ 是怎么定义的.**

Slides 给出的表述非常直接：对输入分布 $\mu$ 与目标算子 $\mathcal{G}$，以及任意精度 $\varepsilon>0$，存在一个 FNO 结构 $\mathcal{N}$ 使得误差 $\hat{\mathcal{E}}<\varepsilon$.

<img src="../assets/solution4-fig2-UA定理.png" style="zoom:25%;" />

这里的误差是一个 "对输入分布取期望的 $L^2$ 输出误差"，其平方形式是

$$
\hat{\mathcal{E}}^2=\int_X\int_U |\mathcal{G}(u)(y)-\mathcal{N}(u)(y)|^2\,{\rm d}y\,{\rm d}\mu(u).
$$

这个定义在直觉上对应：按 $\mu$ 抽样得到一个输入函数 $u$，看模型输出 $\mathcal{N}(u)$ 与真实输出 $\mathcal{G}(u)$ 在空间域 $U$ 上的 $L^2$ 距离，然后再对 $\mu$ 做平均.

> **Note:** 这里我们只强调一个点：这是 **存在性** + **误差定义**. 它没有承诺训练一定找到这个 $\mathcal{N}$，也没有承诺 $\mathcal{N}$ 在换一种表示方式后仍然与同一个连续算子保持一致.

**#3 为什么 universal approximation 不等价于 "跨分辨率泛化一定好".**

跨分辨率泛化的问题，本质上来自一个现实约束：计算机里输入/输出永远是离散的，我们只能通过

- 编码 (discretization) $\mathcal{E}$ 把连续函数变成离散表示，  
- 再通过重建 (reconstruction) $\mathcal{R}$ 把离散表示还原成连续对象或可比较的输出.

CDE/ReNO 的核心就是把 "换表示 (换 grid)" 变成一个可判定的一致性要求：离散计算链是否与连续算子在该 $\mathcal{E}/\mathcal{R}$ 下相容. 也就是说，除了端到端误差小以外，我们还关心模型内部是否引入了与分辨率强绑定的 aliasing / discrepancy.

Universal approximation theorem 并没有把这件事写进假设或结论里，因此它不可能自动推出 "跨 grid 一定泛化好". 这不是理论错了，而是它回答的问题不同：它回答的是 **表达能力**，不是 **表示一致性**.

**#4 在 ReNO 视角下，FNO 的 kernel 与 activation 各自的问题.**

Slides 对 FNO 的拆法是：**Fourier space 的卷积 kernel $K$ + pointwise 非线性 $\sigma$**.

在理想化设定下，如果我们把函数空间限制为周期且 bandlimited 的函数类 $P_K$，那么卷积这一部分更容易与表示变化相容 (直觉上，这是因为 Fourier modes 的截断与卷积结构给了一个更 "可控" 的线性算子).

但问题出在 pointwise 非线性：非线性 $\sigma$ 会把一个 bandlimited 的函数映射到一个可能不再 bandlimited 的函数. 一旦 bandlimit 被打破，后续离散表示就会出现 aliasing 风险，跨分辨率的一致性也就不再被保证. 这就是 slides 用一句话总结 "FNOs are not necessarily ReNOs" 的关键原因.

**#5 "Random Assignment" 的曲线随 resolution 变化在提醒什么风险 (以及它不能说明什么).**

Slides 用一个 synthetic 的 "Random Assignment" 例子画了 "Discrete Aliasing Error (%) vs Resolution" 的曲线，想强调的风险是：**如果一个模型不是 representation equivalent，那么它在某个固定分辨率上看起来表现不错，并不意味着换一个分辨率仍然可信**. 分辨率变化会把隐藏的 aliasing / discrepancy 放大出来.

<img src="../assets/solution4-fig1-随机分配曲线.png" style="zoom:25%;" />

同时，这页图也有明显的边界：
- **它能说明的：**这是一个用来检验表示一致性的合成例子，核心信息是 "换分辨率会暴露 representation inconsistency".  
- **它不能直接说明的：**它不等价于对真实 PDE 任务的最终结论，也不能单凭这一页就推出某个模型在所有任务上必然更好；它更像是在告诉我们：跨分辨率泛化需要额外结构保证 (例如 ReNO 的构造性约束)，而不是只靠 universal approximation 这种存在性表述.
