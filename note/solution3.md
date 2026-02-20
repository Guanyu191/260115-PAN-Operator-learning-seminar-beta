## Big Question 3: 一个 neural operator layer 的结构是什么，为什么离散实现会逼迫我们选择特定 kernel 参数化？

> **Evidence:** AISE25Lect5.pdf p.32-34; AISE25Lect6.pdf p.10-12.

**我们要回答什么：**一个 neural operator layer 在数学上长什么样 (它相对 DNN layer 换掉了什么)，以及它一旦在离散网格上实现，为什么会出现很强的计算复杂度压力，逼迫我们把 kernel $K(x,y)$ 做成特定结构 (low-rank / Fourier / graph / multipole 等).

**结论：**一个 neural operator layer，简单来说可以写成

$$
(\mathcal{N}_{\ell}v)(x)=\sigma\left(\int_{D}K_{\ell}(x,y)v(y){\rm d}y+B_{\ell}(x)\right).
$$

它本质上把 DNN layer 的 `bias vector + 矩阵乘法 + 激活`，替换成 `bias function + kernel integral operator + pointwise 激活`. 一旦离散到 $n$ 个点，朴素实现的 kernel integral operator 需要 $O(n^2)$ 级别的两两交互 (以及潜在的 $O(n^2)$ 存储)，因此必须引入带结构的 kernel 参数化来降复杂度 (这是 slides 列出一堆 kernel family 的直接动机).

> **Note:** 这里的关键不是 "把公式写出来"，而是理解为什么 operator layer 的瓶颈天然落在 $K(x,y)$ 的离散计算上：只要它还是一个全局的 pairwise interaction，它就会在网格加密时迅速变得不可用.

**#1 neural operator layer 相比 DNN layer 的 3 个替换.**

回忆一个 DNN 的单层可以写成

$$
\sigma_k(y)=\sigma(A_k y + B_k).
$$

对应地，neural operator layer 的单层是

$$
(\mathcal{N}_{\ell}v)(x)=\sigma\left(\int_{D}K_{\ell}(x,y)v(y){\rm d}y+B_{\ell}(x)\right).
$$

三个替换对应的位置是：

1) **Bias vector -> bias function.** 把 $B_k$ 从一个向量换成函数 $B_{\ell}(x)$，它给每个空间位置 $x$ 一个偏置 (而不是给每个 neuron 一个偏置).  
2) **Matrix-vector multiply -> kernel integral operator.** 把 $A_k y$ 换成对输入函数 $v(\cdot)$ 的积分型线性算子 $\int_D K_{\ell}(x,y)v(y){\rm d}y$，其中 $K_{\ell}(x,y)$ 类似于一个连续域上的 "权重矩阵".  
3) **Activation stays pointwise.** 激活仍然是 pointwise 的：对每个 $x$，对积分结果做 $\sigma(\cdot)$.

对应地，整网路从 $L_{\theta}=\sigma_K\odot\cdots\odot\sigma_1$ 变成 $\mathcal{N}_{\theta}=\mathcal{N}_L\odot\cdots\odot \mathcal{N}_1$，其中每一层 $\mathcal{N}_{\ell}:X\mapsto X$ 都在函数空间里做一次 "线性算子 + pointwise 非线性".

**#2 单层的数学形式与哪些参数是可学习的.**

Slides 直接把单层写成：

$$
(\mathcal{N}_{\ell}v)(x)=\sigma\left(\int_{D}K_{\ell}(x,y)v(y){\rm d}y+B_{\ell}(x)\right).
$$

这里可学习的参数落在两块：
- **Kernel $K_{\ell}(x,y)$.** 它决定了输入函数在不同位置之间如何耦合 (可以是全局、也可以被结构化为某种可快速计算的形式).  
- **Bias function $B_{\ell}(x)$.** 它是位置相关的偏置项.

> **Note:** Slides 在这里的表达非常克制：它只说 "Learning Parameters in $B_{\ell},K_{\ell}$". 这也意味着 kernel 的结构如何选，既是计算问题，也是 inductive bias 的选择问题.

**#3 离散实现为什么很贵 (复杂度从哪里来).**

假设我们把域 $D$ 用 $n$ 个离散点 $\{x_i\}_{i=1}^n$ 表示，并用求积把积分近似成求和. 那么对每个输出点 $x_i$，核积分会变成：

$$
\int_D K_{\ell}(x_i,y)v(y){\rm d}y \approx \sum_{j=1}^n K_{\ell}(x_i,x_j)v(x_j)\Delta x_j.
$$

这一步的朴素计算成本是：
- 对每个 $i$ 都要扫一遍 $j=1,\dots,n$，因此是 **$O(n^2)$ 次交互**.
- 如果 $K_{\ell}(x_i,x_j)$ 需要显式存储或显式计算，通常也会带来 **$O(n^2)$ 的内存或生成成本**.

因此，只要你的网格更细 (更大的 $n$)，这个 layer 的成本会迅速爆炸. 这就是 slides 提醒 "Caveat: Computational Complexity" 的原因.

<img src="../assets/solution3-fig1-复杂度警告.png" style="zoom:25%;" />

**#4 为什么会出现各种 kernel family (推断：它们在利用什么结构降复杂度).**

Slides 在 "Discrete Realization" 处只给了一个方向性列表 (Different Kernels ⇒ Low-Rank NOs，Graph NOs，Multipole NOs，...). 这背后可以理解为同一个目标：让离散实现从 $O(n^2)$ 降到更可用的量级.

<img src="../assets/solution3-fig2-kernel列表.png" style="zoom:25%;" />

下面是对这些 family 的简要解释 (推断，用于理解计算动机，不把它当作 slides 的原话)：

- **Low-rank kernels：**用低秩分解近似 $K(x,y)\approx \sum_{k=1}^r \phi_k(x)\psi_k(y)$，其中 $\phi_k(\cdot)$ 只依赖 $x$，$\psi_k(\cdot)$ 只依赖 $y$，$r$ 是秩/项数控制参数，从而把全连接的 $n\times n$ 交互压成 $O(nr)$.  
- **Fourier kernels / convolutional structure (FNO 的结构假设)：**FNO 额外增加平移不变条件 $K(x,y)=K(x-y)$. 在这个条件下，核积分变成卷积 $K\ast v$，可用 FFT 在 $O(n\log n)$ 计算.  
- **Graph kernels：**常见做法是把交互限制在图的邻域 (稀疏边集) 上 (例如 message passing)，从而把成本从 $O(n^2)$ 降到随边数增长的 $O(|E|)$.  
- **Multipole / hierarchical kernels：**常见思路是用分组/层次展开去压缩远距离交互，把计算从全局两两交互压到更接近线性复杂度.

> **Note:** 从 "operator layer 的定义" 到 "kernel family 的选择"，中间缺的那块其实就是一句话：**我们想要的算子结构 + 我们能承受的复杂度**. Kernel 参数化是在这两者之间做权衡的方式.
