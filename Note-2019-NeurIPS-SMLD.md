    虽然做扩散模型已经有大半年时间了，但是今天在知乎上面看到了一篇paper，关于 Diffusion 的求解与直线之间的关系的，然后作者和一个人在评论区 battle，我觉得他们说得很精彩，但是我看不懂哈哈，深感自己的不足，所以开一个新的栏目，关于深入学习 DMs 和 FM 的技术原理和数学推导，栏目名字叫 'in-depth study diffusion', 在这个栏目里，我打算把我感兴趣的，也是比较里程碑式的一些在扩散模型领域的一些经典papers好好读一下，推导一下，但是考虑到我本职的研究工作，这个的更新时间不定，我希望一天或者两天能完成一篇。

Author：doomx
Time：2024.10.22 15:04

接下来我们开始今天的第一篇文章，来自 openai 的 Yang Song 的《Generative Modeling by Estimating Gradients of the Data Distribution》.

[[2019-NeurIPS-SMLD.pdf#page=1&selection=17,13,18,80&color=yellow|a new generative model where samples are produced via Langevin dynamics using gradients of the data distribution estimated with score matching.]]
这篇论文主要是在生成模型的采样方式上面下功夫，在未知数据分布的情况下，使用 Langevin dynamics 去在数据分布的 gradients 上进行 score matching。

这篇论文的主要贡献有以下几点：
   a. 提出了一种生成模型，通过估计数据分布的梯度来生成样本。
   b. 为了解决梯度在低维流形上可能难以估计的问题，提出了一种数据扰动方法，通过在不同噪声水平下估计梯度来提高梯度估计的准确性。
   c. 提出了一种退火 Langevin 动力学的采样方法，通过==逐渐降低噪声水平==来生成更接近数据流形的样本。
   d. 该框架允许灵活的模型架构，==不需要在训练过程中进行采样==，也不需要使用对抗方法。

## Score-Based generative modeling
在这篇论文中，作者提出了一种基于分数（score-based）的生成模型。模型的基本设定如下：

假设我们的数据集由来自未知数据分布 $p_{\text{data}}(x)$ 的独立同分布（i.i.d.）样本 $\{x_i \in \mathbb{R}^D\}^N_{i=1}$ 组成。我们定义一个概率密度函数 $p(x)$ 的梯度（score）为 $\nabla_x \log p(x)$。分数网络 $s_{\theta} : \mathbb{R}^D \rightarrow \mathbb{R}^D$ 是一个由 $\theta$ 参数化的神经网络，它将被训练以近似 $p_{\text{data}}(x)$ 的梯度。

生成模型的目标是使用数据集来学习一个模型，以便从 $p_{\text{data}}(x)$ 生成新的样本。基于梯度的生成模型框架包含两个要素：分数匹配（score matching）和 Langevin 动力学。
1. **分数匹配（Score Matching）**：分数匹配是一种训练方法，它通过最小化真实数据分布和模型分布之间的分数差异来训练模型。具体来说，我们希望分数网络 $s_{\theta}(x)$ 能够尽可能地接近真实数据分布 $p_{\text{data}}(x)$ 的分数。
2. **Langevin 动力学**：Langevin 动力学是一种采样方法，它通过模拟从高斯分布开始的随机过程，逐渐向数据分布 $p_{\text{data}}(x)$ 靠近，从而生成新的样本。这种方法可以看作是一种梯度下降过程，其中每一步都受到随机噪声的影响。

通过结合梯度匹配和 Langevin 动力学，基于梯度的生成模型能够在不需要显式建模数据分布的情况下，从数据中学习并生成新的样本。这种方法的一个关键优势是它可以直接在数据的梯度空间上工作，从而避免了传统生成模型中的一些问题，如 mode collapse 和对抗训练中的不稳定性。

### Score matching for score estimation 的数学原理
作者讨论了分数匹配（Score Matching）的原理，这是一种用于学习非归一化统计模型的方法，它基于来自未知数据分布的独立同分布（i.i.d.）样本。分数匹配的目标是直接训练一个分数网络 $s_{\theta}(x)$ 来估计数据分布的梯度 $\nabla_x \log p_{\text{data}}(x)$，而不需要先训练一个模型来估计 $p_{\text{data}}(x)$。

分数匹配的优化目标是最小化以下表达式：
$$\frac{1}{2} \mathbb{E}_{p_{\text{data}}} \left[ \left\| s_{\theta}(x) - \nabla_x \log p_{\text{data}}(x) \right\|^2_2 \right]$$
这个目标可以证明等价于以下表达式（忽略一个常数项）：
$$\mathbb{E}_{p_{\text{data}}} \left[ \text{tr} \left( \nabla_x s_{\theta}(x) \right) + \frac{1}{2} \left\| s_{\theta}(x) \right\|^2_2 \right]$$
其中，$\nabla_x s_{\theta}(x)$ 表示 $s_{\theta}(x)$ 的雅可比矩阵（Jacobian matrix）。

在某些正则性条件下，上述方程的最小化解（记作 $s_{\theta^*}(x)$）满足：
$$s_{\theta^*}(x) = \nabla_x \log p_{\text{data}}(x) \quad \text{almost surely}$$
在实践中，方程中的期望可以通过数据样本快速估计。然而，由于==需要计算雅可比矩阵的迹（trace）==，分数匹配并不适用于深度网络和高维数据。接下来，作者讨论了两种流行的分数匹配方法。这些方法旨在解决在深度网络和高维数据中计算雅可比矩阵迹的挑战。

### 解读 ''Denoising score matching Denoising''

==**Denoising Score Matching**== 是一种分数匹配的变体，它完全避免了计算雅可比矩阵的迹（$\text{tr}(\nabla_x s_{\theta}(x))$）。这种方法首先用一个预先指定的噪声分布 $q_{\sigma}(\tilde{x} | x)$ 对数据点 $x$ 进行扰动，然后使用分数匹配来估计扰动后数据分布 $q_{\sigma}(\tilde{x})$ 的梯度，其中 $q_{\sigma}(\tilde{x}) = \int q_{\sigma}(\tilde{x} | x) p_{\text{data}}(x) dx$。

Denoising Score Matching 的优化目标被证明等价于以下表达式：
$$\frac{1}{2} \mathbb{E}_{q_{\sigma}(\tilde{x}|x) p_{\text{data}}(x)} \left[ \left\| s_{\theta}(\tilde{x}) - \nabla_{\tilde{x}} \log q_{\sigma}(\tilde{x} | x) \right\|^2_2 \right]$$
这个表达式的意义是：在真实数据分布和条件分布的联合分布下，计算模型输出与条件分布梯度之间的平方欧几里得距离的期望，并乘以系数 $\frac{1}{2}$。这个目标函数旨在通过最小化这个距离，使得模型生成的数据与真实数据分布相似，同时保持生成数据的多样性。
如文献 [[2011-Denoise Score Matching.pdf]] 所示，最小化上述方程的最优分数网络（记作 $s_{\theta^*}(x)$）满足：
$$s_{\theta^*}(x) = \nabla_x \log q_{\sigma}(x) \quad \text{almost surely}$$
然而，$s_{\theta^*}(x) = \nabla_x \log q_{\sigma}(x) \approx \nabla_x \log p_{\text{data}}(x)$ 只有在噪声足够小，使得 $q_{\sigma}(x) \approx p_{\text{data}}(x)$ 时才成立。

这种方法的动机是，通过在数据点上添加噪声，我们可以更容易地估计梯度，因为噪声可以平滑数据分布，使得梯度更容易估计。此外，通过选择适当的噪声分布，我们可以控制扰动的程度，从而在估计梯度时获得更好的性能。

### 解读 ''Sliced score matching''
Sliced score matching [[2019-Sliced Score Matching.pdf]] 是一种用于生成模型的优化方法，它通过随机投影来近似梯度的迹（trace）操作，这是在原始的 score matching 方法中需要计算的。Sliced score matching 的目标函数如下：
$$\mathbb{E}_{p_v}[\mathbb{E}_{p_{\text{data}}} \left[ v^{\top} \nabla_{x} s_{\theta}(x) v + \frac{1}{2} \| s_{\theta}(x) \|^2_2 \right]]$$
这里，$p_v$ 是一个简单的随机向量分布，例如多变量标准正态分布。$\mathbb{E}_{p_v}$ 表示在随机向量分布 $p_v$ 下的期望，而 $\mathbb{E}_{p_{\text{data}}}$ 表示在真实数据分布 $p_{\text{data}}$ 下的期望。$v^{\top} \nabla_{x} s_{\theta}(x) v$ 是通过随机向量 $v$ 与模型 $s_{\theta}$ 的梯度的点积来近似梯度的迹。$\| s_{\theta}(x) \|^2_2$ 是模型输出的平方欧几里得范数。

Sliced score matching 的优势在于，它能够为原始未扰动的数据分布提供梯度估计，而不像 Denoise score matching 那样只能估计扰动数据的梯度。然而，由于需要使用前向模式自动微分来高效计算 $v^{\top} \nabla_{x} s_{\theta}(x) v$，这通常需要大约四倍于去噪 score matching 的计算量。

#### 说明 Denoise score matching 和 Sliced score matching 解决 score matching 中需要计算雅可比矩阵问题的思路。
Denoise Score Matching 它不是直接对原始数据点计算梯度，而是对添加了噪声的数据点计算梯度。这样，雅可比矩阵的计算就被简化了，因为噪声通常是高斯分布，其梯度很容易计算。这种方法的关键在于，通过最小化噪声扰动数据的梯度，可以间接地学习原始数据的分布。
Sliced Score Matching 它不是直接计算整个数据空间上的梯度，而是通过随机投影来近似这个迹。对于每个数据点，它会随机选择一个方向（通常是高斯分布的向量），然后在这个方向上计算梯度。优点是，它只需要计算一维的梯度，而不是整个数据空间上的梯度，从而大大减少了计算量。

### Langevin dynamics
Langevin dynamics 是一种从概率密度函数 $p(x)$ 生成样本的方法，它只需要知道该概率密度函数的梯度（即分数函数）$\nabla_x \log p(x)$。这种方法基于物理中的 Langevin 动力学，模拟了粒子在势场中的运动，其中势场由概率密度函数的对数定义。

Langevin dynamics 的采样过程如下：
1. **初始化**：选择一个初始值 $x_0$，它通常从一个先验分布 $\pi(x)$ 中采样得到。
2. **迭代更新**：在每一步 $t$，使用以下公式更新当前的样本 $x_t$：
$$x_{t+1} = x_t + \frac{\epsilon}{2} \nabla_x \log p(x_t) + \sqrt{\epsilon} z_t,$$
其中 $\epsilon > 0$ 是固定的步长，$z_t$ 是从标准正态分布 $N(0, I)$ 中采样的随机噪声。
3. **收敛**：当步长 $\epsilon$ 趋近于 0，迭代次数 $T$ 趋近于无穷大时，$x_T$ 的分布将收敛到 $p(x)$。在这种情况下，$x_T$ 可以被视为从 $p(x)$ 中精确采样得到的样本。
4. **Metropolis-Hastings 校正**：当 $\epsilon > 0$ 且 $T < \infty$ 时，由于迭代公式的近似性质，可能需要使用 Metropolis-Hastings 算法来校正误差。**但在实践中，当 $\epsilon$ 较小且 $T$ 较大时，这种误差通常可以忽略。**

在这项工作中，作者假设当 $\epsilon$ 较小且 $T$ 较大时，这种误差是可以忽略的。重要的是，从公式（4）中采样只需要分数函数 $\nabla_x \log p(x)$。因此，为了从数据分布 $p_{\text{data}}(x)$ 中获得样本，我们可以先训练一个分数网络，使得 $s_{\theta}(x) \approx \nabla_x \log p_{\text{data}}(x)$，然后使用 $s_{\theta}(x)$ 通过 Langevin 动力学近似地获得样本。==这是基于分数的生成模型框架的关键思想。==

### manifold hypothesis
Manifold Hypothesis 是机器学习的一个核心概念，它指出现实世界中的数据往往集中在高维空间中的低维流形上。

在 score-based generative modeling 的背景下，流形假设带来了两个主要挑战：

1. **梯度未定义问题**：由于分数（score）$\nabla_x \log p_{\text{data}}(x)$ 是在高维空间（ambient space）中计算的梯度，当数据点 $x$ 被限制在低维流形上时，这个梯度是未定义的。这是因为在流形上，数据的分布可能不是处处可微的，或者在某些方向上的变化是零（即流形的切空间之外的方向）。

2. **不一致的分数估计**：分数匹配目标（如方程（1）所示）仅当数据分布的支持是整个空间时才提供一致的分数估计器。然而，当数据分布在低维流形上时，这个目标将是不一致的。这意味着，即使模型能够完美地拟合数据的分数，它也无法保证在流形上的任何点都能生成与数据分布一致的样本。

为了说明流形假设对分数估计的负面影响，作者做了一个实验，其中使用 Sliced Score Matching 来训练一个 ResNet 网络来估计 CIFAR-10 数据集的 score。
![[Pasted image 20241022111243.png]]
实验结果表明，当直接在原始 CIFAR-10 图像上训练时，Sliced Score Matching 损失首先减少，然后不规则地波动。这表明，直接在数据上训练分数网络可能无法稳定地估计分数。相比之下，如果在数据上添加一个小的高斯噪声（使得扰动后的数据分布在整个 $\mathbb{R}^D$ 上有完整的支持），损失曲线将会收敛。这表明，通过在数据上添加噪声，可以使数据分布在整个空间上，从而使得分数估计更加稳定和一致。

#### Inaccurate score estimation with score matching
在低数据密度区域使用 score matching 进行分数估计的不准确性问题。以下是对这一部分内容的详细梳理:
1. **低数据密度区域的挑战**：
   - 在低数据密度区域，由于数据样本稀缺，分数匹配和基于 Langevin dynamics 的 MCMC 采样都可能面临困难。分数匹配在这些区域可能无法准确估计分数函数，因为缺乏足够的数据样本。
2. **分数匹配的基本原理**：
   - 分数匹配旨在最小化分数估计的期望平方误差，即 $\frac{1}{2} \mathbb{E}_{p_{\text{data}}}[\|s_{\theta}(x) - \nabla_x \log p_{\text{data}}(x)\|^2]$。
   - 在实践中，数据分布的期望总是通过 i.i.d. 样本 $\{x_i\}_{i=1}^N \sim p_{\text{data}}(x)$ 来估计。
3. **低数据密度区域的分数估计问题**：
   - 考虑任何区域 $R \subset \mathbb{R}^D$，使得 $p_{\text{data}}(R) \approx 0$。
   - 在大多数情况下，$\{x_i\}_{i=1}^N \cap R = \emptyset$，分数匹配将没有足够的数据样本来准确估计 $x \in R$ 时的 $\nabla_x \log p_{\text{data}}(x)$。
4. **实验结果**：
![[Pasted image 20241022145100.png|300]]
   - 作者提供了一个玩具实验的结果（详见附录 B.1），使用 sliced score matching 来估计高斯混合模型 $p_{\text{data}} = \frac{1}{5} \mathcal{N}((-5, -5), I) + \frac{4}{5} \mathcal{N}((5, 5), I)$ 的分数。
   - 实验结果表明，分数估计仅在 $p_{\text{data}}$ 的模式附近可靠，即数据密度高的区域。

### 解读 Slow mixing of Langevin dynamics
当数据分布具有多个模式（modes），并且这些模式之间被低数据密度区域分隔时，Langevin dynamics 在采样时会遇到的混合速度慢（slow mixing）问题。
1. **问题描述**：
   - 考虑一个混合分布 $p_{\text{data}}(x) = \pi p_1(x) + (1-\pi) p_2(x)$，其中 $p_1(x)$ 和 $p_2(x)$ 是具有不相交支持的归一化分布，$\pi \in (0, 1)$ 是混合权重。
   - 在 $p_1(x)$ 的支持内，梯度 $\nabla_x \log p_{\text{data}}(x) = \nabla_x \log p_1(x)$，而在 $p_2(x)$ 的支持内，梯度 $\nabla_x \log p_{\text{data}}(x) = \nabla_x \log p_2(x)$。
   - 由于在任一支持内，梯度 $\nabla_x \log p_{\text{data}}(x)$ 都不依赖于 $\pi$，Langevin dynamics使用这个梯度进行采样时，得到的样本不会依赖于 $\pi$。
2. **理论分析**：
   - 由于 Langevin dynamics 依赖于梯度 $\nabla_x \log p_{\text{data}}(x)$ 来从 $p_{\text{data}}(x)$ 中采样，而这个梯度并不反映混合权重 $\pi$，因此 Langevin dynamics 可能无法在合理的时间内正确恢复这两个模式的相对权重，因此可能无法收敛到真实的分布。
   - 在实践中，即使不同模式的支持大约不相交（它们可能共享相同的支持，但通过小数据密度区域连接），Langevin dynamics 理论上可以产生正确的样本，但可能需要非常小的步长和非常大的步数才能混合。
3. **实验验证**：
   - 为了验证这个分析，作者测试了与第 3.2.1 节中相同的高斯混合模型的 Langevin dynamics 采样，并在图 3 中提供了结果。
   - 使用 Langevin dynamics采样时，使用了真实的梯度。
   - 将图 3(b)与(a)比较，很明显，Langevin dynamics 得到的样本在两个模式之间的相对密度不正确，正如分析所预测的。
4. **结论**：
   - Langevin dynamics 在估计两个模式之间的相对权重时存在问题，而 annealed Langevin dynamics 能够更好地恢复相对权重。
   ![[Pasted image 20241022145214.png|450]]
   - 图 3(c)显示了使用 annealed Langevin dynamics 与真实梯度进行采样的结果，可以看出它能够正确估计两个模式之间的相对密度。

### 解读 NCSN
通过引入噪声条件分数网络（Noise Conditional Score Networks, NCSNs）来改进基于分数的生成模型的学习与推理过程。

1. 引入噪声的条件
   作者观察到通过在数据上添加随机高斯噪声，可以使数据分布更适合基于分数的生成建模。这样做有两个主要好处：
	- **避免流形假设问题**：由于高斯噪声的支撑是整个空间，扰动后的数据不会局限于低维流形，这避免了流形假设带来的问题，使得分数估计变得明确。
	- **填充低密度区域**：大的高斯噪声可以填充原始未扰动数据分布中的低密度区域，从而为分数匹配提供更多的训练信号，以改进分数估计。

2. 多噪声水平的分数估计
   通过使用多个噪声水平，可以获得一系列扰动数据分布，这些分布逐渐收敛到真实的数据分布。这为改进多模态分布上的 Langevin dynamics 的混合速率提供了可能，通过利用这些中间分布，可以借鉴模拟退火和退火重要性采样的思想。
   
3. 噪声条件分数网络（NCSN）
   作者提出了噪声条件分数网络（NCSN），它通过在不同噪声水平下估计分数来迭代参数。NCSN 旨在联合估计所有扰动数据分布的分数，即对于所有噪声水平 $\sigma \in \{\sigma_i\}_{i=1}^L$，有 $s_{\theta}(x, \sigma) \approx \nabla_x \log q_{\sigma}(x)$。
   
4. 通过分数匹配学习 NCSN
   无论是 Sliced score matching 还是 Denoise score matching 都可以训练 NCSN。作者选择了 Denoise score matching，因为它快一些，并且自然适合估计扰动数据分布的分数。给定噪声分布 $q_{\sigma}(\tilde{x} | x) = N(\tilde{x} | x, \sigma^2 I)$，去噪分数匹配的目标函数为：
$$\mathcal{L}(\theta; \sigma) = \frac{1}{2} \mathbb{E}_{p_{\text{data}}(x)}\mathbb{E}_{\tilde{x} \sim N(x, \sigma^2 I)} \left[ \left\| s_{\theta}(\tilde{x}, \sigma) + \frac{\tilde{x} - x}{\sigma^2} \right\|^2_2 \right]$$
   然后，作者将所有 $\sigma \in \{\sigma_i\}_{i=1}^L$ 的目标函数组合起来，得到统一的目标函数：
$$\mathcal{L}(\theta; \{\sigma_i\}_{i=1}^L) = \sum_{i=1}^L \lambda(\sigma_i) \mathcal{L}(\theta; \sigma_i)$$
   其中 $\lambda(\sigma_i) > 0$ 是依赖于 $\sigma_i$ 的系数函数。作者选择了 $\lambda(\sigma) = \sigma^2$，以确保不同 $\sigma$ 下的目标函数的量级大致相同。

作者强调，他们的目标函数不需要对抗训练、替代损失，也不需要在训练过程中从分数网络中采样。它不要求 $s_{\theta}(x, \sigma)$ 具有特殊的架构以便可处理。当 $\lambda(\cdot)$ 和 $\{\sigma_i\}_{i=1}^L$ 固定时，它可以用来定量比较不同的 NCSN。

#### 解读推理时候的 annealed Langevin dynamics
在推理阶段，作者提出了一种称为 annealed Langevin dynamics 的采样方法，该方法受到模拟退火（simulated annealing）和退火重要性采样（annealed importance sampling）的启发。以下是对这种方法的详细解读：

1. **初始化**：
   - 从某个固定的先验分布（例如均匀噪声）初始化样本。

2. **逐步采样**：
   - 使用 Langevin dynamics 从 $q_{\sigma_1}(x)$ 采样，步长为 $\alpha_1$。
   - 从上一步的最终样本开始，使用减小的步长 $\alpha_2$ 运行 Langevin dynamics 以从 $q_{\sigma_2}(x)$ 采样。
   - 以这种方式继续，使用 $q_{\sigma_{i-1}}(x)$ 的 Langevin dynamics 的最终样本作为 $q_{\sigma_i}(x)$ 的 Langevin dynamics 的初始样本，并逐渐减小步长 $\alpha_i$。

3. **最终采样**：
   - 最后，运行 Langevin dynamics 以从 $q_{\sigma_L}(x)$ 采样，当 $\sigma_L \approx 0$ 时，这接近于 $p_{\text{data}}(x)$。

4. **避免流形假设问题**：
   - 由于所有分布 $\{q_{\sigma_i}\}_{i=1}^L$ 都通过高斯噪声扰动，它们的支撑覆盖整个空间，它们的分数（梯度）是明确定义的，避免了流形假设带来的困难。

5. **改进分数估计和混合速率**：
   - 当 $\sigma_1$ 足够大时，$q_{\sigma_1}(x)$ 的低密度区域变得较小，模式之间的隔离度降低。这可以使分数估计更准确，Langevin dynamics 的混合速度更快。
   - 因此，可以假设 Langevin dynamics 为 $q_{\sigma_1}(x)$ 生成良好的样本。这些样本很可能来自 $q_{\sigma_1}(x)$ 的高密度区域，这意味着它们也很可能位于 $q_{\sigma_2}(x)$ 的高密度区域，考虑到 $q_{\sigma_1}(x)$ 和 $q_{\sigma_2}(x)$ 之间的差异很小。
   - 由于分数估计和 Langevin dynamics在高密度区域表现更好，来自 $q_{\sigma_1}(x)$ 的样本将为 $q_{\sigma_2}(x)$ 的 Langevin dynamics提供良好的初始样本。类似地，$q_{\sigma_{i-1}}(x)$ 为 $q_{\sigma_i}(x)$ 提供良好的初始样本，最终从 $q_{\sigma_L}(x)$ 获得高质量的样本。

6. **调整步长**：
   - 在算法 1 中，有许多可能的方法可以根据 $\sigma_i$ 调整 $\alpha_i$。作者选择 $\alpha_i \propto \sigma_i^2$。
   - 动机是在 Langevin 动力学中固定“信噪比” $\alpha_i s_{\theta}(x, \sigma_i)^2 \sqrt{\alpha_i} z$ 的大小。注意，$E[\| \alpha_i s_{\theta}(x, \sigma_i) \sqrt{2\sqrt{\alpha_i} z} \|_2^2] \approx E[\alpha_i \|s_{\theta}(x, \sigma_i)\|_2^2 \cdot 4] \propto \frac{1}{4} E[\| \sigma_i s_{\theta}(x, \sigma_i) \|_2^2]$。
   - 回顾一下，当分数网络接近最优时，我们发现 $\|s_{\theta}(x, \sigma)\|_2 \propto \frac{1}{\sigma}$，此时 $E[\| \sigma_i s_{\theta}(x; \sigma_i) \|_2^2] \propto 1$。因此，$\| \alpha_i s_{\theta}(x, \sigma_i) \sqrt{2\sqrt{\alpha_i} z} \|_2 \propto \frac{1}{4} E[\| \sigma_i s_{\theta}(x, \sigma_i) \|_2^2] \propto \frac{1}{4}$ 不依赖于 $\sigma_i$。

7. **实验验证**：
   - 为了证明 annealed Langevin dynamics 的有效性，作者提供了一个玩具示例，目标是仅使用分数从具有两个明显分离模式的高斯混合分布中采样。
   - 在实验中，作者选择 $\{\sigma_i\}_{i=1}^L$ 为一个几何级数，其中 $L = 10$，$\sigma_1 = 10$，$\sigma_{10} = 0.1$。
   - 结果如图 3 所示。将图 3(b)与(c)比较，可以看出 annealed Langevin dynamics 正确地恢复了两个模式之间的相对权重，而标准 Langevin dynamics 则失败了。

8. **总结**：
   - Annealed Langevin dynamics 通过从高噪声水平开始，逐步降低噪声水平，从而在每个阶段生成更接近数据流形的样本。
   - 这种方法利用了不同噪声水平下的数据分布，使得分数估计更准确，Langevin dynamics 的混合速度更快。
   - 通过逐步调整噪声水平和步长，annealed Langevin dynamics 能够生成高质量的样本，同时避免了流形假设带来的问题。
