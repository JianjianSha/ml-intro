# Fisher准则

以二分类为例，训练集为 $D=\{(\boldsymbol \phi_n,t_n):n=1,2,...,N\}, \ t_n \in \{C_1, C_2\}$，通过线性映射投影到一维空间 $y$，

\begin{equation}y=\mathbf w^{\top} \boldsymbol \phi\end{equation}


Fisher 准则定义如下区分性度量：

\begin{equation}L(\mathbf w)=\frac {(m_1-m_2)^2} {\sigma_1^2+\sigma_2^2} \end{equation}

上式中 $m_1, m_2$ 是 $C_1, C_2$ 的样本点在映射空间（这里是一维）的均值，$\sigma_1^2, \sigma_2^2$ 是映射空间的方差，（2）式表示类间距离越大，类内分散程度越小，则这两类区分性越强。

映射空间 $C_1$ 样本集合记为 $D_1$，数量记为 $N_1=|D_1|$，均值为

$$m_1=\frac 1 {N_1} \sum_{\boldsymbol \phi_i \in D_1} \mathbf w^{\top} \boldsymbol \phi_i=\mathbf w^{\top} \boldsymbol \mu_1$$

其中原始空间 $C_1$ 样本均值为

$$\boldsymbol \mu_1 = \frac 1 {N_1} \sum_{\boldsymbol \phi_i \in D_1} \boldsymbol \phi_i$$

于是

$$(m_2-m_1)^2=[\mathbf w^{\top}(\boldsymbol {\mu_2-\mu_1})]^2=\mathbf w^{\top} \mathbf S_{B} \mathbf w$$

其中

\begin{equation}\mathbf S_{B}=(\boldsymbol {\mu_2-\mu_1})(\boldsymbol {\mu_2-\mu_1})^{\top} \end{equation}

另一方面，映射空间 $C_1$ 样本方差为

$$\sigma_1^2=\frac 1 {N_1} \sum (y_i - m_1)^2=\frac 1 {N_1} \sum (\mathbf w^{\top}(\boldsymbol {\phi_i - \mu_1}))^2 = \frac 1 {N_1} \sum \mathbf w^{\top} (\boldsymbol {\phi_i-\mu_1})(\boldsymbol {\phi_i-\mu_1})^{\top} \mathbf w$$

于是

$$\sigma_1^2+\sigma_2^2=\mathbf w^{\top} \mathbf S_W \mathbf w$$

其中

\begin{equation}\mathbf S_W=\frac 1{N_1} \sum_{\boldsymbol \phi_i \in D_1} (\boldsymbol {\phi_i-\mu_1})(\boldsymbol {\phi_i-\mu_1})^{\top} + \frac 1 {N_2} \sum_{\boldsymbol \phi_i \in D_2} (\boldsymbol {\phi_i-\mu_2})(\boldsymbol {\phi_i-\mu_2})^{\top} \end{equation}

综上 （2）式为

\begin{equation}L(\mathbf w)=\frac {\mathbf w^{\top} \mathbf S_{B} \mathbf w} {\mathbf w^{\top} \mathbf S_W \mathbf w} \end{equation}

对（5）式中的 $\mathbf w$ 进行优化，取梯度并令其等于 0，得到

$$\nabla_{\mathbf w} L(\mathbf w)=\frac {\mathbf S_B \mathbf w(\mathbf w^{\top} \mathbf S_W \mathbf w)-(\mathbf w^{\top} \mathbf S_{B} \mathbf w)\mathbf S_W \mathbf w} {(\mathbf w^{\top} \mathbf S_W \mathbf w)^2}=0$$

整理得

$$\mathbf S_B \mathbf w(\mathbf w^{\top} \mathbf S_W \mathbf w)=(\mathbf w^{\top} \mathbf S_{B} \mathbf w)\mathbf S_W \mathbf w$$

注意到上式中括号部分为标量，且


\begin{aligned}\mathbf S_B \mathbf w &=\ (\boldsymbol {\mu_2-\mu_1})\{(\boldsymbol {\mu_2-\mu_1})^{\top} \mathbf w\}
\\ & \propto \ \boldsymbol {\mu_2-\mu_1}
\end{aligned}

（注：上式右侧大括号部分为标量）

所以

$$\mathbf S_W \mathbf w \propto \ \boldsymbol {\mu_2-\mu_1}$$

如果 $\mathbf S_W$ 满秩，则有

\begin{equation}\mathbf w \propto \ \mathbf S_W^{-1} (\boldsymbol {\mu_2-\mu_1}) \end{equation}

（6）式表示最具有区分性的 $\mathbf w$ 应与这两类样本中心的连线 $\boldsymbol {\mu_2-\mu_1}$ 大致同向，但应基于类内方差矩阵 $\mathbf S_W$ 进行调整。

:::{note}
在我们的直观理解上，判别超平面表示为 
```{math}
y=\mathbf w \boldsymbol \phi=0
```
例如 $y>0$ 表示 $C_1$ 类，$y<0$ 表示 $C_2$ 类，显然判别超平面大致垂直于这两类样本中心连线，而超平面的法线方向就是 $\mathbf w$，所以 $\mathbf w$ 大致与中心连线 $\boldsymbol {\mu_2-\mu_1}$ 同向。
:::