

#### 1.代码运行图示例

数据为鸢尾花前两类的第四特征或三四号特征。

求解一维混合高斯分布最终结果图 （柱状图为数据分布，曲线为所求混合高斯分布密度函数及各成分密度函数） 

![image](/pic/train-1d.png)

求得的一维高斯分布概率密度曲线，以及其采样的分布 

![image](/pic/samp&pdf-1d.png)

求解二维混合高斯分布最终结果图 （散点图为二维数据，曲线为所求混合高斯分布各成分密度函数等高线图）

![image](/pic/train-2d.png)

求得的一维高斯分布概率密度热力图，以及其采样的分布 

![image](/pic/pdf-2d.png)

![image](/pic/samp-2d.png)

#### 2.原理简介

##### 2.1正态分布（高斯分布）

d维正态分布概率密度函数：

$$
N(X|\mu,\Sigma) = \frac{1}{(2\pi)^\frac{d}{2}|\Sigma|^\frac{1}{2}}e^{-\frac{1}{2}(X-\mu)^T\Sigma^{-1}(X-\mu)}N(X|\mu,\Sigma) = \frac{1}{(2\pi)^\frac{d}{2}|\Sigma|^\frac{1}{2}}e^{-\frac{1}{2}(X-\mu)^T\Sigma^{-1}(X-\mu)}
$$
其中，X是d维向量，$\mu$是d维的均值，$$\Sigma$$是d*d的协方差矩阵。

正态分布的密度函数值（概率）与按其分布采样的样本分布（频率）有着对应关系。

![image](/pic/samp&pdf.png)

##### 2.2混合高斯分布

定义一个将$$K$$个正态分布加权求和形成的分布，各个正态分布其概率密度函数：
$$
M(X|\pi,\mu,\Sigma) = \sum_{k=1}^K\pi_kN(X|\mu_k,\Sigma_k)
$$
其中$$N(X|\mu_k,\Sigma_k)$$是2.1中前述的正态分布概率密度函数，$$\pi_k$$是每个正态分布成分的权重，应满足约束条件$$\sum_{k=1}^K\pi_k=1$$
$$\pi$$表示$$\pi_k, \ k=1,2,...,K$$整体。$$\mu,\Sigma$$同理。

混合多个正态分布成分的目的是用多个正态分布去近似一个复杂的分布，如上例，两个正态分布可以构造出一个双峰的分布。

![image](/pic/samp&pdf-1d.png)

按上图，混合高斯分布的概率密度值（概率）也与按混合方式对正态分布采样的样本分布有着对应关系。
混合方式采样时$$\pi_k$$为，选择按正态分布$$N(X|\mu_k,\Sigma_k)$$进行采样的概率。

##### 2.3正态分布（高斯分布）参数估计

2.1中，已知正态分布的参数$$\mu,\Sigma$$可获得概率密度函数$$N(X|\mu,\Sigma)$$，但实际往往是根据一系列样本数据（n个样本）$$X_1,X_2,...,Xn$$来反向推算（估计）正态分布参数$$\mu,\Sigma$$。
从结果上来看，$$\mu$$的估计为这组数据的均值，$$\Sigma$$的估计为这组数据的协方差矩阵。

估计的原理为极大似然估计，此时一系列数据$$X_1,X_2,...,Xn$$为已知，相当于要挑选适当的$$\mu,\Sigma$$，使得此$$\mu,\Sigma$$参数值的正态分布下，取到一系列样本数据$$X_1,X_2,...,X_n$$（目前的既定结果）的概率最大，即最大化$$\mathcal{P}(\mu,\Sigma) = \prod_{i=1}^nN(X_i|\mu,\Sigma)$$


求取$$\mathcal{P}(\mu,\Sigma)$$极值时的$$\mu,\Sigma$$值（即极值点）

可转变为求$$\ln\mathcal{P}$$的极值点，

进而转变为求$$\ln \mathcal{P}$$的对$$\mu,\Sigma$$偏导数的零点。

即方程
$$
\frac{\part \ln\mathcal{P}}{\part \mu}=0,\ \frac{\part\ln \mathcal{P}}{\part \Sigma}=0
$$

的解。


对于d维数据$$X_i$$，求解结果为：
$$
\mu = (\mu_1,\mu_2,...,\mu_d)\\
\mu_j=\frac{1}{n}\sum_{i=1}^nX_{i,j}.\\
j=1,2,...,d.\\
$$

均值为d维，$$X_{i,j}$$表示第i个样本第j维度的数值.$$\mu_j$$表示所有样本第j维度的均值。

$$
\Sigma = [a_{r,c}]\\
a_{r,c}=\frac{1}{n}\sum_{i=1}^n (X_{i,r}-\mu_r)(X_{i,c}-\mu_c)\\
r=1,2,...,d.\\
c=1,2,...,d.
$$

协方差矩阵为d*d矩阵，$$a_{r,c}$$表示协方差矩阵第r行第c列的值。
$$X_{i,r}$$表示第i个样本第r维度的数值，$$\mu_r$$表示所有样本第r维度的均值。当r为c时同理。




##### 2.4混合高斯分布

同2.3，2.2中高斯混合分布的$$K$$个成分$$N(X|\mu_k,\Sigma_k)$$和权重$$\pi_k$$为已知。

目前要根据数据$$X_1,X_2,...,X_n$$估计各个高斯成分的参数$$\mu_k,\Sigma_k$$和该高斯成分的权重$$\pi_k, \ \ k=1,2,...,K$$.

混合高斯分布与正态分布的参数估计思路同理，对于已有的样本数据$$X_1,X_2,...,X_n$$。参数为$$\pi,\mu,\Sigma$$或者说参数为$$\pi_k, \mu_k,\Sigma_k,\ k=1,2,...K.$$的混合高斯分布，取到全部样本值$$X_1,X_2,...,X_n$$的概率为：
$$
\mathcal{P}=\prod_{i=1}^N M(X_i|\pi,\mu,\Sigma) = \prod_{i=1}^N \sum_{k=1}^K\pi_kN(X_i|\mu_k,\Sigma_k)
$$

仍然是极大似然估计的思想，将$$\pi_k, \mu_k,\Sigma_k,\ k=1,2,...K.$$视为参数。样本值$$X_1,X_2,...,X_n$$视为已知。求使得
$$\mathcal{P}$$值最大的$$\pi_k, \mu_k,\Sigma_k,\ k=1,2,...K.$$

同2.3，也即是求$$\mathcal{P}$$的极值点，转变为求$$\ln\mathcal{P}$$极值点。

化简有$$\ln\mathcal{P}=\sum_{i=1}^N\ln \sum_{k=1}^K\pi_kN(X_i|\mu_k,\Sigma_k)$$

此处因为$$\pi_k$$是权重，需要满足约束条件$$\sum_{k=1}^K\pi_k=1$$。根据拉格朗日乘子法，将有约束的优化问题$$\ln\mathcal{P}$$变为无约束$$\mathcal{F} = \ln\mathcal{P} + \lambda(\sum_{k=1}^K\pi_k-1)$$。

此处求$$\ln\mathcal{P}$$极值点问题，

转变为求$$\mathcal{F}$$极值点，

同2.1，转变为求$$\mathcal{F}$$偏导数零点问题。即：

根据方程
$$
\frac{\part \mathcal{F}}{\part \mu_k} = 0\ ,\  \frac{\part \mathcal{F}}{\part \Sigma_k}=0\ ,\ \frac{\part \mathcal{F}}{\part \pi_k}=0\\
k=1,2,...,K.
$$
求解$$\pi_k, \mu_k,\Sigma_k,\ k=1,2,...K.$$


##### 2.5 EM迭代

在单一高斯分布问题中（如2.3），求解偏导方程时问题已经结束，但2.4中的偏导等式求解复杂，难以直接求解。
将三组等式
$$
\frac{\part \mathcal{F}}{\part \mu_k} = 0\ ,\  \frac{\part \mathcal{F}}{\part \Sigma_k}=0\ ,\ \frac{\part \mathcal{F}}{\part \pi_k}=0\\
k=1,2,...,K.
$$
中化简结果中一个经常出现的项定义为$$\gamma$$，有：
$$
\gamma_{i,k} = \frac{\pi_kN(X_i|\mu_k,\Sigma_k)}{\sum_{s=1}^K\pi_sN(X_i|\mu_s,\Sigma_s)}
$$

每个样本和每个高斯成分对应一个$$\gamma$$，可以认为$$\gamma_{i,k}$$是第i个样本属于第k类高斯成分的概率。

通过$$\gamma_{i,k}$$代换，三组原始等式
$$
\frac{\part \mathcal{F}}{\part \mu_k} = 0\ ,\  \frac{\part \mathcal{F}}{\part \Sigma_k}=0\ ,\ \frac{\part \mathcal{F}}{\part \pi_k}=0.
$$
变形为
$$
\mu_k = \frac{\sum_{i=1}^{n}\gamma_{i,k}X_i}{\sum_{i=1}^{n}\gamma_{i,k}}\\
\Sigma_k= \frac{\sum_{i=1}^{n}\gamma_{i,k}(X_i-\mu_k)^T(X_i-\mu_k)}{\sum_{i=1}^{n}\gamma_{i,k}}\\
\pi_k = \frac{\sum_{i=1}^{n}\gamma_{i,k}}{n}
$$

新的三组等式只是原始三组偏导等于0等式的变形。没有引入新的任何条件，但我们可据此进行EM算法迭代。

EM算法迭代方式为

1）先预设一组值$$\pi_k, \mu_k,\Sigma_k. \ \ k=1,2,...K.$$
2）利用预设的$$\pi_k, \mu_k,\Sigma_k$$求出所有样本对应所有类别的概率：
$$
\gamma_{i,k} = \frac{\pi_kN(X_i|\mu_k,\Sigma_k)}{\sum_{s=1}^K\pi_sN(X_i|\mu_s,\Sigma_s)}\\
i=1,2,...n.\\ k=1,2,..,K.
$$

3）利用2）中求得的$$\gamma_{i,k},\ \ i=1,2,...n.\ \ k=1,2,..,K.$$求（或者说更新）$$\pi_k, \mu_k,\Sigma_k$$：
$$
\mu_k = \frac{\sum_{i=1}^{n}\gamma_{i,k}X_i}{\sum_{i=1}^{n}\gamma_{i,k}}\\
\Sigma_k= \frac{\sum_{i=1}^{n}\gamma_{i,k}(X_i-\mu_k)^T(X_i-\mu_k)}{\sum_{i=1}^{n}\gamma_{i,k}}\\
\pi_k = \frac{\sum_{i=1}^{n}\gamma_{i,k}}{n}
$$

之后重复2）3）步，直到达到规定迭代次数或者迭代中$$\mathcal{P}$$收敛。


##### 2.6 高斯混合模型聚类算法

高斯混合模型的一个重要用途就是聚类。而且2.4，2.5中已经表述了模型训练及推理的方法。

在已知样本$$X_1,X_2,...,X_n$$时，可通过2.4，2.5的方法获取高斯混合模型的参数$$\pi_k, \mu_k,\Sigma_k, \ \ k=1,2,...K.$$这就是训练过程。

当训练完成，对任意样本$$X_i$$（这里的$$i$$可以是在$$1,2,...,n$$范围中的训练数据，也可以是训练集外的新数据）属于$$K$$类高斯成分中的哪一个，可由2.4中定义的
$$
\gamma_{i,k} = \frac{\pi_kN(X_i|\mu_k,\Sigma_k)}{\sum_{s=1}^K\pi_sN(X_i|\mu_s,\Sigma_s)}\\
k=1,2,..,K.
$$
给出。（所需参数$$\pi_k, \mu_k,\Sigma_k, \ \ k=1,2,...K.$$已通过训练得到）

对单个样本$$X_i$$，$$\gamma_{i,1},\gamma_{i,2},...,\gamma_{i,K}$$这K个值表示了该样本属于各个高斯成分（或者说K类）的概率。可以借由概率最大值对应的类，完成类别判断（聚类）。
