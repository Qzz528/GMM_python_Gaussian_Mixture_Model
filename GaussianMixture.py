# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

#根据原理编写的高斯混合模型
class GaussianMixtureModel(object):
    def __init__(self, n_components = 1):
        self.n_comps = n_components #聚类的类别数
    
    def _pdf(self, data, mu, cov):
        #data:(n_samps, n_feats) | mu:(n_feats,) | cov:(n_feats, n_feats)
        #概率密度函数，支持一维与高维
        assert len(data.shape) == 2
        
        data = data.T
        mu = mu.reshape(-1,1)
        k = 1/((2*np.pi)**(len(cov)/2)*np.sqrt(np.linalg.det(cov)))
        x_mu = (data-mu).T
        exp = np.exp(-0.5*np.sum(x_mu @ np.linalg.inv(cov) * x_mu, axis=1))
        return k*exp
    
    def fit(self, data, disp=False,max_iter = 200, tol = 1e-3):
        # data:(n_samps,n_feats)
        # max_iter求解的最大迭代次数，tol终止迭代的条件（迭代中似然函数变化小于tol认为训练完成）
        
        assert len(data.shape) == 2
        n_samps,n_feats = data.shape
        n_comps = self.n_comps
        
        ##设置参数，样本i:1~n_samps,种类k:1~n_comps,特征j:1~n_feats
        #Gamma:样本i属于k类的概率γ (n_samps,n_comps)
        Gamma = np.zeros((n_samps, n_comps)) 
        #Mu:k类均值μ (n_comps,n_feats) 
        Mu = np.zeros((n_comps,n_feats)) 
        for j in range(n_feats):#对第j维特征初始化
            count,interval = np.histogram(data[:,j], bins=n_comps*2)
            for k in range(n_comps):#预设k个类的中心
                #根据数据分布最高的n_comps个尖峰初始化均值
                index = np.argmax(count)
                Mu[k,j] = (interval[index]+interval[index+1])/2
                count[index] = 0 #最大值置0，下个循环挑选次最大值                       
        #Sigma:k类协方差∑ (n_comps,n_feats,n_feats)
        Sigma = np.array([np.eye(n_feats)]*n_comps)
        #Pi:k类权重π (n_comps,)
        Pi = np.ones(n_comps) / n_comps 
        
        #记录每次迭代的似然函数值，极大似然法是求其极值，当其值收敛时结束训练
        Likehood = [np.inf]
        for _ in range(max_iter):
            
            #E步 使用Mu,Sigma,Pi求Gamma
            for k in range(n_comps):#对每一成分k求所有样本i的Gamma
                Gamma[:,k] = Pi[k]*self._pdf(data, Mu[k], Sigma[k])
            #记录最新的似然函数值
            Likehood.append(sum(np.log(Gamma.sum(axis=1))))
            #按类别k归一化求出最终公式中Gamma
            Gamma = Gamma/Gamma.sum(axis=1).reshape(-1,1)
            #最新似然函数值与上次似然函数值的差，足够小提前结束训练
            if tol:
                if abs(Likehood[-1]-Likehood[-2])<tol:          
                    break
            
            #M步 使用Gamma更新Mu,Sigma,Pi
            for k in range(n_comps):#对每一成分k求其对应Mu,Sigma,Pi
                Sigma[k,:,:] = (Gamma[:,k].reshape(-1,1)*(data-Mu[k])).T@(data-Mu[k]) \
                               /Gamma.sum(axis=0)[k]
                Mu[k,:] = sum(Gamma[:,k].reshape(-1,1)*data)/Gamma.sum(axis=0)[k]
                Pi[k] = Gamma.sum(axis=0)[k]/n_samps
         

            #作图展示训练过程
            if disp:
                if n_feats == 1:
                    x = np.linspace(data.min(),data.max(), 1000).reshape(-1,1)
                    y = 0
                    for k in range(n_comps):#对每一成分k作图
                        y_k = self._pdf(x, Mu[k], Sigma[k])*Pi[k]
                        y += y_k
                        plt.plot(x,y_k, label = f'pdf-{k}', linewidth=3)
                    plt.hist(data, bins=15, density = True, label = 'hist')
                    plt.plot(x,y, '--',label = 'mix pdf', linewidth=3)
                    plt.legend()
                    plt.show()
                if n_feats == 2:
                    data_min = data.min(axis=0)
                    data_max = data.max(axis=0)
                    x = np.linspace(data_min[0],data_max[0], 100)
                    y = np.linspace(data_min[1],data_max[1], 100)
                    xx,yy = np.meshgrid(x,y)
                    xy = np.hstack((xx.reshape(-1,1),yy.reshape(-1,1)))
                    # z = 0
                    for k in range(n_comps):
                        z_k = self._pdf(xy, Mu[k], Sigma[k]).reshape(100,100)*Pi[k]
                        plt.contour(xx,yy,z_k)
                        # z += z_k
                    # plt.contour(xx,yy,z)
                    plt.scatter(data[:,0],data[:,1],label='data')
                    plt.legend()
                    plt.show()
                
        #存入类内变量，供类内其他函数使用
        self.Mu = Mu
        self.Sigma = Sigma
        self.Pi = Pi
        
        #返回训练数据属于各个类的概率
        return Gamma #(n_samps,n_comps)
    
    def set_params(self,Mu,Sigma,Pi): 
        #不通过数据训练，按指定参数创造混合高斯分布
        #Mu各成分均值(n_comps,n_feats)
        #Sigma各成分协方差矩阵(n_comps,n_feats,n_feats)
        #Pi各成分权重(n_comps,)
        assert len(Mu.shape) == 2
        n_comps,n_feats = Mu.shape
        assert n_comps == self.n_comps
        self.Mu = Mu
        
        assert len(Sigma.shape) == 3
        assert Sigma.shape[0] == n_comps
        assert Sigma.shape[1]==Sigma.shape[2]==n_feats
        self.Sigma = Sigma
        
        assert len(Pi.shape) == 1
        assert len(Pi)==n_comps
        assert sum(Pi)==1
        self.Pi = Pi
        
    def mix_pdf(self,data):
        #训练或给定的参数的混合高斯分布求其密度函数值
        y = 0
        for k in range(self.n_comps):#对每一成分
            y_k = self._pdf(data, self.Mu[k], self.Sigma[k])*self.Pi[k]
            y += y_k
        return y
    def mix_sample(self,n):
        #训练或给定的参数进行混合高斯分布进行采样
        #n: int，要进行采样的样本个数
        n_comps,n_feats = self.Mu.shape
        #对所有成分的协方差矩阵进行特征值分解
        Strech,Rotate = [],[]
        for k in range(n_comps):
            s,r = np.linalg.eig(self.Sigma[k])
            Strech.append(s)
            Rotate.append(r)
       
        #存放采样数据               
        samp_data = np.zeros((n,n_feats))
        #权重累加展开，0-1之间均匀分布落在哪个区间就选择哪个正态分布
        bound = np.array([0]+list(np.cumsum(self.Pi)))
        for i in range(n):#每个样本
            #以权重Pi为概率，决定选择某个正态分布成分
            choice = np.random.uniform(0,1)
            judge = np.array(bound) > choice
            k = int(np.argwhere(judge==0)[-1])
            #任意正态分布采样可通过各维独立标准正态分布采样
            #和协方差矩阵、均值向量的矩阵运算得到
            x = np.random.randn(1,n_feats)
            samp_data[i] = ((Strech[k]**0.5*Rotate[k]) @ x.T).T+ self.Mu[k]
        
        return samp_data #(n,n_feats)
            

    def predict(self,data,prob = False):
        #推理实际上是对新的样本，按照迭代后的Mu,Sigma,Pi计算Gamma
        assert len(data.shape) == 2
        n_samps,n_feats = data.shape
        
        Gamma = np.zeros((n_samps, self.n_comps))
        for k in range(self.n_comps):#对每一成分k求所有样本的Gamma
            Gamma[:,k] = self.Pi[k]*self._pdf(data, self.Mu[k], self.Sigma[k])
        #按类别k归一化求出最终公式中Gamma
        Gamma = Gamma/Gamma.sum(axis=1).reshape(-1,1)
        
        if prob:#输出各类概率
            return Gamma
        else:#输出所属类别
            return np.argmax(Gamma,axis=1)
        
        
if __name__ == '__main__':
    
    
    print("***聚类 1维情况***")
    ##鸢尾花数据集 分类任务 1维情况
    #使用无标签的数据训练，推理测试数据所属的类
    from sklearn.datasets import load_iris
    iris = load_iris()   
    #训练集 
    X_data = iris.data[:100,-1:] #(100,1)
    y_data = iris.target[:100] #(100,)
    #测试集（聚类，用训练集中的数据测试）
    X_test = iris.data[49:51,-1:] #(2,1)
    y_test = iris.target[49:51] #(2,)  
    
    print("真实结果：", y_test)    
    #自写方法
    gs = GaussianMixtureModel(n_components=2)
    gs.fit(X_data,disp=True)
    print("自写方法判断类别: ",gs.predict(X_test))
    print("自写方法各类概率: ",gs.predict(X_test,prob = True))   
    #scikit-learn方法
    from sklearn.mixture import GaussianMixture
    gr = GaussianMixture(n_components=2)
    gr.fit(X_data) #训练
    print("sklearn方法判断类别: ",gr.predict(X_test))
    print("sklearn方法各类概率: ",gr.predict_proba(X_test))
    #自写方法，按求取的混合高斯分布采样
    data = gs.mix_sample(10000)#采样
    plt.hist(data,density= True,bins=20,label="sample dist")#采样点分布图 
    #自写方法，按求取的混合高斯密度函数求值
    x = np.linspace(data.min(),data.max())
    y = gs.mix_pdf(x.reshape(-1,1))#求概率密度
    plt.plot(x,y,label = "pdf")#概率密度函数图
    plt.legend()
    plt.show()   
    

    
    
    
    
    
    
    
    print("***聚类 2维情况***")
    ##鸢尾花数据集 分类任务 高维情况
    #使用无标签的数据训练，推理测试数据所属的类  
    from sklearn.datasets import load_iris
    iris = load_iris()   
    #训练集 
    X_data = iris.data[:100,-2:] #(100,2)
    y_data = iris.target[:100] #(100,)
    #测试集（聚类，用训练集中的数据测试）
    X_test = iris.data[49:51,-2:] #(2,2)
    y_test = iris.target[49:51] #(2,)  
    
    print("真实结果：", y_test)
    #自写方法
    gs = GaussianMixtureModel(n_components=2)
    gs.fit(X_data,disp=True)
    print("自写方法判断类别: ",gs.predict(X_test))
    print("自写方法各类概率: ",gs.predict(X_test,prob = True))   
    #scikit-learn方法
    from sklearn.mixture import GaussianMixture
    gr = GaussianMixture(n_components=2)
    gr.fit(X_data) #训练
    print("sklearn方法判断类别: ",gr.predict(X_test))
    print("sklearn方法各类概率: ",gr.predict_proba(X_test))  
    #自写方法，按求取的混合高斯分布采样
    data = gs.mix_sample(10000)#采样
    plt.hist2d(data[:,0],data[:,1],density= True,bins=50,cmap = 'viridis')#采样点分布图
    plt.show()
    #自写方法，按求取的混合高斯密度函数求值
    x = np.linspace(data.min(axis=0)[0],data.max(axis=0)[0], 100)
    y = np.linspace(data.min(axis=0)[1],data.max(axis=0)[1], 100)
    xx,yy = np.meshgrid(x,y)#网格点
    xy = np.hstack((xx.reshape(-1,1),yy.reshape(-1,1)))#mix_pdf方法接收的数据格式为(n_samps,n_feats)
    z = gs.mix_pdf(xy).reshape(100,100)#求概率密度    
    plt.contourf(xx,yy,z,levels=50,cmap = 'viridis')#概率密度函数图
    plt.show()
    

    

    
    ##上例中聚类将数据分为两类，但哪一类是0哪一类是1无法判断
    ##sklearn库和自写方法有时会把两类的标签颠倒，属于正常情况