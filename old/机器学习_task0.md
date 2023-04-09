# 机器学习综述

## 什么是机器学习？什么又是深度学习？与AI有什么联系？

	机器学习是一门多领域交叉学科，涉及的理论知识是高等数学内容，而深度学习是机器学习的一个分支；而人工智能的学科是研究计算机怎样模拟或实现人类的学习行为，使得计算机具有人的思维方式，目前实现的方式是用机器学习的方式，而深度学习是实现智能化的机器学习中的一种方法。简单说就是人工智能是目标，机器学习是其中实现目的的一种方式。

## 机器学习的发展

	二十世纪五十年代开始，经历了早期研究阶段、蓬勃发展时期、统计学习、深度学习的阶段


## 一.机器学习分类
    
	机器学习一般可分为有监督学习、无监督学习、半监督学习和强化学习。

1.1 监督学习

    监督学习是指利用一组已知类别的样本调整分类器的参数，使其达到所要求性能的过程，也称为监督训练或有教师学习。在监督学习的过程中会提供对错指示，通过不断地重复训练，使其找到给定的训练数据集中的某种模式或规律，当新的数据到来时，可以根据这个函数预测结果。监督学习的训练集要求包括输入和输出，主要应用于分类和预测。
	
	在有监督学习中，目标变量可分为标称型和数值型，前者常与分类任务相关，后者常与回归任务相关。分类或者是回归，除了决策树、支持向量机等算法之外，也包含众多深度学习的算法和模型，如VGG16网络、残差神经网络ResNet等。

1.2 非监督学习

    与监督学习不同，在非监督学习中，无须对数据集进行标记，即没有输出。其需要从数据集中发现隐含的某种结构，从而获得样本数据的结构特征，判断哪些数据比较相似。因此，非监督学习目标不是告诉计算机怎么做，而是让它去学习怎样做事情。

	在无监督学习中，如果仅仅需要将数据划分为不同的离散的分组，属于聚类问题，而如果还需要判断数据与每个分组的相似程度，就属于密度估计问题。无监督学习中，常见的包括K-均值算法、最大期望算法、DBSCAN算法和Parzen窗设计等。

1.3 半监督学习

    半监督学习是监督学习和非监督学习的结合，其在训练阶段使用的是未标记的数据和已标记的数据，不仅要学习属性之间的结构关系，也要输出分类模型进行预测。

1.4强化学习

    强化学习（Reinforcement Learning, RL），又称再励学习、评价学习或增强学习，是机器学习的范式和方法论之一，用于描述和解决智能体（agent）在与环境的交互过程中通过学习策略以达成回报最大化或实现特定目标的问题. 

## 二.机器学习模型

	机器学习 = 数据（data） + 模型（model） + 优化方法（optimal strategy）

机器学习的算法导图[来源网络]

<img src="https://blog.griddynamics.com/content/images/2018/04/machinelearningalgorithms.png">


常见的机器学习算法

1. Linear Algorithms
   1. Linear Regression
   2. Lasso Regression 
   3. Ridge Regression
   4. Logistic Regression
2. Decision Tree
   1. ID3
   2. C4.5
   3. CART
3. SVM
4. Naive Bayes Algorithms
   1. Naive Bayes
   2. Gaussian Naive Bayes
   3. Multinomial Naive Bayes
   4. Bayesian Belief Network (BBN)
   5. Bayesian Network (BN)
5. kNN
6.  Clustering Algorithms
    1.  k-Means
    2.  k-Medians
    3.  Expectation Maximisation (EM)
    4.  Hierarchical Clustering

7.  K-Means
8.  Random Forest
9.  Dimensionality Reduction Algorithms
10. Gradient Boosting algorithms
    1.  GBM
    2.  XGBoost
    3.  LightGBM
    4.  CatBoost
11. Deep Learning Algorithms
    1.  Convolutional Neural Network (CNN)
    2.  Recurrent Neural Networks (RNNs)
    3.  Long Short-Term Memory Networks (LSTMs)
    4.  Stacked Auto-Encoders
    5.  Deep Boltzmann Machine (DBM)
    6.  Deep Belief Networks (DBN)


## 三.机器学习损失函数
1. 0-1损失函数


2. 绝对值损失函数

	L(y,f(x))=|y-f(x)|

3. 平方损失函数

    L(y,f(x))=(y-f(x))^2

4. log对数损失函数

	L(y,f(x))=log(1+e^{-yf(x)})

5. 指数损失函数

	L(y,f(x))=exp(-yf(x))

6. Hinge损失函数

	L(w,b)=max\{0,1-yf(x)\}


## 四.机器学习优化方法

梯度下降是最常用的优化方法之一，它使用梯度的反方向更新参数，使得目标函数达到最小化的一种优化方法，这种方法我们叫做梯度更新. 
1. (全量)梯度下降

2. 随机梯度下降

3. 小批量梯度下降

4. 引入动量的梯度下降

5. 自适应学习率的Adagrad算法

6. 牛顿法


## 五.机器学习的评价指标
1. MSE(Mean Squared Error)

2. MAE(Mean Absolute Error)

3. RMSE(Root Mean Squard Error)

4. Top-k准确率

5. 混淆矩阵

混淆矩阵|Predicted as Positive|Predicted as Negative
|:-:|:-:|:-:|
|Labeled as Positive|True Positive(TP)|False Negative(FN)|
|Labeled as Negative|False Positive(FP)|True Negative(TN)|

* 真正例(True Positive, TP):真实类别为正例, 预测类别为正例
* 假负例(False Negative, FN): 真实类别为正例, 预测类别为负例
* 假正例(False Positive, FP): 真实类别为负例, 预测类别为正例 
* 真负例(True Negative, TN): 真实类别为负例, 预测类别为负例

* 真正率(True Positive Rate, TPR): 被预测为正的正样本数 / 正样本实际数

* 假负率(False Negative Rate, FNR): 被预测为负的正样本数/正样本实际数

* 假正率(False Positive Rate, FPR): 被预测为正的负样本数/负样本实际数，

* 真负率(True Negative Rate, TNR): 被预测为负的负样本数/负样本实际数，

* 准确率(Accuracy)

* 精准率

* 召回率

* F1-Score

* ROC

ROC曲线的横轴为“假正例率”，纵轴为“真正例率”. 以FPR为横坐标，TPR为纵坐标，那么ROC曲线就是改变各种阈值后得到的所有坐标点 (FPR,TPR) 的连线，画出来如下。红线是随机乱猜情况下的ROC，曲线越靠左上角，分类器越佳. 


* AUC(Area Under Curve)

AUC就是ROC曲线下的面积. 真实情况下，由于数据是一个一个的，阈值被离散化，呈现的曲线便是锯齿状的，当然数据越多，阈值分的越细，”曲线”越光滑. 

<img src="https://gss3.bdstatic.com/-Po3dSag_xI4khGkpoWK1HF6hhy/baike/c0%3Dbaike80%2C5%2C5%2C80%2C26/sign=b9cb389a68d0f703f2bf9d8e69933a58/f11f3a292df5e0feaafde78c566034a85fdf7251.jpg">

用AUC判断分类器（预测模型）优劣的标准:

- AUC = 1 是完美分类器，采用这个预测模型时，存在至少一个阈值能得出完美预测。绝大多数预测的场合，不存在完美分类器.
- 0.5 < AUC < 1，优于随机猜测。这个分类器（模型）妥善设定阈值的话，能有预测价值.
- AUC < 0.5，比随机猜测还差；但只要总是反预测而行，就优于随机猜测.

## 六、机器学习模型选择

1. 交叉验证

所有数据分为三部分：训练集、交叉验证集和测试集。交叉验证集不仅在选择模型时有用，在超参数选择、正则项参数 [公式] 和评价模型中也很有用。

2. k-折叠交叉验证

- 假设训练集为S ，将训练集等分为k份:$\{S_1, S_2, ..., S_k\}$. 
- 然后每次从集合中拿出k-1份进行训练
- 利用集合中剩下的那一份来进行测试并计算损失值
- 最后得到k次测试得到的损失值，并选择平均损失值最小的模型

3. Bias与Variance，欠拟合与过拟合

**欠拟合**一般表示模型对数据的表现能力不足，通常是模型的复杂度不够，并且Bias高，训练集的损失值高，测试集的损失值也高.

**过拟合**一般表示模型对数据的表现能力过好，通常是模型的复杂度过高，并且Variance高，训练集的损失值低，测试集的损失值高.

<img src="https://pic3.zhimg.com/80/v2-e20cd1183ec930a3edc94b30274be29e_hd.jpg">

<img src="https://pic1.zhimg.com/80/v2-22287dec5b6205a5cd45cf6c24773aac_hd.jpg">

4. 解决方法

- 增加训练样本: 解决高Variance情况
- 减少特征维数: 解决高Variance情况
- 增加特征维数: 解决高Bias情况
- 增加模型复杂度: 解决高Bias情况
- 减小模型复杂度: 解决高Variance情况


## 七.机器学习参数调优

1. 网格搜索

一种调参手段；穷举搜索：在所有候选的参数选择中，通过循环遍历，尝试每一种可能性，表现最好的参数就是最终的结果

2. 随机搜索

与网格搜索相比，随机搜索并未尝试所有参数值，而是从指定的分布中采样固定数量的参数设置。它的理论依据是，如果随即样本点集足够大，那么也可以找到全局的最大或最小值，或它们的近似值。通过对搜索范围的随机取样，随机搜索一般会比网格搜索要快一些。

3. 贝叶斯优化算法

贝叶斯优化用于机器学习调参由J. Snoek(2012)提出，主要思想是，给定优化的目标函数(广义的函数，只需指定输入和输出即可，无需知道内部结构以及数学性质)，通过不断地添加样本点来更新目标函数的后验分布(高斯过程,直到后验分布基本贴合于真实分布。简单的说，就是考虑了上一次参数的信息，从而更好的调整当前的参数。


## 八.参考文献
1.  https://github.com/datawhalechina/team-learning/blob/master/%E5%88%9D%E7%BA%A7%E7%AE%97%E6%B3%95%E6%A2%B3%E7%90%86/Task1_ml_overvirew.md

