#encoding:utf-8
import sys
from numpy import *
#from svm import *
from os import listdir
import numpy as np

def clipAlpha(a_j,H,L):
    """
        clipAlpha(调整aj的值，使aj处于 L<=aj<=H)
        Args:
        aj  目标值
        H   最大值
        L   最小值
        Returns:
        aj  目标值
    """
    if a_j > H:
        a_j = H
    if L > a_j:
        a_j = L
    return a_j

def selectJrand(i,m):
    """
        随机选择一个整数
        Args:
        i  第一个alpha的下标
        m  所有alpha的数目
        Returns:
        j  返回一个不为i的随机数，在0~m之间的整数值
    """
    j = i
    while j == i:
        j = int(random.uniform(0,m))
    return j


class PlattSMO:
    def __init__(self,dataMat,classlabels,C,toler,maxIter,**kernelargs):
        """
            Args:
            dataMatIn    数据集
            classLabels  类别标签
            C   松弛变量(常量值)，允许有些数据点可以处于分隔面的错误一侧。
            控制最大化间隔和保证大部分的函数间隔小于1.0这两个目标的权重。
            可以通过调节该参数达到不同的结果。
            toler   容错率
            kTup    包含核函数信息的元组
        """

        self.x = array(dataMat)
        self.label = array(classlabels).transpose()
        self.C = C
        self.toler = toler
        self.maxIter = maxIter
        # 数据的行数
        self.m = shape(dataMat)[0]
        self.n = shape(dataMat)[1]
        self.alpha = array(zeros(self.m),dtype='float64')
        self.b = 0.0
        # 误差缓存，第一列给出的是eCache是否有效的标志位，第二列给出的是实际的E值。
        self.eCache = array(zeros((self.m,2)))
        # m行m列的矩阵
        self.K = zeros((self.m,self.m),dtype='float64')
        self.kwargs = kernelargs
        self.SV = ()
        self.SVIndex = None
        for i in range(self.m):
            if i%100==0:
                print i
            for j in range(self.m):
                self.K[i,j] = self.kernelTrans(self.x[i,:],self.x[j,:])

    def calcEK(self,k):
        """calcEk（求 Ek误差：预测值-真实值的差）
            该过程在完整版的SMO算法中陪出现次数较多，因此将其单独作为一个方法
            Args:
            oS  optStruct对象
            k   具体的某一行
            Returns:
            Ek  预测结果与真实结果比对，计算误差Ek
        """
        fxk = dot(self.alpha*self.label,self.K[:,k])+self.b
        Ek = fxk - float(self.label[k])
        return Ek
    
    def updateEK(self,k):
        """updateEk（计算误差值并存入缓存中。）
            在对alpha值进行优化之后会用到这个值。
            Args:
            oS  optStruct对象
            k   某一列的行号
        """
        # 求 误差：预测值-真实值的差
        Ek = self.calcEK(k)
        self.eCache[k] = [1 ,Ek]
    
    def selectJ(self,i,Ei):
        """selectJ（返回最优的j和Ej）
            内循环的启发式方法。
            选择第二个(内循环)alpha的alpha值
            这里的目标是选择合适的第二个alpha值以保证每次优化中采用最大步长。
            该函数的误差与第一个alpha值Ei和下标i有关。
            Args:
            i   具体的第i一行
            oS  optStruct对象
            Ei  预测结果与真实结果比对，计算误差Ei
            Returns:
            j  随机选出的第j一行
            Ej 预测结果与真实结果比对，计算误差Ej
        """
        maxE = 0.0
        selectJ = 0
        Ej = 0.0
        validECacheList = nonzero(self.eCache[:,0])[0]
        if len(validECacheList) > 1:
            for k in validECacheList:
                #print k,i,validECacheList.shape
                if k == i:continue
                Ek = self.calcEK(k)
                deltaE = abs(Ei-Ek)
                if deltaE > maxE:
                    selectJ = k
                    maxE = deltaE
                    Ej = Ek
            return selectJ,Ej
        else:
            selectJ = selectJrand(i,self.m)
            Ej = self.calcEK(selectJ)
            return selectJ,Ej

    def innerL(self,i):
        """innerL
        内循环代码
        Args:
            i   具体的某一行
            oS  optStruct对象
        Returns:
            0   找不到最优的值
            1   找到了最优的值，并且oS.Cache到缓存中
        """
        # 求 Ek误差：预测值-真实值的差
        Ei = self.calcEK(i)
        # 约束条件 (KKT条件是解决最优化问题的时用到的一种方法。我们这里提到的最优化问题通常是指对于给定的某一函数，求其在指定作用域上的全局最小值)
        # 0<=alphas[i]<=C，但由于0和C是边界值，我们无法进行优化，因为需要增加一个alphas和降低一个alphas。
        # 表示发生错误的概率：labelMat[i]*Ei 如果超出了 toler， 才需要优化。至于正负号，我们考虑绝对值就对了。
        '''
            # 检验训练样本(xi, yi)是否满足KKT条件
            yi*f(i) >= 1 and alpha = 0 (outside the boundary)
            yi*f(i) == 1 and 0<alpha< C (on the boundary)
            yi*f(i) <= 1 and alpha = C (between the boundary)
        '''
        if (self.label[i] * Ei < -self.toler and self.alpha[i] < self.C) or \
                (self.label[i] * Ei > self.toler and self.alpha[i] > 0):
            self.updateEK(i)
            # 选择最大的误差对应的j进行优化。效果更明显
            j,Ej = self.selectJ(i,Ei)
            alphaIOld = self.alpha[i].copy()
            alphaJOld = self.alpha[j].copy()
            # L和H用于将alphas[j]调整到0-C之间。如果L==H，就不做任何改变，直接return 0
            if self.label[i] != self.label[j]:
                L = max(0,self.alpha[j]-self.alpha[i])
                H = min(self.C,self.C + self.alpha[j]-self.alpha[i])
            else:
                L = max(0,self.alpha[j]+self.alpha[i] - self.C)
                H = min(self.C,self.alpha[i]+self.alpha[j])
            if L == H:
                return 0
            # eta是alphas[j]的最优修改量，如果eta==0，需要退出for循环的当前迭代过程
            # 参考《统计学习方法》李航-P125~P128<序列最小最优化算法>
            eta = 2*self.K[i,j] - self.K[i,i] - self.K[j,j]
            if eta >= 0:
                return 0
            # 计算出一个新的alphas[j]值
            self.alpha[j] -= self.label[j]*(Ei-Ej)/eta
            # 并使用辅助函数，以及L和H对其进行调整
            self.alpha[j] = clipAlpha(self.alpha[j],H,L)
            # 更新误差缓存
            self.updateEK(j)
            # 检查alpha[j]是否只是轻微的改变，如果是的话，就退出for循环。
            if abs(alphaJOld-self.alpha[j]) < 0.00001:
                return 0
            # 然后alphas[i]和alphas[j]同样进行改变，虽然改变的大小一样，但是改变的方向正好相反
            self.alpha[i] +=  self.label[i]*self.label[j]*(alphaJOld-self.alpha[j])
            # 更新误差缓存
            self.updateEK(i)
            
            # 在对alpha[i], alpha[j] 进行优化之后，给这两个alpha值设置一个常数b。
            # w= Σ[1~n] ai*yi*xi => b = yi- Σ[1~n] ai*yi(xi*xj)
            # 所以：  b1 - b = (y1-y) - Σ[1~n] yi*(a1-a)*(xi*x1)
            # 为什么减2遍？ 因为是 减去Σ[1~n]，正好2个变量i和j，所以减2遍
            b1 = self.b - Ei - self.label[i] * self.K[i, i] * (self.alpha[i] - alphaIOld) - \
                 self.label[j] * self.K[i, j] * (self.alpha[j] - alphaJOld)
            b2 = self.b - Ej - self.label[i] * self.K[i, j] * (self.alpha[i] - alphaIOld) - \
                 self.label[j] * self.K[j, j] * (self.alpha[j] - alphaJOld)
            if 0<self.alpha[i] and self.alpha[i] < self.C:
                self.b = b1
            elif 0 < self.alpha[j] and self.alpha[j] < self.C:
                self.b = b2
            else:
                self.b = (b1 + b2) /2.0
            return 1
        else:
            return 0

    def smoP(self):
        """
        完整SMO算法外循环，与smoSimple有些类似，但这里的循环退出条件更多一些
        Args:
            dataMatIn    数据集
            classLabels  类别标签
            C   松弛变量(常量值)，允许有些数据点可以处于分隔面的错误一侧。
            控制最大化间隔和保证大部分的函数间隔小于1.0这两个目标的权重。
            可以通过调节该参数达到不同的结果。
            toler   容错率
            maxIter 退出前最大的循环次数
            kTup    包含核函数信息的元组
        Returns:
            b       模型的常量值
            alphas  拉格朗日乘子
        """
        iter = 0
        entrySet = True
        alphaPairChanged = 0
        while iter < self.maxIter and ((alphaPairChanged > 0) or (entrySet)):
            print "here3"
            alphaPairChanged = 0
            if entrySet:
                for i in range(self.m):
                    alphaPairChanged+=self.innerL(i)
                iter += 1
            else:
                nonBounds = nonzero((self.alpha > 0)*(self.alpha < self.C))[0]
                for i in nonBounds:
                    alphaPairChanged+=self.innerL(i)
                iter+=1
            if entrySet:
                entrySet = False
            elif alphaPairChanged == 0:
                entrySet = True
            print "iteration number: %d" % iter
        self.SVIndex = nonzero(self.alpha)[0]
        self.SV = self.x[self.SVIndex]
        self.SVAlpha = self.alpha[self.SVIndex]
        self.SVLabel = self.label[self.SVIndex]
        self.x = None
        self.K = None
        self.label = None
        self.alpha = None
        self.eCache = None
#   def K(self,i,j):
#       return self.x[i,:]*self.x[j,:].T
    def kernelTrans(self,x,z):
        """
            核转换函数
            Args:
            X     dataMatIn数据集
            A     dataMatIn数据集的第i行的数据
            kTup  核函数的信息
            Returns:
        """
        if array(x).ndim != 1 or array(x).ndim != 1:
            raise Exception("input vector is not 1 dim")
        if self.kwargs['name'] == 'linear':
            return sum(x*z)
        elif self.kwargs['name'] == 'rbf':
            theta = self.kwargs['theta']
            return exp(sum((x-z)*(x-z))/(-1*theta**2)) # 径向基函数的高斯版本

    def calcw(self):
        for i in range(self.m):
            self.w += dot(self.alpha[i]*self.label[i],self.x[i,:])

    def predict(self,testData):
        test = array(testData)
        #return (test * self.w + self.b).getA()
        result = []
        m = shape(test)[0]
        for i in range(m):
            tmp = self.b
            for j in range(len(self.SVIndex)):
                tmp += self.SVAlpha[j] * self.SVLabel[j] * self.kernelTrans(self.SV[j],test[i,:])
            while tmp == 0:
                tmp = random.uniform(-1,1)
            if tmp > 0:
                tmp = 1
            else:
                tmp = -1
            result.append(tmp)
        return result

def plotBestfit(data,label,w,b):
    import matplotlib.pyplot as plt
    n = shape(data)[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    for i in range(n):
        if int(label[i]) == 1:
            x1.append(data[i][0])
            y1.append(data[i][1])
        else:
            x2.append(data[i][0])
            y2.append(data[i][1])
    ax.scatter(x1,y1,s=10,c='red',marker='s')
    ax.scatter(x2,y2, s=10, c='green', marker='s')
    x = arange(-2,10,0.1)
    y = ((-b-w[0]*x)/w[1])
    plt.plot(x,y)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()

def loadImage(dir,maps = None):
    dirList = listdir(dir)
    data = []
    label = []
    for file in dirList:
        label.append(file.split('_')[0])
        lines = open(dir +'/'+file).readlines()
        row = len(lines)
        col = len(lines[0].strip())
        line = []
        for i in range(row):
            for j in range(col):
                line.append(float(lines[i][j]))
        data.append(line)
        if maps != None:
            label[-1] = float(maps[label[-1]])
        else:
            label[-1] = float(label[-1])
    return array(data),array(label)

def loadDataSet(filename):
    fr = open(filename)
    data = []
    label = []
    for line in fr.readlines():
        lineAttr = line.strip().split('\t')
        data.append([float(x) for x in lineAttr[:-1]])
        label.append(float(lineAttr[-1]))
    return data,label

def main():
    data,label = loadDataSet('test_num_5000.txt')
    data = mat(data)
    print data.max(axis=0).shape
    data = data / data.max(axis=0)
    print "here1"
    '''
    dataMatIn    数据集
    classLabels  类别标签
    C   松弛变量(常量值)，允许有些数据点可以处于分隔面的错误一侧。
    控制最大化间隔和保证大部分的函数间隔小于1.0这两个目标的权重。
    可以通过调节该参数达到不同的结果。
    toler   容错率
    kTup    包含核函数信息的元组
    '''
    smo = PlattSMO(data,label,200,0.0001,10000,name = 'rbf',theta = 2.0)
    # smo = PlattSMO(data,label,200,0.0001,10000,name = 'rbf',theta = 1.3)
    print "here2"
    smo.smoP()
    print len(smo.SVIndex)
    
    print smo.b
    print smo.SVAlpha.shape
    print smo.SVAlpha

    np.savetxt('model_b.csv',mat(smo.b),fmt='%f',delimiter=',')
    np.savetxt('model_alpha.csv',mat(smo.SVAlpha),fmt='%f',delimiter=',')

    test,testLabel = loadDataSet('train_num_15000.txt')
    test = mat(test)
    test = test / test.max(axis=0)
    testResult = smo.predict(test)
    m = shape(test)[0]
    tp  = 0.0
    fp = 0.0
    fn = 0.0
    for i in range(m):
        if testLabel[i] > 0 and testResult[i] > 0:
            tp += 1
        if testLabel[i] < 0 and testResult[i] > 0:
            fp += 1
        if testLabel[i] > 0 and testResult[i] < 0:
            fn += 1
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    f1_score = 2*precision*recall/(precision+recall)
    print tp,fp,fn
    print "precision is: ",precision
    print "recall is: ",recall
    print "f1 score is: ",f1_score

    #smo.kernelTrans(data,smo.SV[0])
    #smo.calcw()
    #print smo.predict(data)
    '''
    maps = {'1':1.0,'9':-1.0}
    data,label = loadImage("digits/trainingDigits",maps)
    smo = PlattSMO(data, label, 200, 0.0001, 10000, name='rbf', theta=20)
    smo.smoP()
    print len(smo.SVIndex)
    test,testLabel = loadImage("digits/testDigits",maps)
    testResult = smo.predict(test)
    m = shape(test)[0]
    count  = 0.0
    for i in range(m):
        if testLabel[i] != testResult[i]:
            count += 1
    print "classfied error rate is:",count / m
    #smo.kernelTrans(data,smo.SV[0])
    '''
if __name__ == "__main__":
    sys.exit(main())



