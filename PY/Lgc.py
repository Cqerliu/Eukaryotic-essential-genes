from sklearn import preprocessing
from sklearn.semi_supervised import LabelSpreading
import numpy as np
import math
class lgc:
    def __init__(self,sigma,X_labeled_pos,X_labeled_neg,X_unlabeled,Y_unlabeled,beta,kernel="Laplacian",K=-10,alpha = 0.999):
        '''
        :param sigma:
        :param X_labeled: labeled samples
        :param X_unlabeled: unlabeled samples
        :param Y_labeled: Label of labeled samples
        :param Y_unlabeled: True label of unlabeled samples
        :param kernel: The selection function
        :param K: The K value of KNN
        '''
        self.sigma = sigma
        self.X_labeled_pos = X_labeled_pos
        self.X_labeled_neg = X_labeled_neg
        self.X_unlabeled = X_unlabeled
        self.Y_unlabeled = Y_unlabeled
        self.kernel = kernel
        self.K = -K
        self.alpha = alpha
        self.beta = beta



    def calSimilarity(self,vecA,vecB):
        '''
        :param vecA:
        :param vecB:
        :return: 返回向量相似性
        '''
        dist = np.sqrt(np.sum(np.square(vecA - vecB)))
        dot = np.dot(vecA,vecB)
        if self.kernel == "Laplacian":
            similarity = math.exp(-(dist/(self.sigma)))
            return similarity
        elif self.kernel=="Gauss":
            similarity = math.exp(-((dist*dist) / (self.sigma)))
            return similarity
        else:
            raise NameError ("核函数选择错误")

    def similarityMatrix(self,matrix,labelMatrix):
        '''
        :return: 返回S
        '''
        m, n = np.shape(matrix)
        affinity_matrix = np.array(np.zeros((m, m)))
        diagonal_matrix = np.array(np.zeros((m, m)))
        for i in range(m):
            for j in range(i+1,m):
                s = self.calSimilarity(matrix[i],matrix[j])
                affinity_matrix[i,j] = s
                affinity_matrix[j,i] = s
            affinity_matrix[i,i] = 0
        for i in range(m):
            index = np.argsort(affinity_matrix[i])[self.K]
            value = affinity_matrix[i][index]
            affinity_matrix[i][affinity_matrix[i]<value]=0
        for i in range(m):
            coe = np.sqrt(np.sum(affinity_matrix[i, :]))
            if coe!=0:
                diagonal_matrix[i, i] = 1 / np.sqrt(np.sum(affinity_matrix[i, :]))
            else:
                raise ValueError("error")
        S = np.dot(np.dot(diagonal_matrix, affinity_matrix), diagonal_matrix)
        return S
    def getSpecialLabel(self):
        m1, n1 = np.shape(self.X_labeled_pos)
        m2,n2 = np.shape(self.X_labeled_neg)
        m0, n0 = np.shape(self.X_unlabeled)
        label_Y = np.array(np.zeros((m0 + m1+m2, 3)))
        label_Y[0:m1, 1] = 1
        label_Y[m1:m1+m2, 0] = 1
        label_Y[m1+m2:, 2] = self.Y_unlabeled
        label_Y[0:m1,2] = -1
        label_Y[m1:m1+m2,2] = -2
        return label_Y
    def getWholeMatrix(self,Y_matrix):
        matrix0 = np.vstack((self.X_labeled_pos, self.X_labeled_neg))
        matrix1 = np.vstack((matrix0, self.X_unlabeled))
        complete_matrix = np.hstack((matrix1, Y_matrix))
        np.random.shuffle(complete_matrix)
        return complete_matrix
    def deal_F_default(self,Y_true,Y_pre):
        s1, s2 = np.shape(Y_pre)
        for find in range(s1):
            if Y_pre[find][0] == 0 and Y_pre[find][1] == 0:
                Y_pre[find][0] = 0.1
                Y_pre[find][1] = 0.1
            else:
                pass
        sum = np.sum(Y_pre, axis=1)
        sum = 1 / sum
        total = np.c_[sum, sum]
        Y_pre = Y_pre * total
        result_pre = Y_pre.copy()
        result_pre[result_pre >= 0.5] = 1
        result_pre[result_pre < 0.5] = 0
        result_pre = result_pre[np.where(Y_true >= 0)]
        result_true = Y_true[np.where(Y_true >= 0)]
        return result_pre,result_true,Y_pre[np.where(Y_true>=0)]
    def label_propagation(self):
        '''
        LGC
        '''
        Y = self.getSpecialLabel()
        whole_matrix = self.getWholeMatrix(Y)
        matrix_x = whole_matrix[:, 0:-3]
        S = self.similarityMatrix(matrix_x,whole_matrix[:,-1])
        beta = self.alpha / (1 + self.alpha)
        miu = 1 / (1 + self.alpha)
        I = np.identity(np.shape(whole_matrix)[0])
        F_begin = whole_matrix[:,-3:-1]
        F = beta * (np.dot(np.linalg.inv((I - miu * S)), F_begin))
        true_Y = whole_matrix[:,-1]
        pre,true,pro= self.deal_F_default(true_Y,F)
        return pre[:, 1], true,pro









