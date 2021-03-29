import os
import sys
import numpy as np
import pandas as pd
import math
import random
import sklearn
from Test.PY.Lgc import  lgc


def read_filename(path, fileType):  # 读取文件夹的文件
    pathDir = os.listdir(path)
    file_path = []
    for dir in pathDir:
        if os.path.splitext(dir)[1] == fileType:  # 筛选txt文件
            child = os.path.join('%s%s' % (path, dir))
            file_path.append(child)
    return file_path


def data_pre(filepath):
    sheet = pd.read_table(filepath,sep="\t")
    ess_flag = sheet.iloc[:,3]
    data = sheet.iloc[:, 4:-1]
    data.fillna(0,inplace = True)
    data.loc[data['Nc'] == "*****", 'Nc'] = 0
    feature_id = data.columns
    feature_id = feature_id.values
    return data.values.astype(np.float64),ess_flag.values,feature_id


def data_balance(original_data, original_label):
    classes = np.unique(original_label)
    class_num = len(classes)
    least_class = 0
    least_class_num = 100000000
    for i in range(class_num):
        if list(original_label).count(classes[i]) < least_class_num:
            least_class = classes[i]
            least_class_num = list(original_label).count(classes[i])
    least_class_sample = original_data[np.where(original_label == least_class)]
    label = np.array(np.random.randint(2, 8, size=(np.shape(least_class_sample)[0], 1)))
    label[0:] = least_class
    sampled_data = least_class_sample.copy()
    for i in range(class_num):
        if classes[i] != least_class:
            sample = original_data[np.where(original_label == classes[i])]
            sample_loc = np.random.randint(np.shape(sample)[0], size=least_class_num)
            data = sample[sample_loc]
            label1 = np.array(np.random.randint(2, 8, size=(np.shape(data)[0], 1)))
            label1[0:] = classes[i]
            sampled_data = np.vstack((sampled_data, data))
            label = np.vstack((label, label1))
    balanced_data = np.hstack((sampled_data, label))
    np.random.shuffle(balanced_data)
    return balanced_data[:, 0:-1], balanced_data[:, -1].astype(np.float)
def evaluate(y_true, y_pre, y_pro):
    TP_e = 0
    FP_e = 0
    FN_e = 0
    TN_e = 0

    for i in range(np.shape(y_true)[0]):
        if (y_true[i] == 1) and (y_pre[i] == 1):
            TP_e = TP_e + 1
        elif (y_true[i] == 0) and (y_pre[i] == 0):
            TN_e = TN_e + 1
        elif (y_true[i] == 1) and (y_pre[i] == 0):
            FN_e = FN_e + 1
        elif (y_true[i] == 0) and (y_pre[i] == 1):
            FP_e = FP_e + 1
    print(TP_e,TN_e,FP_e,FN_e)
    precision = TP_e/(TP_e+FP_e)
    recall = TP_e/(TP_e+FN_e)
    ACC_e = (TP_e + TN_e) / (TP_e + TN_e + FP_e + FN_e)
    f = 2 * TP_e / (TP_e + TN_e + FP_e + FN_e + TP_e - TN_e)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_pro[:, 1])
    auc = sklearn.metrics.auc(fpr, tpr)
    print("ACC:" + str(ACC_e))
    print("TP:" + str(TP_e))
    print("FP:" + str(FP_e))
    print("TN:" + str(TN_e))
    print("FN:" + str(FN_e))
    print("AUC:" + str(auc))
    return TP_e, FP_e, TN_e, FN_e,precision,recall, ACC_e,f, auc
def get_tag(essential,nonessential,train_num):
    '''
    :param essential: essential genes
    :param nonessential: nonessential genes
    :return: esssnential/nonessential genes  with tag and without tag
    '''
    '''
    '''
    count_essential = np.shape(essential)[0]
    count_nonessential = np.shape(nonessential)[0]

    count_essenstial_tag = (count_essential*train_num)//10
    count_nonessential_tag = (count_nonessential*train_num)//10
    essential_tag_index = random.sample(range(0,count_essential),count_essenstial_tag)
    nonessential_tag_index = random.sample(range(0,count_nonessential),count_nonessential_tag)
    ess_no_tag = list(set(np.arange(count_essential))-set(essential_tag_index))
    non_no_tag = list(set(np.arange(count_nonessential))-set(nonessential_tag_index))
    essential_with_tag = essential[essential_tag_index]
    essential_with_tag_y = np.ones(len(essential_tag_index))
    nonessential_with_tag = nonessential[nonessential_tag_index]
    nonessential_with_tag_y = np.zeros(len(nonessential_tag_index))
    essential_without_tag = essential[ess_no_tag]
    essential_without_tag_y = np.ones(len(ess_no_tag))
    nonessential_without_tag = nonessential[non_no_tag]
    nonessential_without_tag_y = np.zeros(len(non_no_tag))
    return essential_with_tag, essential_with_tag_y, essential_without_tag, essential_without_tag_y, nonessential_with_tag, nonessential_with_tag_y, nonessential_without_tag, nonessential_without_tag_y

def pre_work():
    path = "../Data/"  #filepath
    type = ".txt"
    path_list = read_filename(path, type)
    ratio = [5]# The ratio of the labeled samples
    # ratio=[1,2,3,4,5,6,7,8,9] #This "ratio" is used to replace "ratio" in the previous row when performing experiments based on the proportions of different labeled data
    selectSigma_list = [10,0.1,0.5,1,5]# Sigma value of kernel function
    for selectSigma in selectSigma_list:
        '''
        Create a folder to save the results. The default is to save the results based on different kernel functions. If you need to do other experiments, you can change it as needed
        '''
        p="../different_kernel/"
        os.mkdir(p)
        p1="../different_kernel/Laplacian/"# create folder to save the results
        os.mkdir(p1)
        file = p1+"K=100%,laplacian,gamma ="+str(selectSigma)+", alpha = 0.999/"# Create a file to store the results of each experiment
        os.mkdir(file)
        for ratio_index in ratio:
            for item in path_list:
                txt = open(file+str(str(item).split("/")[-1].split(".")[0])+".txt", "w")
                (data_imblance, target, feature_id) = data_pre(item)
                a = np.where(data_imblance.astype(np.float64) == np.inf)
                data_imblance = np.delete(data_imblance, a[0], axis=0)
                target = np.delete(target, a[0], axis=0)
                for time in range(50):
                    #Data preprocessing
                    sample_data,sample_target = data_balance(data_imblance,target)
                    geneNums = np.shape(sample_data)[0]
                    sample_data = sklearn.preprocessing.MinMaxScaler((0, 1)).fit_transform(sample_data)
                    ess = sample_data[np.where(sample_target == 1)]
                    non = sample_data[np.where(sample_target == 0)]
                    train_ess, train_ess_y, test_ess, test_ess_y, train_non, train_non_y, test_non, test_non_y = get_tag(ess, non, ratio_index)  # 划分训练集和测试集
                    data_stack = np.vstack((test_ess,test_non))
                    target_stack = np.hstack((test_ess_y.transpose(),test_non_y.transpose()))
                    data_all_stack = np.hstack((data_stack,target_stack.reshape((np.shape(target_stack)[0],1))))
                    np.random.shuffle(data_all_stack)
                    test_y = data_all_stack[:,-1].astype(np.float64)
                    test_x = data_all_stack[:,0:-1]
                    #Predict essential genes
                    lp = lgc(selectSigma,train_ess,train_non,test_x,test_y,beta=0,kernel="Gauss",K=0)
                    result,result_true,degree = lp.label_propagation()
                    # Evaluate the results
                    TP, FP, TN, FN, p,r,ACC, f,AUC= evaluate(result_true, result, degree)
                    #Write results
                    txt.write(item  +"\t"+"TP:" + str(TP) + "\t" + "TN:" + str(TN) + "\t" + "FP:" + str(FP) + "\t" + "FN:" + str(FN) + "\t" + "precision:"+str(p)+"\t"+"recall:"+str(r)+"\t"+"ACC:" + str(ACC) + "\t" +"F:"+str(f)+"\t"+"AUC:" + str(AUC) + "\t")
                    txt.write("\n")


if __name__ == "__main__":
    pre_work()