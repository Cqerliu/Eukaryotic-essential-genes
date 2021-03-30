import os
import pandas as pd
import rpy2.robjects as robjects
import numpy as np
from rpy2.robjects.packages import importr
from collections import Counter
from scipy.fftpack import fft
import math
from sampen import sampen2
import gc
def read_filename(path,fileType):#读取文件夹的文件
    pathDir = os.listdir(path)
    file_path = []
    for dir in pathDir:
        if os.path.splitext(dir)[1] == fileType: #筛选txt文件
            child = os.path.join('%s%s' % (path,dir))
            file_path.append(child)
    return file_path
def cal_feature(file):#计算蛋白质特征
    for filepath in file:
        content = pd.read_table(filepath, skiprows=[0], sep="\t",header=None)
        gene = content.iloc[:, 2].values
        id = content.iloc[:, 0].values
        protein = content.iloc[:, 1].values
        ess_tag = content.iloc[:, 3].values
        txt = open("../protein_feature/" + filepath.split("/")[-1].split(".")[0] + ".txt", "w")
        col_name = ["ID","pro","seq","A", "R", "D", "C", "Q", "E", "G", "H", "I", "N", "L", "K", "M", "F", "P","S", "T", "W", "Y", "V", "rare_aa_ratio", "close_aa_ratio","label" ]
        for item in col_name:
            txt.write(item + "\t")
        for num in range(len(protein)):
            txt.write("\n")
            A = protein[num].count("A")
            R = protein[num].count("R")
            D = protein[num].count("D")
            C = protein[num].count("C")
            Q = protein[num].count("Q")
            E = protein[num].count("E")
            G = protein[num].count("G")
            H = protein[num].count("H")
            I = protein[num].count("I")
            N = protein[num].count("N")
            L = protein[num].count("L")
            K = protein[num].count("K")
            M = protein[num].count("M")
            F = protein[num].count("F")
            P = protein[num].count("P")
            S = protein[num].count("S")
            T = protein[num].count("T")
            W = protein[num].count("W")
            Y = protein[num].count("Y")
            V = protein[num].count("V")
            Rare_aa_ratio = (H+M+W)/len(protein[num])
            close_aa_ratio = (C+W+Y)/len(protein[num])
            feature = [id[num], protein[num], gene[num],A/len(protein[num]),R/len(protein[num]),D/len(protein[num]),C/len(protein[num]),Q/len(protein[num]),E/len(protein[num]),G/len(protein[num]),H/len(protein[num]),I/len(protein[num]),N/len(protein[num]),L/len(protein[num]),K/len(protein[num]),M/len(protein[num]),F/len(protein[num]),P/len(protein[num]),S/len(protein[num]),T/len(protein[num]),W/len(protein[num]),Y/len(protein[num]),V/len(protein[num]),Rare_aa_ratio,close_aa_ratio,ess_tag[num]]
            for item in feature:
                txt.write(str(item) + "\t")
def cal_nucleotide_fre(file):
    for filepath in file:
        print(filepath)
        content = pd.read_table(filepath, skiprows=[0], sep="\t",header=None)
        gene = content.iloc[:, 2].values
        id = content.iloc[:, 0].values
        protein = content.iloc[:, 1].values
        ess_tag = content.iloc[:, 3].values
        txt = open("../nucleotide_feature/" + filepath.split("/")[-1].split(".")[0] + ".txt", "w")
        col_name =  ["ID", "pro", "seq","A1","A2","A3","C1","C2","C3","U1","U2","U3","G1","G2","G3","label"]
        for item in col_name:
            txt.write(item + "\t")
        for num in range(len(protein)):
            txt.write("\n")
            A = protein[num].count("A")
            R = protein[num].count("R")
            D = protein[num].count("D")
            C = protein[num].count("C")
            Q = protein[num].count("Q")
            E = protein[num].count("E")
            G = protein[num].count("G")
            H = protein[num].count("H")
            I = protein[num].count("I")
            N = protein[num].count("N")
            L = protein[num].count("L")
            K = protein[num].count("K")
            M = protein[num].count("M")
            F = protein[num].count("F")
            P = protein[num].count("P")
            S = protein[num].count("S")
            T = protein[num].count("T")
            W = protein[num].count("W")
            Y = protein[num].count("Y")
            V = protein[num].count("V")
            pro_len = len(protein[num])
            A1 =  (I+M+T+N+K+S+R)/pro_len
            A2 = (Y+H+Q+N+K+D+E)/pro_len
            A3 = (L+S+P+Q+R+I+T+K+V+A+E+G)/pro_len
            C1 = (L+P+H+R+Q)/pro_len
            C2 = (S+P+T+A)/pro_len
            C3 = (F+S+Y+C+L+P+H+R+I+T+N+S+V+A+D+G)/pro_len
            U1 = sum([F,L,S,Y,C,W])/pro_len
            U2 = sum([F,L,I,M,V])/pro_len
            U3 = sum([ F,S,Y,C,L,P,H,R,I,T,N,S,V,A,D,G])/pro_len
            G1 = sum([V,A,D,E,G])/pro_len
            G2 = sum([C,W,R,S,G])/pro_len
            G3 = sum([L,S,W,P,Q,M,T,K,R,V,A,E,G])/pro_len
            feature = [id[num],protein[num],gene[num],A1,A2,A3,C1,C2,C3,U1,U2,U3,G1,G2,G3,ess_tag[num]]
            for item in feature:
                txt.write(str(item)+"\t")
def cal_inter_nucleotide_features(file):
    for filepath in file:
        print(filepath)
        content = pd.read_table(filepath, skiprows=[0], sep="\t",header=None)
        gene = content.iloc[:, 2].values
        id = content.iloc[:, 0].values
        protein = content.iloc[:, 1].values
        ess_tag = content.iloc[:, 3].values
        txt = open("../inter_nucleotide_feature/" + filepath.split("/")[-1].split(".")[0] + ".txt", "w")
        col_name = ["ID", "pro", "seq", "Fa","Aa", "Da","Ma", "ma","Fg","Ag", "Dg", "Mg", "mg","Fc", "Ac","Dc", "Mc","mc","Ft", "At",  "Dt",  "Mt",  "mt","label"]
        for item in col_name:
            txt.write(str(item)+"\t")
        for num in range(len(gene)):
            print(id[num])
            txt.write("\n")
            feature = [id[num],protein[num],gene[num]]
            seq = list(gene[num])
            length = len(seq)
            indexA = [i for i, x in enumerate(seq) if x == 'A' or x=="a"]
            indexG = [i for i, x in enumerate(seq) if x == 'G' or x == "g"]
            indexC = [i for i, x in enumerate(seq) if x == 'C' or x == "c"]
            indexT = [i for i, x in enumerate(seq) if x == 'T' or x == "t"]
            index_arr = [indexA,indexG,indexC,indexT]
            distance = []
            distance_order = []
            print(index_arr)
            for i in range(4):
                dis = []
                for j in range(len(index_arr[i])-1):
                    dis.append(index_arr[i][j+1]-index_arr[i][j])
                # dis.append(len(seq)-1-index_arr[i][-1])
                distance.append(dis)#未排序真实距离
                distance_order.append(sorted(dis))#升序排序真实距离
            for i in range(4):
                item = distance_order[i]
                feature.append((len(index_arr[i]))/length)
                feature.append(np.mean(item))
                feature.append(np.var(item))
                if len(item)==0:
                    feature.append(0)
                else:
                    occur_time = Counter(item)  # 统计核苷酸间距离数组每个元素出现的次数
                    print(occur_time)
                    mode = max(occur_time.values())
                    mode_item = []
                    for (key, value) in occur_time.items():
                        if value == mode:
                            mode_item.append(key)  # 众数对应的距离值
                    feature.append(min(mode_item))
                feature.append(np.median(item))
            feature.append(ess_tag[num])
            for item in feature:
                txt.write(str(item)+"\t")

def cal_inter_amino_features(file):
    for filepath in file:
        print(filepath)
        content = pd.read_table(filepath, skiprows=[0], sep="\t",header=None)
        gene = content.iloc[:, 2].values
        id = content.iloc[:, 0].values
        protein = content.iloc[:, 1].values
        ess_tag = content.iloc[:, 3].values
        txt = open("../Scale_entropy2/" + filepath.split("/")[-1].split(".")[0] + ".txt", "w")
        amino_name = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]
        col_name = ["ID", "pro", "seq"]
        for item in amino_name:
            col_name.append("F"+item)
            col_name.append("A" + item)
            col_name.append("D" + item)
            col_name.append("M" + item)
            col_name.append("m" + item)
        col_name.append("label")
        for item in col_name:
            txt.write(str(item)+"\t")
        for num in range(len(protein)):
            print(id[num])
            txt.write("\n")
            feature = [id[num],protein[num],gene[num]]
            seq = list(protein[num].upper())
            length = len(seq)
            amino = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]
            index_arr = []
            for item in amino:
                index =  [i for i, x in enumerate(seq) if x == item]
                index_arr.append(index)
            distance = []
            distance_order = []
            print(index_arr)
            for i in range(20):
                dis = []
                for j in range(len(index_arr[i])-1):
                    dis.append(index_arr[i][j+1]-index_arr[i][j])
                # dis.append(len(seq)-1-index_arr[i][-1])
                distance.append(dis)#未排序真实距离
                distance_order.append(sorted(dis))#升序排序真实距离
            for i in range(20):
                item = distance_order[i]
                feature.append((len(index_arr[i]))/length)
                feature.append(np.mean(item))
                feature.append(np.var(item))
                if len(item)==0:
                    feature.append(0)
                else:
                    occur_time = Counter(item)  # 统计核苷酸间距离数组每个元素出现的次数
                    print(occur_time)
                    mode = max(occur_time.values())
                    mode_item = []
                    for (key, value) in occur_time.items():
                        if value == mode:
                            mode_item.append(key)  # 众数对应的距离值
                    feature.append(min(mode_item))
                feature.append(np.median(item))
            feature.append(ess_tag[num])
            for item in feature:
                txt.write(str(item)+"\t")



def cal_CPS_features(file):
    for filepath in file:
        print(filepath)
        content = pd.read_table(filepath, skiprows=[0], sep="\t", header=None)
        gene = content.iloc[:, 2].values
        id = content.iloc[:, 0].values
        protein = content.iloc[:, 1].values
        ess_tag = content.iloc[:, 3].values
        txt = open("../CPS_feature/" + filepath.split("/")[-1].split(".")[0] + ".txt", "w")
        col_name = ["ID", "pro", "seq","m1a", "m2a", "m1g", "m2g", "m1c", "m2c", "m1t", "m2t", "cm1a", "cm2a", "cm1g", "cm2g", "cm1c", "cm2c", "cm1t", "cm2t","label"]
        for item in col_name:
            txt.write(str(item)+"\t")
        gene_count = len(gene)
        for num in range(gene_count):
            txt.write("\n")
            na = 0
            nc = 0
            ng = 0
            nt = 0
            ua = []
            ug = []
            uc = []
            ut = []
            seq = gene[num]
            seq_len = len(seq)
            for i in range(seq_len):
                if seq[i]=="A" or seq[i]=="a":
                    ua.append(1)
                    ug.append(0)
                    uc.append(0)
                    ut.append(0)
                    na += 1
                elif seq[i]=="G" or seq[i]=="g":
                    ua.append(0)
                    ug.append(1)
                    uc.append(0)
                    ut.append(0)
                    ng += 1
                elif seq[i] == "C" or seq[i] == "c":
                    ua.append(0)
                    ug.append(0)
                    uc.append(1)
                    ut.append(0)
                    nc += 1
                elif seq[i] == "T" or seq[i] == "t":
                    ua.append(0)
                    ug.append(0)
                    uc.append(0)
                    ut.append(1)
                    nt += 1
                else:
                    ua.append(0)
                    ug.append(0)
                    uc.append(0)
                    ut.append(0)

            UA = fft(ua)#傅里叶变换
            UG = fft(ug)
            UC = fft(uc)
            UT = fft(ut)
            PSA = [0 for _ in range(seq_len)]
            PSG =[0 for _ in range(seq_len)]
            PSC = [0 for _ in range(seq_len)]
            PST = [0 for _ in range(seq_len)]
            for i in range(seq_len):
                PSA[i] = pow(abs(UA[i]),2)
                PSG[i] = pow(abs(UG[i]), 2)
                PSC[i] = pow(abs(UC[i]), 2)
                PST[i] = pow(abs(UT[i]), 2)
            CPSA = [0 for _ in range(seq_len)]#累计傅里叶功率谱
            CPSG = [0 for _ in range(seq_len)]
            CPSC = [0 for _ in range(seq_len)]
            CPST = [0 for _ in range(seq_len)]
            for  i in range(1,seq_len):
                CPSA[i] = CPSA[i-1] + PSA[i]
                CPSG[i] = CPSG[i - 1] + PSG[i]
                CPSC[i] = CPSC[i - 1] + PSC[i]
                CPST[i] = CPST[i - 1] + PST[i]
            del CPSA[0]
            del CPSG[0]
            del CPSC[0]
            del CPST[0]
            MA1 = sum(CPSA)/seq_len
            MA2 = sum([ num*num for num in CPSA])/(na*(seq_len-na)*seq_len*seq_len)
            MG1 = sum(CPSG) / seq_len
            MG2 = sum([num * num for num in CPSG]) / (ng * (seq_len - ng) * seq_len * seq_len)
            MC1 = sum(CPSC) / seq_len
            MC2 = sum([num * num for num in CPSC]) / (nc * (seq_len - nc) * seq_len * seq_len)
            MT1 = sum(CPST) / seq_len
            MT2 = sum([num * num for num in CPST]) / (nt * (seq_len - nt) * seq_len * seq_len)
            CMA1 =sum([abs(num) for num in (CPSA-np.mean(CPSA))])/seq_len
            CMA2 =sum([num*num for num in (CPSA-np.mean(CPSA))])/(na*(seq_len-na)*seq_len*seq_len)
            CMG1 = sum([abs(num) for num in (CPSG-np.mean(CPSG))]) / seq_len
            CMG2 = sum([num * num for num in (CPSG - np.mean(CPSG))]) / (ng * (seq_len - ng) * seq_len * seq_len)
            CMC1 = sum([abs(num) for num in (CPSC-np.mean(CPSC))]) / seq_len
            CMC2 = sum([num * num for num in (CPSC - np.mean(CPSC))]) / (nc * (seq_len - nc) * seq_len * seq_len)
            CMT1 = sum([abs(num) for num in (CPST-np.mean(CPST))]) / seq_len
            CMT2 = sum([num * num for num in (CPST - np.mean(CPST))]) / (nt * (seq_len - nt) * seq_len * seq_len)
            res = [id[num],protein[num],gene[num],MA1,MA2,MG1,MG2,MC1,MC2,MT1,MT2,CMA1,CMA2,CMG1,CMG2,CMC1,CMC2,CMT1,CMT2,ess_tag[num]]
            for item in res:
                txt.write(str(item)+"\t")
def cal_shannon_entropy(file):#计算DNA序列香农熵（单个核苷酸，二联核苷酸，三联核苷酸）
    for filepath in file:
        print(filepath)
        content = pd.read_table(filepath,sep="\t",skiprows= [0],header=None)
        gene = content.iloc[:,2].values
        id = content.iloc[:,0].values
        protein = content.iloc[:,1].values
        ess_tag = content.iloc[:,3].values
        txt = open("../shannon_entropy/"+filepath.split("/")[-1].split(".")[0]+".txt","w")
        col_name = ["ID","pro","seq","single","double","triple","label"]
        for item in col_name:
            txt.write(item+"\t")
        for num  in range(len(gene)):
            txt.write("\n")
            seq = gene[num].replace("A","0")
            seq = seq.replace("a", "0")
            seq = seq.replace("G", "1")
            seq = seq.replace("g", "1")
            seq = seq.replace("C", "2")
            seq = seq.replace("c", "2")
            seq = seq.replace("T", "3")
            seq = seq.replace("t", "3")
            seq = seq.replace("B", "1")
            seq = seq.replace("b", "1")
            seq = seq.replace("D", "1")
            seq = seq.replace("d", "1")
            seq = seq.replace("H", "0")
            seq = seq.replace("h", "0")
            seq = seq.replace("V", "2")
            seq = seq.replace("v", "2")
            seq = seq.replace("M", "0")
            seq = seq.replace("m", "0")
            seq = seq.replace("R", "1")
            seq = seq.replace("r", "1")
            seq = seq.replace("K", "1")
            seq = seq.replace("k", "1")
            seq = seq.replace("S", "2")
            seq = seq.replace("s", "2")
            seq = seq.replace("W", "3")
            seq = seq.replace("w", "3")
            seq = seq.replace("Y", "2")
            seq = seq.replace("y", "2")
            seq = seq.replace("N", "2")
            seq = seq.replace("n", "2")
            seq = seq.replace("I", "4")
            seq = seq.replace("i", "4")
            shannon1 = 0
            for item in ["0","1","2","3"]:
                p = seq.count(item)/len(seq)
                if p!=0:
                    shannon1 = shannon1-p*math.log(p,2)
                else:
                    pass
            nuc_2=[]
            for i in range(len(gene)-1):
                nuc_2.append(seq[i:i+2])
            shannon2 = 0
            for one in ["0", "1", "2", "3"]:
                for two in ["0", "1", "2", "3"]:
                    p = nuc_2.count(one + two) / (len(seq) - 1)
                    if p != 0:
                        shannon2 = shannon2 - p * math.log(p, 2)
                    else:
                        pass
            nuc_3=[]
            for i in range(len(gene)-2):
                nuc_3.append(seq[i:i+3])
            shannon3 = 0
            for one in  ["0","1","2","3"]:
                for two in  ["0","1","2","3"]:
                    for three in  ["0","1","2","3"]:
                        p = nuc_3.count(one + two + three) / (len(seq) - 2)
                        if p!=0:
                            shannon3 = shannon3-p*math.log(p,2)
                        else:
                            pass
            feature = [id[num], protein[num], gene[num], shannon1, shannon2, shannon3, ess_tag[num]]
            for item in feature:
                txt.write(str(item) + "\t")
def cal_MI(file):#计算互信息
    for filepath in file:
        print(filepath)
        content = pd.read_table(filepath, sep="\t", skiprows=[0], header=None)
        gene = content.iloc[:, 2].values
        id = content.iloc[:, 0].values
        protein = content.iloc[:, 1].values
        ess_tag = content.iloc[:, 3].values
        txt = open("../MI/" + filepath.split("/")[-1].split(".")[0] + ".txt", "w")
        col_name = ["ID", "pro","seq"]
        for item in col_name:
            txt.write(item + "\t")
        for i in ["A","G","C","T"]:
            for j in ["A", "G", "C", "T"]:
                txt.write(i+j+"_MI\t")
        txt.write("total_MI"+"\t"+"label"+"\t")
        for num in range(len(gene)):
            txt.write("\n")
            seq = gene[num].replace("A", "0")
            seq = seq.replace("a", "0")
            seq = seq.replace("G", "1")
            seq = seq.replace("g", "1")
            seq = seq.replace("C", "2")
            seq = seq.replace("c", "2")
            seq = seq.replace("T", "3")
            seq = seq.replace("t", "3")
            seq = seq.replace("B", "1")
            seq = seq.replace("b", "1")
            seq = seq.replace("D", "1")
            seq = seq.replace("d", "1")
            seq = seq.replace("H", "0")
            seq = seq.replace("h", "0")
            seq = seq.replace("V", "2")
            seq = seq.replace("v", "2")
            seq = seq.replace("M", "0")
            seq = seq.replace("m", "0")
            seq = seq.replace("R", "1")
            seq = seq.replace("r", "1")
            seq = seq.replace("K", "1")
            seq = seq.replace("k", "1")
            seq = seq.replace("S", "2")
            seq = seq.replace("s", "2")
            seq = seq.replace("W", "3")
            seq = seq.replace("w", "3")
            seq = seq.replace("Y", "2")
            seq = seq.replace("y", "2")
            seq = seq.replace("N", "2")
            seq = seq.replace("n", "2")
            seq = seq.replace("I", "4")
            seq = seq.replace("i", "4")
            key = ["0","1","2","3"]#a,g,c,t
            total = 0
            MI = []
            nuc_2=[]
            for i in range(len(seq)-1):
                nuc_2.append(seq[i:i+2])
            for one in key:
                for two in key:
                    p1 = seq.count(one) / len(seq)
                    p2 = seq.count(two) / len(seq)
                    p12 = nuc_2.count(one+two)/(len(seq)-1)
                    if p1!=0 and p2 !=0:
                        if p12 !=0:
                            mi = p12*math.log(p12/(p1*p2),2)
                            total =total +  mi
                            MI.append(mi)
                        else:
                            MI.append(0)
                    else:
                        MI.append(0)
            MI.append(total)
            feature = [id[num], protein[num], gene[num]]
            for item in feature:
                txt.write(str(item)+"\t")
            for item in MI:
                txt.write(str(item)+"\t")
            txt.write(str(ess_tag[num])+"\t")

def cal_CMI(file):#计算CMI
    for filepath in file:
        print(filepath)
        content = pd.read_table(filepath, sep="\t", skiprows=[0], header=None)
        gene = content.iloc[:, 2].values
        id = content.iloc[:, 0].values
        protein = content.iloc[:, 1].values
        ess_tag = content.iloc[:, 3].values
        txt = open("../CMI/" + filepath.split("/")[-1].split(".")[0] + ".txt", "w")
        col_name = ["ID", "pro","seq"]
        for item in col_name:
            txt.write(item + "\t")
        for z in ["A","G","C","T"]:
            for x in ["A","G","C","T"]:
                for y in ["A", "G", "C", "T"]:
                    txt.write(x+z+y+"_CMI\t")
        txt.write("total_CMI"+"\t"+"label"+"\t")
        for num in range(len(gene)):
            txt.write("\n")
            seq = gene[num].replace("A", "0")
            seq = seq.replace("a", "0")
            seq = seq.replace("G", "1")
            seq = seq.replace("g", "1")
            seq = seq.replace("C", "2")
            seq = seq.replace("c", "2")
            seq = seq.replace("T", "3")
            seq = seq.replace("t", "3")
            seq = seq.replace("B", "1")
            seq = seq.replace("b", "1")
            seq = seq.replace("D", "1")
            seq = seq.replace("d", "1")
            seq = seq.replace("H", "0")
            seq = seq.replace("h", "0")
            seq = seq.replace("V", "2")
            seq = seq.replace("v", "2")
            seq = seq.replace("M", "0")
            seq = seq.replace("m", "0")
            seq = seq.replace("R", "1")
            seq = seq.replace("r", "1")
            seq = seq.replace("K", "1")
            seq = seq.replace("k", "1")
            seq = seq.replace("S", "2")
            seq = seq.replace("s", "2")
            seq = seq.replace("W", "3")
            seq = seq.replace("w", "3")
            seq = seq.replace("Y", "2")
            seq = seq.replace("y", "2")
            seq = seq.replace("N", "2")
            seq = seq.replace("n", "2")
            seq = seq.replace("I", "4")
            seq = seq.replace("i", "4")
            key = ["0", "1", "2", "3"]  # a,g,c,t
            total = 0
            CMI = []
            nuc_2 = []
            for i in range(len(seq) - 1):
                nuc_2.append(seq[i:i + 2])
            nuc_3=[]
            for i in range(len(seq) - 2):
                nuc_3.append(seq[i:i + 3])
            for z in key:
                for x in key:
                    for y in key:
                        pz = seq.count(z)/len(seq)
                        pxz = nuc_2.count(x+z)/(len(seq)-1)
                        pzy = nuc_2.count(z+y)/(len(seq)-1)
                        pxzy = nuc_3.count(x+z+y)/(len(seq)-2)
                        if pz!=0 and pxz!=0 and pzy !=0 and pxzy!=0:
                            cmi = pxzy*math.log((pz*pxzy)/(pxz*pzy),2)
                            CMI.append(cmi)
                            total = total+cmi
                        else:
                            CMI.append(0)

            CMI.append(total)
            feature = [id[num], protein[num], gene[num],]
            for item in feature:
                txt.write(str(item) + "\t")
            for item in CMI:
                txt.write(str(item) + "\t")
            txt.write(str(ess_tag[num])+"\t")



if __name__ == '__main__' :
    path = "../data_ori/"
    type = ".txt"
    filelist = read_filename(path,type)
    hurst(filelist)#计算hurst指数，蛋白质长度，基因长度三个特征
    cal_feature(filelist)#计算氨基酸频率
    cal_nucleotide_fre(filelist)#计算核苷酸在密码子三个位置的频率
    cal_inter_nucleotide_features(filelist)#计算核苷酸间距离特征
    cal_inter_amino_features(filelist)#计算氨基酸间距离特征
    cal_CPS_features(filelist)#计算基于累计傅里叶功率谱特征
    cal_shannon_entropy(filelist)#计算序列香农熵，三维特征
    cal_MI(filelist)#计算序列互信息，17维特征
    cal_CMI(filelist)#计算条件互信息
