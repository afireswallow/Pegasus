#!/usr/bin/env python

from time import sleep
import torch
import numpy as np
import pickle

model = torch.load(
    'CICIOT2022_finalLUT.pt',
    map_location=torch.device('cpu'), 
    weights_only=True
)
max_num = 100000000
min_num = -max_num


def from_T_H_to_range(T, H):#T矩阵对饮一颗树的层次遍历，H矩阵对应树的遍历方式
    tt = []
    for i in range(T.shape[0]):
        t = []
        for j in range(15):
            valid = 1
            node = 0
            left = -10000000; right = 100000000
            for k in range(4):
                if int(H[node][j]) == -1:
                    if left > int(T[i][node] // 1) :
                        valid = 0
                    elif right > int(T[i][node] // 1) :
                        right = int(T[i][node] // 1)
                    node = node * 2 + 1
                elif int(H[node][j]) == 1:
                    if right < int(T[i][node] // 1) :
                        valid = 0
                    elif left < int(T[i][node] // 1) :
                        left = int(T[i][node] // 1)
                    node = node * 2 + 2
                else :
                    print('error:' + str(int(H[node][j])) + ' in H is invalid!')
            if valid == 1 and left < right:
                t.append([right, j])
        tt.append(t)
    return tt
def from_S_T_H_to_range2(S, T, H):#T矩阵对饮一颗树的层次遍历，H矩阵对应树的遍历方式, S矩阵对应选择的比较方式
    cc = []
    for i in range(T.shape[0]): #i遍历聚类树
        t = [] ; u = [] ; c = []
        for j in range(len(S[i])): #j遍历片选维度
            t.append([]) ; u.append([]) ; c.append([])
        for j in range(16): #j遍历每棵树的叶节点
            for m in range(len(S[i])): #m遍历量化维度
                node = 0
                left = min_num; right = max_num
                for k in range(4):
                    if int(H[node][j]) == -1:
                        if int(S[i][m][node]) == 1 and right > int(T[i][node] // 1) :
                            right = int(T[i][node] // 1)
                        node = node * 2 + 1
                    elif int(H[node][j]) == 1:
                        if int(S[i][m][node]) == 1 and left < int(T[i][node] // 1) :
                            left = int(T[i][node] // 1)
                        node = node * 2 + 2
                    else :
                        print('error:' + str(int(H[node][j])) + ' in H is invalid!')
                if left < right:
                    t[m].append([left, right, j]) #列表t为[区间左端点,区间右端点,区间对应的叶节点编号]
                    if right not in u[m] :
                        u[m].append(right)
                u[m].sort()
        for m in range(len(u)) :
            for j in range(len(u[m])) :
                c[m].append([u[m][j], 0])
        for m in range(len(t)) :
            for j in range(len(t[m])) :
                l = 0 ; r = len(u[m]) - 1
                while u[m][l] <= t[m][j][0] :
                    l += 1
                while u[m][r] > t[m][j][1] :
                    r -= 1
                while l <= r :
                    c[m][l][1] = c[m][l][1] | (1 << t[m][j][2])
                    l += 1
        cc.append(c)
    return cc
def from_range_to_ternary(tt, bitLengthList, LUT, signed = 1, returnCategory = False, reservedOperation = 1): # tt: 行X树节点数值和范围号, bitLength: 位数
    ternary = [];usedCategoryList = []
    for i in range(len(tt)):
        bitLength = bitLengthList[i] if isinstance(bitLengthList, list) else bitLengthList
        if signed == 1: #带符号数
            min = -(1 << (bitLength - 1)) - 1
            max = (1 << (bitLength - 1)) - 1
        elif signed == 0: #无符号数
            min = -1
            max = (1 << bitLength) - 1
        t = [[min, -1]]
        t.extend(tt[i])
        t.append([max, 15])
        if reservedOperation == 1:
            if t[1][0] <= min:
                t[1][0] = min + 1
        elif reservedOperation == 2:
            if t[1][0] <= min:
                t[1][0] = -128
        elif reservedOperation == 3:
            if t[1][0] <= min:
                t[1][0] = -1024
        elif reservedOperation == 4:
            if t[1][0] <= min:
                t[1][0] = -12800
        elif reservedOperation == 5:
            j = len(t) - 2
            while j > 0:
                if t[j][0] < 0:
                    t[j][0] = t[j+1][0] - 1
                    t[j][1] = t[j+1][1]
                j -= 1
        usedCategory = [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]
        if returnCategory :
            for j in range(len(t)):
                usedCategory[t[j][1]] = 1
            usedCategoryList.append(usedCategory)
        n = 0
        s = []
        while n < bitLength - 1:
            k = []
            j = 1
            while j < len(t)-1 :
                if t[j][0] % 2 == 0 :
                    m = t[j][0]//2
                    if m % 2 == 0 : #父节点为左孩子
                        if t[j][0] + 1 != t[j+1][0] : #右节点连续多个同类别
                            s.append(from_tree_node_to_ternary(t[j][0], n, bitLength, t[j][1], i, LUT, returnCategory))
                            if t[j][0] - 1 != t[j-1][0] :
                                k.append([m-1, t[j][1]]); t[j][0] -= 1
                        else :
                            k.append([m, t[j][1]]); t[j][0] += 1
                            s.append(from_tree_node_to_ternary(t[j][0], n, bitLength, t[j+1][1], i, LUT, returnCategory))
                            j += 1
                    else : #父节点为右孩子
                        if t[j][0] - 1 != t[j-1][0] : #左节点连续多个同类别
                            s.append(from_tree_node_to_ternary(t[j][0] + 1, n, bitLength, t[j+1][1], i, LUT, returnCategory))
                            k.append([m, t[j][1]]); t[j][0] += 1
                            if t[j][0] == t[j+1][0] :
                                t[j+1][1] = t[j][1]
                                j += 1
                        else :
                            s.append(from_tree_node_to_ternary(t[j][0], n, bitLength, t[j][1], i, LUT, returnCategory))
                else :
                    k.append([t[j][0]//2, t[j][1]])
                j += 1
            if j == len(t) - 1 :
                k.append([t[-1][0]//2, t[-1][1]])
            t = [[t[0][0]//2, t[0][1]]]
            t.extend(k)
            #print(t)
            n += 1
        s.append(from_tree_node_to_ternary(t[-1][0], n, bitLength, t[-1][1], i, LUT, returnCategory))
        if t[-1][0] == t[0][0] :
            print('warning ' + str(t))
        elif t[-2][0] != t[0][0] :
            s.append(from_tree_node_to_ternary(t[-2][0], n, bitLength, t[-2][1], i, LUT, returnCategory))
        else:
            s.append(from_tree_node_to_ternary(t[-2][0] + 1, n, bitLength, t[-1][1], i, LUT, returnCategory))
        ternary.append(s)
    #行X表项数量X(优先级，匹配串，掩码，[参数列表])
    return (ternary, usedCategoryList) if returnCategory else ternary
def from_tree_node_to_ternary(m, n, bitLength, category, key = None, LUT = None, returnCategory = True): # m: 树节点数值, n: 树节点高度, bitLength: 位数
    mask = "0"
    for i in range(bitLength - n) :
        mask += "1"
    for i in range(n) :
        mask += "0"
    parameter = category if returnCategory else [int(num) for num in torch.floor(LUT[key][category]).tolist()]
    return [n, (m % (1 << (bitLength - n))) << n, int(mask, 2),  parameter]
def from_range_to_ternary_partly(t, bitLength): #t:n个区间X[区间右端点,区间包含的类号],区间为左开右闭
    n = 0 ; s = []
    while n < bitLength - 1:
        k = []
        j = 1
        while j < len(t) - 1 :
            if t[j][0] % 2 == 0 :
                m = t[j][0] // 2
                if m % 2 == 0 : #父节点为左孩子
                    if t[j][0] + 1 != t[j+1][0] : #右节点连续多个同类别
                        s.append(from_tree_node_to_ternary(t[j][0], n, bitLength, t[j][1], returnCategory = True))
                        if t[j][0] - 1 != t[j-1][0] :
                            k.append([m-1, t[j][1]]); t[j][0] -= 1
                    else :
                        k.append([m, t[j][1]]); t[j][0] += 1
                        s.append(from_tree_node_to_ternary(t[j][0], n, bitLength, t[j+1][1], returnCategory = True))
                        j += 1
                else : #父节点为右孩子
                    if t[j][0] - 1 != t[j-1][0] : #左节点连续多个同类别
                        s.append(from_tree_node_to_ternary(t[j][0] + 1, n, bitLength, t[j+1][1], returnCategory = True))
                        k.append([m, t[j][1]]); t[j][0] += 1
                        if t[j][0] == t[j+1][0] :
                            t[j+1][1] = t[j][1]
                            j += 1
                    else :
                        s.append(from_tree_node_to_ternary(t[j][0], n, bitLength, t[j][1], returnCategory = True))
            else :
                k.append([t[j][0] // 2, t[j][1]])
            j += 1
        if j == len(t) - 1 :
            k.append([t[-1][0] // 2, t[-1][1]])
        t = [[t[0][0] // 2, t[0][1]]]
        t.extend(k)
        n += 1
    s.append(from_tree_node_to_ternary(t[-1][0], n, bitLength, t[-1][1], returnCategory = True))
    if t[-1][0] == t[0][0] :
        print('warning ' + str(t))
    elif t[-2][0] != t[0][0] :
        s.append(from_tree_node_to_ternary(t[-2][0], n, bitLength, t[-2][1], returnCategory = True))
    else:
        s.append(from_tree_node_to_ternary(t[-2][0] + 1, n, bitLength, t[-1][1], returnCategory = True))
    return s
def from_range2_to_ternary2(tt, bitLengthList1, bitLengthList2, LUT = None, shift = 19, signed = 1, returnCategory = True, link = False, reservedOperation = 1): # tt: 行X树节点数值和范围号, bitLength: 位数
    bitLength1 = bitLengthList1
    bitLength2 = bitLengthList2
    if signed == 1: #带符号数
        min1 = -(1 << (bitLength1 - 1)) - 1
        max1 = (1 << (bitLength1 - 1)) - 1
    elif signed == 0: #无符号数
        min1 = -1
        max1 = (1 << bitLength1) - 1
    if signed == 1: #带符号数
        min2 = -(1 << (bitLength2 - 1)) - 1
        max2 = (1 << (bitLength2 - 1)) - 1
    elif signed == 0: #无符号数
        min2 = -1
        max2 = (1 << bitLength2) - 1
    usedCategoryList = []; ternary1 = []; ternary2 = []
    for i in range(len(tt)):
        t1 = tt[i][0] ; t2 = tt[i][1]
        l = [[min1, 0]] ; r = [[min2, 0]]
        for j in range(len(t1)):
            if min1 < t1[j][0]:
                if max1 < t1[j][0]:
                    t1[j][0] = max1
                if l[-1][0] != t1[j][0]: #如果有多个超过max的区间有边界，那么只保留第一个
                    l.append([t1[j][0], t1[j][1]])
        for j in range(len(t2)):
            if min2 < t2[j][0]:
                if max2 < t2[j][0]:
                    t2[j][0] = max2
                if r[-1][0] != t2[j][0]:
                    r.append([t2[j][0], t2[j][1]])
        ternary1.append(from_range_to_ternary_partly(l, bitLength1))
        ternary2.append(from_range_to_ternary_partly(r, bitLength2))
    ternary = []
    mask1 = (1 << bitLength1) - 1
    mask2 = (1 << bitLength2) - 1
    for i in range(len(ternary1)):
        s = [] ; usedCategory = [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]
        for j in range(len(ternary1[i])):
            for k in range(len(ternary2[i])):
                a = ternary1[i][j][-1] & ternary2[i][k][-1]
                if a > 0:
                    for b in range(16):
                        if a & (1 << b) > 0:
                            usedCategory[b] = 1
                            if link == False:
                                if returnCategory:
                                    s.append([ternary1[i][j][0] + ternary2[i][k][0], ((ternary1[i][j][1] & mask1) << shift) | (ternary2[i][k][1] & mask2), ((ternary1[i][j][2] & mask1) << shift) | (ternary2[i][k][2] & mask2), b])
                                else :
                                    s.append([ternary1[i][j][0] + ternary2[i][k][0], ((ternary1[i][j][1] & mask1) << shift) | (ternary2[i][k][1] & mask2), ((ternary1[i][j][2] & mask1) << shift) | (ternary2[i][k][2] & mask2),
                                            [int(num) for num in torch.floor(LUT[i][b]).tolist()]])
                            else :
                                if returnCategory:
                                    s.append([ternary1[i][j][0] + ternary2[i][k][0], ternary1[i][j][1] & mask1, ternary2[i][k][1] & mask2, ternary1[i][j][2] & mask1, ternary2[i][k][2] & mask2, b])
                                else :
                                    s.append([ternary1[i][j][0] + ternary2[i][k][0], ternary1[i][j][1] & mask1, ternary2[i][k][1] & mask2, ternary1[i][j][2] & mask1, ternary2[i][k][2] & mask2,
                                            [int(num) for num in torch.floor(LUT[i][b]).tolist()]])
        ternary.append(s)
        usedCategoryList.append(usedCategory)
    #[优先级，匹配串1，匹配串2，掩码1，掩码2，类别号]
    return (ternary, usedCategoryList) if returnCategory else ternary


#ternary为i个输入的查找表,ternary[i]对应所有的匹配元组,ternary[i][j]的格式为[优先级，匹配串，掩码，[参数列表]]
def from_tables_to_table(ternary, bitlength, match = 0, doFuseOutput = True, exact = False):
    i = 0
    table = []
    if match == 0: # 什么也不干
        table = ternary
    elif match == 1: # 将两个元组对应的两个输入融合为一个数
        while i < len(ternary) - 1:
            table.append(fuseInput(ternary[i], bitlength, ternary[i+1], exact = exact))
            i += 2
        if i != len(ternary):
            table.append(ternary[i])
    elif match == 2: # 将两个元组对应的两个输入链接为两个数
        while i < len(ternary) - 1:
            table.append(linkInput(ternary[i], ternary[i+1], exact = exact))
            i += 2
        if i != len(ternary):
            table.append(ternary[i])
    elif match == 3: # 匹配结果为分类编号，而不是参数列表
        while i < len(ternary) - 1:
            table.append(fuseInput(ternary[i], bitlength, ternary[i+1], useCategory = True, bitlength2 = 4))
            i += 2
        if i != len(ternary):
            table.append(ternary[i])
    elif match == 4: # 将三个元组对应的三个输入链接为三个数
        while i < len(ternary) - 1:
            table.append(linkInput3(ternary[i], ternary[i+1], ternary[i+2]))
            i += 3
    elif match == 5:
        while i < len(ternary) - 1:
            table.append(fuseInput(ternary[i], bitlength, ternary[i+1], exact = True))
            i += 2
        if i != len(ternary):
            table.append(ternary[i])
    if doFuseOutput: #有潜在问题
        table = fuseOutput(table, bitlength, bitlength)
    return table
def fuseInput(table1, bitlength1, table2, useCategory = False, exact = False, bitlength2 = 0):
    table = []
    for i in range(len(table1)):
        for j in range(len(table2)):
            parameter = []
            if useCategory :
                parameter = ((table1[i][3] << bitlength2) | table2[j][3])
            else :
                for k in range(len(table1[i][-1])):
                    parameter.append(table1[i][-1][k] + table2[j][-1][k])
            if exact == False:
                table.append([table1[i][0] + table2[j][0], ((table1[i][1] & 0x1fff) << bitlength1) | (table2[j][1] & 0x1fff), ((table1[i][2] & 0x1fff) << bitlength1) | (table2[j][2] & 0x1fff), parameter])
            else :
                table.append([(table1[i][0] << bitlength1) | table2[j][0], parameter])
    return table
def linkInput(table1, table2, exact = False):
    table = []
    for i in range(len(table1)):
        for j in range(len(table2)):
            parameter = []
            for k in range(len(table1[i][-1])):
                parameter.append(table1[i][-1][k] + table2[j][-1][k])
            if exact == False:
                table.append([table1[i][0] + table2[j][0], table1[i][1], table2[j][1], table1[i][2], table2[j][2], parameter])
            else :
                table.append([table1[i][0], table2[j][0], parameter])
    return table
def linkInput3(table1, table2, table3, exact = True):
    table = []
    for i in range(len(table1)):
        for j in range(len(table2)):
            for l in range(len(table3)):
                parameter = []
                for k in range(len(table1[i][-1])):
                    parameter.append(table1[i][-1][k] + table2[j][-1][k] + table3[l][-1][k])
                if exact :
                    table.append([table1[i][0], table2[j][0], table3[l][0], parameter])
    return table
def fuseOutput(table1, bitlength1, bitlength2):
    mask = (1 << bitlength1) - 1
    for i in range(len(table1)):
        for k in range(len(table1[i])):
            table = []
            for j in range(len(table1[i][k][-1]) // 2):
                table.append(((table1[i][k][-1][2*j] & mask) << bitlength2) | (table1[i][k][-1][2*j+1] & mask))
            table1[i][k][-1] = table
    return table1
def fuseOutput_partly(table1, bitlength2):
    mask = (1 << bitlength2) - 1
    for k in range(len(table1)):
        table = []
        for j in range(len(table1[k][-1]) // 2):
            table.append(((table1[k][-1][2*j] & mask) << bitlength2) | (table1[k][-1][2*j+1] & mask))
        table1[k][-1] = table
    return table1
def from_LUT_to_exact(table1, input, LUT_shape0 = 16):
    tables = []
    for i in range(len(table1)):
        table = []
        for j in range(LUT_shape0):
            if input[i][j] == 1:
                table.append([j, [int(num) for num in torch.floor(table1[i][j]).tolist()]])
        tables.append(table)
    return tables
def from_LUT_to_exact_partly(table1, input = None, LUT_shape0 = None):
    table = []
    if LUT_shape0 == None:
        LUT_shape0 = len(table1)
    for j in range(LUT_shape0):
        if input == None or input[j] == 1:
            table.append([j, [int(num) for num in torch.floor(table1[j]).tolist()]])
    return table
def test_from_T_H_to_range(Range, T, l):
    print('test_from_T_H_to_range'); sleep(1)
    for k in range(len(T)):
        for r in range(len(Range[k])):
            while l <= Range[k][r][0]:
                node = 0
                category = 0
                for i in range(4):
                    if l <= T[k][node]:
                        node = node * 2 + 1
                        category = category * 2
                    else :
                        node = node * 2 + 2
                        category = category * 2 + 1
                t = (category == Range[k][r][1])
                if t == False:
                    print(l, t, category); print('error')
                    sleep(10)
                print(l, t, category)
                l += 1
def get_category_from_T(T, k, l):
    node = 0
    category = 0
    for i in range(4):
        if l <= int(T[k][node]//1):
            node = node * 2 + 1
            category = category * 2
        else :
            node = node * 2 + 2
            category = category * 2 + 1
    return category
def get_parameter_from_T_LUT(T, LUT, k, l):
    category = get_category_from_T(T, k, l)
    return [int(num) for num in torch.floor(LUT[k][category]).tolist()]
def list_add(list1, list2):
    list = []
    for i in range(len(list1)):
        list.append(list1[i] + list2[i])
    return list
def lists_add(list):
    lists = list[-1]
    for i in range(len(list) - 1):
        lists = list_add(lists, list[i])
    return lists
def cnn_bias(table1):
    temp = table1[0][0][-1][-1]
    for j in range(len(table1[0])):
        for k in range(len(table1[0][j][-1])):
            if k != 0:
                table1[0][j][-1][k] -= temp
    for i in range(1, len(table1)):
        for j in range(len(table1[i])):
            if i < len(table1[i][j][-1]):
                table1[i][j][-1][i] += temp
    return table1



S1 = model['MM1.S']
T1 = model['MM1.T']
H1 = model['MM1.H']
LUT1 = model['MM1.LUT']
S2 = model['MM2.S']
T2 = model['MM2.T']
H2 = model['MM2.H']
LUT2 = model['MM2.LUT']

print('S1: ', S1.shape)
print('T1: ', T1.shape)
print('H1: ', H1.shape)
print('LUT1: ', LUT1.shape)
print('S2: ', S2.shape)
print('T2: ', T2.shape)
print('H2: ', H2.shape)
print('LUT2: ', LUT2.shape)
print('LUT1: ', LUT1)
print('T2: ', T2)


tt1 = from_S_T_H_to_range2(S1, T1, H2)
tt2 = from_S_T_H_to_range2(S2, T2, H2)
table01, usedCategory01 = from_range2_to_ternary2(tt1[0:27], 8, 8, shift = 0, signed = 0, returnCategory = True, link = True, reservedOperation = 0)
table02, usedCategory02 = from_range2_to_ternary2(tt1[27:28], 8, 16, shift = 0, signed = 0, returnCategory = True, link = True, reservedOperation = 0)
table03, usedCategory03 = from_range2_to_ternary2(tt1[28:29], 16, 8, shift = 0, signed = 0, returnCategory = True, link = True, reservedOperation = 0)
table04, usedCategory04 = from_range2_to_ternary2(tt1[29:30], 4, 4, shift = 0, signed = 0, returnCategory = True, link = True, reservedOperation = 0)
table2, usedCategory2 = from_range2_to_ternary2(tt2, 16, 16, shift = 19, signed = 1, returnCategory = True, link = True, reservedOperation = 0)
usedCategory1 = []
usedCategory1.extend(usedCategory01)
usedCategory1.extend(usedCategory02)
usedCategory1.extend(usedCategory03)
usedCategory1.extend(usedCategory04)
table11 = []
for j in range(10):
    table = []
    for p in range(16):
        for q in range(16):
            for r in range(16):
                if usedCategory1[3*j][p] == 1 and usedCategory1[3*j+1][q] == 1 and usedCategory1[3*j+2][r] == 1:
                    table.append([p, q, r,
                                    [int(num) for num in torch.floor(LUT1[10*0+j][(p << 8) + (q << 4) + r]).tolist()]])
    table11.append(table)
t2 = from_LUT_to_exact(LUT2, usedCategory2)
table22 = from_tables_to_table(t2[0:6], 4, match = 4, doFuseOutput = False, exact = True)
table23 = from_tables_to_table(t2[6:8], 4, match = 2, doFuseOutput = False, exact = True)



with open('cnnl-CICIOT.pkl','wb') as f:
    #p1 #p2 #p3 #x4
    pickle.dump([table01, table02, table03, table04, table2, table11, table22, table23], f, protocol = 2)

