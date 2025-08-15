from time import sleep
import torch
import numpy as np
import pickle
import pandas as pd
import torch
from sklearn.metrics import f1_score



model = torch.load('/home/yang1210/THU_net/BMV2_Pegasus/make_dataset/mlp/CICIOT2022/mlp_CICIOT2022_0.8563548831138729.pt', weights_only = True)
# model = torch.load('/home/yang1210/THU_net/BMV2_Pegasus/make_dataset/mlp/PeerRush/mlp_PeerRush_0.8835371087494018.pt', weights_only = True)
# model = torch.load('/home/yang1210/THU_net/BMV2_Pegasus/make_dataset/mlp/ISCXVPN/mlp_ISCXVPN_0.7578214949676978.pt', weights_only = True)
def from_T_H_to_range(T, H):#T matrix corresponds to level traversal of a tree, H matrix corresponds to tree traversal method
    tt = []
    for i in range(T.shape[0]):
        t = []
        for j in range(15):
            valid = 1
            node = 0
            left = -10000000; right = 100000000
            for k in range(4):
                if int(H[node][j]) == -1:
                    if left > int(T[i][node]//1) :
                        valid = 0
                    elif right > int(T[i][node]//1) :
                        right = int(T[i][node]//1)
                    node = node * 2 + 1
                elif int(H[node][j]) == 1:
                    if right < int(T[i][node]//1) :
                        valid = 0
                    elif left < int(T[i][node]//1) :
                        left = int(T[i][node]//1)
                    node = node * 2 + 2
                else :
                    print('error:' + str(int(H[node][j])) + ' in H is invalid!')
            if valid == 1 and left < right:
                t.append([right, j])
        tt.append(t)
    return tt
def from_S_T_H_to_range2(S, T, H):#T matrix corresponds to level traversal of a tree, H matrix corresponds to tree traversal method, S matrix corresponds to selected comparison method
    cc1 = [] ; cc2 = []
    for i in range(T.shape[0]):
        t1 = [] ; t2 = [] ; u1 = [] ; u2 = [] ; c1 = [] ; c2 = []
        for j in range(16):
            valid = 1
            node = 0
            left = -10000000; right = 100000000
            for k in range(4):
                if int(H[node][j]) == -1:
                    if int(S[i][0][node]) == 1 :
                        if left > int(T[i][node]//1) :
                            valid = 0
                        elif right > int(T[i][node]//1) :
                            right = int(T[i][node]//1)
                    node = node * 2 + 1
                elif int(H[node][j]) == 1:
                    if int(S[i][0][node]) == 1 :
                        if right < int(T[i][node]//1) :
                            valid = 0
                        elif left < int(T[i][node]//1) :
                            left = int(T[i][node]//1)
                    node = node * 2 + 2
                else :
                    print('error:' + str(int(H[node][j])) + ' in H is invalid!')
            if valid == 1 and left < right:
                t1.append([left, right, j])
                if right not in u1:
                    u1.append(right)
            valid = 1
            node = 0
            left = -10000000; right = 100000000
            for k in range(4):
                if int(H[node][j]) == -1:
                    if int(S[i][1][node]) == 1 :
                        if left > int(T[i][node]//1) :
                            valid = 0
                        elif right > int(T[i][node]//1) :
                            right = int(T[i][node]//1)
                    node = node * 2 + 1
                elif int(H[node][j]) == 1:
                    if int(S[i][1][node]) == 1 :
                        if right < int(T[i][node]//1) :
                            valid = 0
                        elif left < int(T[i][node]//1) :
                            left = int(T[i][node]//1)
                    node = node * 2 + 2
                else :
                    print('error:' + str(int(H[node][j])) + ' in H is invalid!')
            if valid == 1 and left < right:
                t2.append([left, right, j])
                if right not in u2:
                    u2.append(right)
        u1.sort() ; u2.sort()
        for j in range(len(u1)):
            c1.append([u1[j], 0])
        for j in range(len(u2)):
            c2.append([u2[j], 0])
        for j in range(len(t1)):
            l = 0 ; r = len(u1) - 1
            while u1[l] <= t1[j][0]:
                l += 1
            while u1[r] > t1[j][1]:
                r -= 1
            while l <= r:
                c1[l][1] =  c1[l][1] | (1 << t1[j][2])
                l += 1
        for j in range(len(t2)):
            l = 0 ; r = len(u2) - 1
            while u2[l] <= t2[j][0]:
                l += 1
            while u2[r] > t2[j][1]:
                r -= 1
            while l <= r:
                c2[l][1] =  c2[l][1] | (1 << t2[j][2])
                l += 1
        cc1.append(c1) ; cc2.append(c2)
    return cc1, cc2
def from_range_to_ternary(tt, bitLengthList, LUT, signed = 1, returnCategory = False, reservedOperation = 1): # tt: rows X tree node values and range numbers, bitLength: bit count
    ternary = [];usedCategoryList = []
    for i in range(len(tt)):
        bitLength = bitLengthList[i] if isinstance(bitLengthList, list) else bitLengthList
        if signed == 1: #signed numbers
            min = -(1 << (bitLength - 1)) - 1
            max = (1 << (bitLength - 1)) - 1
        elif signed == 0: #unsigned numbers
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
        #print('t', t)
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
                    if m % 2 == 0 : #parent node is left child
                        if t[j][0] + 1 != t[j+1][0] : #right node has consecutive multiple same categories
                            s.append(from_tree_node_to_ternary(t[j][0], n, bitLength, t[j][1], i, LUT, returnCategory))
                            if t[j][0] - 1 != t[j-1][0] :
                                k.append([m-1, t[j][1]]); t[j][0] -= 1
                        else :
                            k.append([m, t[j][1]]); t[j][0] += 1
                            s.append(from_tree_node_to_ternary(t[j][0], n, bitLength, t[j+1][1], i, LUT, returnCategory))
                            j += 1
                    else : #parent node is right child
                        if t[j][0] - 1 != t[j-1][0] : #left node has consecutive multiple same categories
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
    #rows X number of table entries X (priority, match string, mask, [parameter list])
    return (ternary, usedCategoryList) if returnCategory else ternary
def from_tree_node_to_ternary(m, n, bitLength, category, key, LUT, returnCategory = False): # m: tree node value, n: tree node height, bitLength: bit count
    mask = "0"
    for i in range(bitLength - n) :
        mask += "1"
    for i in range(n) :
        mask += "0"
    parameter = category if returnCategory else [int(num) for num in torch.floor(LUT[key][category]).tolist()] 
    return [n, (m % (1 << (bitLength - n))) << n, int(mask, 2),  parameter]

def from_range2_to_ternary2(tt1, tt2, bitLengthList1, bitLengthList2, LUT, signed = 1, returnCategory = True, reservedOperation = 1): # tt: rows X tree node values and range numbers, bitLength: bit count
    bitLength1 = bitLengthList1
    bitLength2 = bitLengthList2
    if signed == 1: #signed numbers
        min1 = -(1 << (bitLength1 - 1)) - 1
        max1 = (1 << (bitLength1 - 1)) - 1
    elif signed == 0: #unsigned numbers
        min1 = -1
        max1 = (1 << bitLength1) - 1
    if signed == 1: #signed numbers
        min2 = -(1 << (bitLength2 - 1)) - 1
        max2 = (1 << (bitLength2 - 1)) - 1
    elif signed == 0: #unsigned numbers
        min2 = -1
        max2 = (1 << bitLength2) - 1
    usedCategoryList = []; ternary1 = []; ternary2 = []
    for i in range(len(tt1)):
        t1 = tt1[i] ; t2 = tt2[i]
        l = [[min1, 0]] ; r = [[min2, 0]]
        for j in range(len(t1)):
            if min1 < t1[j][0]:
                if max1 < t1[j][0]:
                    t1[j][0] = max1
                if l[-1][0] != t1[j][0]:
                    l.append([t1[j][0], t1[j][1]])
        for j in range(len(t2)):
            if min2 < t2[j][0]:
                if max2 < t2[j][0]:
                    t2[j][0] = max1
                if r[-1][0] != t2[j][0]:
                    r.append([t2[j][0], t2[j][1]])
        n = 0 ; s = [] ; t = l
        while n < bitLength1 - 1:
            k = []
            j = 1
            while j < len(t) - 1 :
                if t[j][0] % 2 == 0 :
                    m = t[j][0]//2
                    if m % 2 == 0 : #parent node is left child
                        if t[j][0] + 1 != t[j+1][0] : #right node has consecutive multiple same categories
                            s.append(from_tree_node_to_ternary(t[j][0], n, bitLength1, t[j][1], i, LUT, returnCategory = True))
                            if t[j][0] - 1 != t[j-1][0] :
                                k.append([m-1, t[j][1]]); t[j][0] -= 1
                        else :
                            k.append([m, t[j][1]]); t[j][0] += 1
                            s.append(from_tree_node_to_ternary(t[j][0], n, bitLength1, t[j+1][1], i, LUT, returnCategory = True))
                            j += 1
                    else : #parent node is right child
                        if t[j][0] - 1 != t[j-1][0] : #left node has consecutive multiple same categories
                            s.append(from_tree_node_to_ternary(t[j][0] + 1, n, bitLength1, t[j+1][1], i, LUT, returnCategory = True))
                            k.append([m, t[j][1]]); t[j][0] += 1
                            if t[j][0] == t[j+1][0] :
                                t[j+1][1] = t[j][1]
                                j += 1
                        else :
                            s.append(from_tree_node_to_ternary(t[j][0], n, bitLength1, t[j][1], i, LUT, returnCategory = True))
                else :
                    k.append([t[j][0]//2, t[j][1]])
                j += 1
            if j == len(t) - 1 :
                k.append([t[-1][0]//2, t[-1][1]])
            t = [[t[0][0]//2, t[0][1]]]
            t.extend(k)
            n += 1
        s.append(from_tree_node_to_ternary(t[-1][0], n, bitLength1, t[-1][1], i, LUT, returnCategory = True))
        if t[-1][0] == t[0][0] :
            print('warning ' + str(t))
        elif t[-2][0] != t[0][0] :
            s.append(from_tree_node_to_ternary(t[-2][0], n, bitLength1, t[-2][1], i, LUT, returnCategory = True))
        else:
            s.append(from_tree_node_to_ternary(t[-2][0] + 1, n, bitLength1, t[-1][1], i, LUT, returnCategory = True))
        ternary1.append(s)
        n = 0 ; s = [] ; t = r
        while n < bitLength2 - 1:
            k = []
            j = 1
            while j < len(t) - 1 :
                if t[j][0] % 2 == 0 :
                    m = t[j][0]//2
                    if m % 2 == 0 : #parent node is left child
                        if t[j][0] + 1 != t[j+1][0] : #right node has consecutive multiple same categories
                            s.append(from_tree_node_to_ternary(t[j][0], n, bitLength2, t[j][1], i, LUT, returnCategory = True))
                            if t[j][0] - 1 != t[j-1][0] :
                                k.append([m-1, t[j][1]]); t[j][0] -= 1
                        else :
                            k.append([m, t[j][1]]); t[j][0] += 1
                            s.append(from_tree_node_to_ternary(t[j][0], n, bitLength2, t[j+1][1], i, LUT, returnCategory = True))
                            j += 1
                    else : #parent node is right child
                        if t[j][0] - 1 != t[j-1][0] : #left node has consecutive multiple same categories
                            s.append(from_tree_node_to_ternary(t[j][0] + 1, n, bitLength2, t[j+1][1], i, LUT, returnCategory = True))
                            k.append([m, t[j][1]]); t[j][0] += 1
                            if t[j][0] == t[j+1][0] :
                                t[j+1][1] = t[j][1]
                                j += 1
                        else :
                            s.append(from_tree_node_to_ternary(t[j][0], n, bitLength2, t[j][1], i, LUT, returnCategory = True))
                else :
                    k.append([t[j][0]//2, t[j][1]])
                j += 1
            if j == len(t) - 1 :
                k.append([t[-1][0]//2, t[-1][1]])
            t = [[t[0][0]//2, t[0][1]]]
            t.extend(k)
            n += 1
        s.append(from_tree_node_to_ternary(t[-1][0], n, bitLength1, t[-1][1], i, LUT, returnCategory = True))
        if t[-1][0] == t[0][0] :
            print('warning ' + str(t))
        elif t[-2][0] != t[0][0] :
            s.append(from_tree_node_to_ternary(t[-2][0], n, bitLength1, t[-2][1], i, LUT, returnCategory = True))
        else:
            s.append(from_tree_node_to_ternary(t[-2][0] + 1, n, bitLength1, t[-1][1], i, LUT, returnCategory = True))
        ternary2.append(s)
    ternary = []
    if reservedOperation == -1:
        print(ternary1)
        print(ternary2)
    for i in range(len(ternary1)):
        s = [] ; usedCategory = [0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]
        for j in range(len(ternary1[i])):
            for k in range(len(ternary2[i])):
                a = ternary1[i][j][-1] & ternary2[i][k][-1] 
                if a > 0:
                    for b in range(16):
                        if a & (1 << b) > 0:
                            usedCategory[b] = 1
                            if returnCategory:
                                s.append([ternary1[i][j][0] + ternary2[i][k][0], ((ternary1[i][j][1] & 0x1fff) << 19) | (ternary2[i][k][1] & 0x1fff), ((ternary1[i][j][2] & 0x1fff) << 19) | (ternary2[i][k][2] & 0x1fff), b])
                            else :
                                s.append([ternary1[i][j][0] + ternary2[i][k][0], ((ternary1[i][j][1] & 0x1fff) << 19) | (ternary2[i][k][1] & 0x1fff), ((ternary1[i][j][2] & 0x1fff) << 19) | (ternary2[i][k][2] & 0x1fff), 
                                        [int(num) for num in torch.floor(LUT[i][b]).tolist()]])
        ternary.append(s)
        usedCategoryList.append(usedCategory)
    #[priority, match string1, match string2, mask1, mask2, category number]
    return (ternary, usedCategoryList) if returnCategory else ternary


#ternary is a lookup table for i inputs, ternary[i] corresponds to all matching tuples, ternary[i][j] format is [priority, match string, mask, [parameter list]]
def from_tables_to_table(ternary, bitlength, match = 0, doFuseOuput = True):
    i = 0
    table = []
    if match == 0: # do nothing
        table = ternary
    elif match == 1: # fuse two inputs corresponding to two tuples into one number
        while i < len(ternary) - 1:
            table.append(fuseInput(ternary[i], bitlength, ternary[i+1]))
            i += 2
        if i != len(ternary):
            table.append(ternary[i])
    elif match == 2: # link two inputs corresponding to two tuples as two numbers
        while i < len(ternary) - 1:
            table.append(linkInput(ternary[i], ternary[i+1]))
            i += 2
        if i != len(ternary):
            table.append(ternary[i])
    elif match == 3: # match result is category number, not parameter list
        while i < len(ternary) - 1:
            table.append(fuseInput(ternary[i], bitlength, ternary[i+1], useCategory = True, bitlength2 = 4))
            i += 2
        if i != len(ternary):
            table.append(ternary[i])
    elif match == 4: # link three inputs corresponding to three tuples as three numbers
        while i < len(ternary) - 1:
            table.append(linkInput3(ternary[i], ternary[i+1], ternary[i+2]))
            i += 3
    elif match == 5:
        while i < len(ternary) - 1:
            table.append(fuseInput(ternary[i], bitlength, ternary[i+1], exact = True))
            i += 2
        if i != len(ternary):
            table.append(ternary[i])
    if doFuseOuput:
        table = fuseOuput(table, bitlength)
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
def linkInput(table1, table2):
    table = []
    for i in range(len(table1)):
        for j in range(len(table2)):
            parameter = []
            for k in range(len(table1[i][-1])):
                parameter.append(table1[i][-1][k] + table2[j][-1][k])
            table.append([table1[i][0] + table2[j][0], table1[i][1], table2[j][1], table1[i][2], table2[j][2], parameter])
    return table
def linkInput3(table1, table2, table3, SRAM = True):
    table = []
    for i in range(len(table1)):
        for j in range(len(table2)):
            for l in range(len(table3)):
                parameter = []
                for k in range(len(table1[i][-1])):
                    parameter.append(table1[i][-1][k] + table2[j][-1][k] + table3[l][-1][k])
                if SRAM :
                    table.append([table1[i][0], table2[j][0], table3[l][0], parameter])
    return table
def fuseOuput(table1, bitlength2):
    for i in range(len(table1)):
        for k in range(len(table1[i])):
            table = []
            for j in range(len(table1[i][k][-1]) // 2):
                table.append(((table1[i][k][-1][2*j] & 0x1fff) << bitlength2) | (table1[i][k][-1][2*j+1] & 0x1fff))
            table1[i][k][-1] = table
    return table1
def from_LUT_to_exact(table1, input):
    tables = []
    for i in range(len(table1)):
        table = []
        for j in range(16):
            if input[i][j] == 1:
                table.append([j, [int(num) for num in torch.floor(table1[i][j]).tolist()]])
        tables.append(table) 
    return tables
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

bitLengthOfFeature = [8,16,8,8,16,32,16,32,32,32]
T1 = model['MM1.T']
H = model['MM1.H']
LUT1 = model['MM1.LUT']
tt1 = from_T_H_to_range(T1, H) #;print('tt1', tt1)
table1 = from_range_to_ternary(tt1, bitLengthOfFeature, LUT1, signed = 0, reservedOperation = 5)
S2 = model['MM2.S']
T2 = model['MM2.T']
H = model['MM2.H']
LUT2 = model['MM2.LUT']
tt2_1 ,tt2_2 = from_S_T_H_to_range2(S2, T2, H);print(tt2_1[0], tt2_2[0])
table2_0_1 = from_range2_to_ternary2(tt2_1[0:1], tt2_2[0:1], 13, 13, LUT2[0:1], signed = 1, returnCategory = False, reservedOperation = -1)
table2_1_16, usedCategory = from_range2_to_ternary2(tt2_1[1:16], tt2_2[1:16], 13, 13, LUT2[1:15], signed = 1, returnCategory = True, reservedOperation = 0)
T3 = model['MM3.T']
H = model['MM3.H']
LUT3 = model['MM3.LUT']
tt3 = from_T_H_to_range(T3, H)
table3 = from_range_to_ternary(tt3, 13, LUT3)
table1 = from_tables_to_table(table1, 19, match = 2)
table2_0_1 = from_tables_to_table(table2_0_1, 19, match = 0, doFuseOuput = True)
table2_3 = from_LUT_to_exact(LUT2[1:16], usedCategory)
table2_3 = from_tables_to_table(table2_3, 4, 0, doFuseOuput = False)
table2_3 = from_tables_to_table(table2_3, 19, 4, doFuseOuput = True)
table3 = from_tables_to_table(table3, 19, match = 1, doFuseOuput = False)


# Model pkl conversion path
# with open('/home/yang1210/THU_net/BMV2_Pegasus/make_dataset/mlp/CICIOT2022/mlp_CICIOT.pkl','wb') as f:
#     pickle.dump([table1, table2_0_1, table2_1_16, table2_3, table3], f, protocol = 2)

# with open('/home/yang1210/THU_net/BMV2_Pegasus/make_dataset/mlp/PeerRush/mlp_PeerRush.pkl','wb') as f:
#     pickle.dump([table1, table2_0_1, table2_1_16, table2_3, table3], f, protocol = 2)

# with open('/home/yang1210/THU_net/BMV2_Pegasus/make_dataset/mlp/ISCXVPN/mlp_ISCXVPN.pkl','wb') as f:
#     pickle.dump([table1, table2_0_1, table2_1_16, table2_3, table3], f, protocol = 2)



##Test section
def get_parameter_from_ternary(ternary, k, b):
    for j in range(len(ternary[k])):
        if b & ternary[k][j][2] == ternary[k][j][1]:
            #print(ternary[k][j])
            return ternary[k][j][3]
    print('No Matching! code = 1')
    return ternary[k][j][3]
def match1(ternary, k, a, b):
    for j in range(len(ternary[k])):
        if a & ternary[k][j][3] == ternary[k][j][1] and b & ternary[k][j][4] == ternary[k][j][2]:
            #print(ternary[k][j])
            return ternary[k][j][5]
    print('No Matching! code = 2')
    return ternary[k][j][5]
def match_with_exact(ternary, k, a, b, c):
    for j in range(len(ternary[k])):
        if a == ternary[k][j][0] and b == ternary[k][j][1] and c == ternary[k][j][2]:
            return ternary[k][j][3]
    print('No Matching! code = 3')
    return ternary[k][j][3]



# Open CSV file
with open('/home/yang1210/THU_net/project/software/mlpdataset/CICIOT2022_test.csv', mode='r', encoding='utf-8') as file:
    # Create a CSV reader
    data = pd.read_csv(file)
    # First column is labels


# with open('/home/yang1210/THU_net/project/software/mlpdataset/PeerRush_test.csv', mode='r', encoding='utf-8') as file:
#     # Create a CSV reader
#     data = pd.read_csv(file)
    # First column is labels

# with open('/home/yang1210/THU_net/project/software/mlpdataset/ISCXVPN_test.csv', mode='r', encoding='utf-8') as file:
#     # Create a CSV reader
#     data = pd.read_csv(file)
# #     # First column is labels

# with open('/home/yang1210/THU_net/software_all/test.csv', mode='r', encoding='utf-8') as file:
#     # Create a CSV reader
#     data = pd.read_csv(file)
#     # First column is labels



labels = torch.tensor(data.iloc[:, 0].values, dtype=torch.long)
features = torch.tensor(data.iloc[:, 1:].values, dtype=torch.float32)
features = features.long()
features = features.tolist()
labels = labels.tolist()
# with open('/home/yang1210/THU_net/BMV2_Pegasus/make_dataset/mlp/CICIOT2022/CICIOT2022_dataset.pkl','wb') as f:
#     pickle.dump([features, labels], f, protocol = 2)

# with open('/home/yang1210/THU_net/BMV2_Pegasus/make_dataset/mlp/PeerRush/PeerRush_dataset.pkl','wb') as f:
#     pickle.dump([features, labels], f, protocol = 2)

# with open('/home/yang1210/THU_net/BMV2_Pegasus/make_dataset/mlp/ISCXVPN/ISCXVPN_dataset.pkl','wb') as f:
#     pickle.dump([features, labels], f, protocol = 2)

t = 0 ; tt = 0 
n = 0
fault = 0
pred = []
all_label = []
    
##Simulation after fuse
for i in range(len(features)):
    at = []
    for j in range(len(features[i]) // 2):
        at.append(match1(table1, j, features[i][2*j], features[i][2*j+1]))
    if len(features[i]) % 2 == 1:
        at.append(get_parameter_from_ternary(table1, len(features[i]) // 2, features[i][len(features[i]) - 1]))
    at = lists_add(at)
    # print('at')
    # print(at)
    bt = []
    for j in range(1):
        bt.append(get_parameter_from_ternary(table2_0_1, j, at[j]))
    #print('bt1')
    bb = [];#
    for j in range(1, 16): 
        bb.append(get_parameter_from_ternary(table2_1_16, j-1, at[j]))
    #print('bb')
    for j in range(5):
        bt.append(match_with_exact(table2_3, j, bb[3*j+0], bb[3*j+1], bb[3*j+2]))
    bt = lists_add(bt)
    #print('bt2')
    print(len(bt))
    ct = []
    for j in range(len(bt)):
        ct.append(get_parameter_from_ternary(table3, j, bt[j]))
    ct = lists_add(ct)
    print('ct')
    print(ct)



    label = 0
    max = ct[0]
    if ct[1] > max:
        label = 1
        max = ct[1]
    if ct[2] > max:
        label = 2
    d = (label == labels[i])
    if d == True:
        tt += 1
    n += 1
    if n % 128 == 0:
        print(tt, n, tt/n)

    # for ISCXVPN
    # label = 0
    # max = ct[0]
    # if ct[1] > max:
    #     label = 1
    #     max = ct[1]
    # if ct[2] > max:
    #     label = 2
    #     max = ct[2]
    # if ct[3] > max:
    #     label = 3
    #     max = ct[3]
    # if ct[4] > max:
    #     label = 4
    #     max = ct[4]
    # if ct[5] > max:
    #     label = 5
    # d = (label == labels[i])
    # if d == True:
    #     tt += 1
    # n += 1
    # if n % 128 == 0:
    #     print(tt, n, tt/n)

    pred.append(label)
    all_label.append(labels[i])

f1_macro = f1_score(all_label, pred, average='macro')
print(fault)
print(tt, n, tt/n)
print(f"F1 Macro: {f1_macro:.4f}")
print(f"Precision: {tt/n:.4f}")



