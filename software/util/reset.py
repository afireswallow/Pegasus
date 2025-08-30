
import torch
import numpy as np
import torch.nn.functional as F


def adjust_norm_layer(weight, bias, running_var, running_mean):
    """Adjusts the weight and bias of a normalization layer."""
    weight = weight.cpu()
    bias = bias.cpu()
    running_var = running_var.cpu()
    running_mean = running_mean.cpu()
    adjusted_weight = weight / torch.sqrt(running_var + 1e-5)
    adjusted_bias = bias - running_mean * adjusted_weight
    return adjusted_weight, adjusted_bias


def reset_fc(weight, bias, LUT, device):
    C = LUT.shape[0]
    K = LUT.shape[1]
    D = LUT.shape[2]
    output_size = weight.shape[0]
    weight = weight.to(device)
    LUT = LUT.to(device)
    LUT_new = torch.randn(C, 16, output_size).to(device)
    for i in range(C):
        LUT_new[i] = torch.einsum('kd, do->ko', LUT[i], weight[:, i * D: (i + 1) * D].T)
    bias = bias.to(device)
    LUT_new[0, :, :] += bias

    return LUT_new


def reset_norm_to_T(T, S, weight, bias):
    assert T.shape[0] == S.shape[0]
    weight = weight.cpu()
    bias = bias.cpu()
    S = S.cpu()
    T = T.cpu()
    C = S.shape[0]
    D = S.shape[1]
    K = S.shape[2]

    for i in range(C):
        for j in range(D):
            for k in range(K):
                if S[i, j, k] == 1:

                    T[i, k] = T[i, k] - bias[i * D + j]
                    T[i, k] = T[i, k]/ weight[i * D + j]

    return T



def reset_norm_to_LUT(weight, bias, LUT,device):
    LUT_new = torch.randn(LUT.shape).to(device)
    weight = weight.to(device)
    bias = bias.to(device)
    LUT_new = torch.einsum('cko,o->cko', LUT, weight)
    LUT_new[0, :, :] += bias
    return LUT_new


def reset_relu(T):
    T[T < 0] = -100
    return T






def reset_int(X, scalar):
    return (X * scalar).floor()

def reset_int_double(LUT, T,sn):
    maxx = torch.max(torch.max(torch.abs(LUT)), torch.max(torch.abs(T)))

    boundary = (2 ** sn) - 1
    scalar = boundary/maxx
    return reset_int(LUT, scalar), reset_int(T, scalar)

def reset_int_single(X,sn):
    maxx = torch.max(torch.abs(X))
    boundary = (2 ** sn) - 1
    scalar = boundary/maxx
    return reset_int(X, scalar)



def reset_int_quadruple(LUT1_1, LUT1_2, LUT2, T,device,sn):
    finite_T = T[torch.isfinite(T)].to(device)
    LUT1_1 = LUT1_1.to(device)
    LUT1_2 = LUT1_2.to(device)
    LUT2 = LUT2.to(device)
    T = T.to(device)
    maxx = torch.max(
        torch.max(
                    torch.max(torch.abs(LUT1_1)),
                    torch.max(torch.abs(LUT1_2))
                    ),
        torch.max(
                    torch.max(torch.abs(LUT2)),
                    torch.max(torch.abs(finite_T)) if finite_T.numel() > 0 else torch.tensor(0.0)
                    )
    )
    boundary = (2 ** sn) - 1
    scalar = 1
    while maxx * scalar * 2 < boundary:
        scalar = scalar * 2
    return reset_int(LUT1_1, scalar), reset_int(LUT1_2, scalar),reset_int(LUT2, scalar), reset_int(T, scalar)

def reset_int_double_inf(LUT, T,sn,device):
    finite_T = T[torch.isfinite(T)].to(device)
    LUT = LUT.to(device)
    maxx = torch.max(
        torch.max(LUT), 
        torch.max(torch.abs(finite_T)) if finite_T.numel() > 0 else torch.tensor(0.0)
        )
    boundary = (2 ** sn) - 1
    scalar = 1
    while maxx * scalar * 2 < boundary:
        scalar = scalar * 2
    return reset_int(LUT, scalar), reset_int(T, scalar)





def reset_int_triple_l(LUT1_1, LUT1_2, T,device,sn):
    LUT1_1 = LUT1_1.to(device)
    LUT1_2 = LUT1_2.to(device)
    T = T.to(device)
    maxx = torch.max(
        torch.max(
                    torch.max(torch.abs(LUT1_1)),
                    torch.max(torch.abs(LUT1_2))
                    ),
        torch.max(torch.abs(T))
    )
    boundary = (2 ** sn) - 1
    scalar = boundary/maxx
    return reset_int(LUT1_1, scalar), reset_int(LUT1_2, scalar),reset_int(T, scalar)

def reset_int_triple_t(LUT, T1, T2,device,sn):
    LUT = LUT.to(device)
    T1 = T1.to(device)
    T2 = T2.to(device)
    finite_T1 = T1[torch.isfinite(T1)].to(device)
    finite_T2 = T2[torch.isfinite(T2)].to(device)
    maxx = torch.max(
        torch.max(
                    torch.max(torch.abs(finite_T1)) if finite_T1.numel() > 0 else torch.tensor(0.0),
                    torch.max(torch.abs(finite_T2)) if finite_T2.numel() > 0 else torch.tensor(0.0)
                    ),
        torch.max(torch.abs(LUT))
    )
    boundary = (2 ** sn) - 1
    scalar = boundary/maxx
    return reset_int(LUT, scalar), reset_int(T1, scalar),reset_int(T2, scalar)



def reset_norm_to_T_(T, S, weight, bias):
    assert T.shape[0] == S.shape[0]
    weight = weight.cpu()
    bias = bias.cpu()
    S = S.cpu()
    T = T.cpu()
    C = S.shape[0]
    D = S.shape[1]
    K = S.shape[2]
    for i in range(C):
        for j in range(D):
            for k in range(K):
                if S[i, j, k] == 1:
                    T[i, k] = (T[i, k] - bias[i * D + j]) / weight[i * D + j]
                    if i == 8 or i == 9 or i == 10:
                        T[i, k] = T[i, k]*10000
    return T






def reset_rnn_threshold(T,device):
    T[T>=1] = 0.99999999
    T[T<=-1] = -0.99999999
    T = torch.atanh(T)
    return T




def reset_embeding_LUT(embdweight1,embdweight2,fcweight,fcbias,device):
    fcweight1 = fcweight[:,:embdweight1.shape[1]]
    fcweight2 = fcweight[:,embdweight1.shape[1]:]
    lookup_table1 = torch.zeros(embdweight1.shape[0], fcweight.shape[0]).to(device)
    lookup_table2 = torch.zeros(embdweight2.shape[0], fcweight.shape[0]).to(device)
    lookup_table1 = torch.einsum('ab,bc->ac', embdweight1, fcweight1.T)+fcbias
    lookup_table2 = torch.einsum('ab,bc->ac', embdweight2, fcweight2.T)
    return lookup_table1,lookup_table2



def kernel_to_toeplitz(kernel, input_shape):
    out_channels, in_channels, kernel_height, kernel_width = kernel.shape
    in_channels, input_height, input_width = input_shape

    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1

    toeplitz_height = output_height * output_width
    toeplitz_width = input_height * input_width * in_channels
    toeplitz_matrix = torch.zeros((out_channels, toeplitz_height, toeplitz_width))

    for out_c in range(out_channels):
        for in_c in range(in_channels):
            current_kernel = kernel[out_c, in_c]
            for i in range(output_height):
                for j in range(output_width):
                    start_row = i
                    end_row = i + kernel_height
                    start_col = j
                    end_col = j + kernel_width
                    kernel_flatten = current_kernel.view(-1)
                    row_index = i * output_width + j
                    for k, value in enumerate(kernel_flatten):
                        input_row = start_row + k // kernel_width
                        input_col = start_col + k % kernel_width
                        input_index = in_c * input_height * input_width + input_row * input_width + input_col
                        toeplitz_matrix[out_c, row_index, input_index] = value

    return toeplitz_matrix

def reset_conv_LUT(LUT,kernel, input_shape, convbias, device):
    toeplitz_matrix = kernel_to_toeplitz(kernel, input_shape)
    toeplitz_matrix = toeplitz_matrix.to(device)
    C = LUT.shape[0]
    K = LUT.shape[1]
    D = LUT.shape[2]
    lutlist = []
    print(toeplitz_matrix.shape)
    for i in range(kernel.shape[0]):
        weight = toeplitz_matrix[i]

        bias = convbias[i].repeat(toeplitz_matrix.shape[1])
        LUT_new = reset_fc(weight, bias, LUT, device)

        lutlist.append(LUT_new)
    
    convLUT = torch.stack(lutlist, dim=-1)
    print(convLUT.shape)


    return lutlist
