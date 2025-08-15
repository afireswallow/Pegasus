import torch
import numpy as np
from torch.quantization import QuantStub, DeQuantStub, prepare_qat, convert
import copy
import torch.nn as nn




def get_data(model, data, weight):
    tempmodel = model
    tempmodel.load_state_dict(weight)

    combined_output = None
    n_element = None
    """
    torch.backends.quantized.engine = 'fbgemm'
    prepare_qat(tempmodel, inplace=True)
    for xbatch, labelbatch in data:
        _, output = tempmodel(xbatch)
    convert(tempmodel, inplace=True)
    """
    model.eval()

    for xbatch, labelbatch in data:

        _, output = tempmodel(xbatch)
        output[0] = output[0].to(torch.device('cpu'))

        if combined_output is None:
            n_element = len(output)
            combined_output = [[] for _ in range(n_element)]
        
        # Iterate over each element in the output and store it in the corresponding list
        for i in range(n_element):
            combined_output[i].append(output[i].detach().numpy())
    
    # Concatenate the collected batches for each element into a single tensor
    final_output = [np.concatenate(combined_output[i], axis=0) for i in range(n_element)]

    
    return final_output
