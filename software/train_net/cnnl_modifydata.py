import numpy as np
import pandas as pd
import argparse

def process_and_save(a_path, b_path, c_path, save_path):
    A = pd.read_csv(a_path, header=None).values 
    B = pd.read_csv(b_path, header=None).values  
    C = pd.read_csv(c_path, header=None).values 

    assert A.shape[0] == B.shape[0], "mismatch in A and B row counts"
    assert A.shape[0] % 8 == 0, "the number of rows in A must be a multiple of 8"
    datasize = A.shape[0] // 8
    assert C.shape[0] == datasize, "mismatch in C row count and A/B derived datasize"


    D = np.concatenate([A, B], axis=1)  

    D_prime = D.reshape(datasize, -1) 

    E = np.concatenate([C, D_prime], axis=1)  
    pd.DataFrame(E).to_csv(save_path, index=False, header=False)





parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--dataset",type=str,required=True)
parser.add_argument("--mode",type=str,required=True,choices=["train", "test", "val"])
parser.add_argument("--w_pth",type=str,required=True)
parser.add_argument("--b_pth",type=str,required=True)
parser.add_argument("--label_pth",type=str,required=True)
parser.add_argument("--save_pth",type=str,required=True)
args = parser.parse_args()

if __name__ == '__main__':
    w_path = args.w_pth
    b_path = args.b_pth
    label_path = args.label_pth
    save_path = args.save_pth
    save_path = save_path +"/"+ args.dataset + "_quantilinear_"+args.mode+".csv"
    print(save_path)
    process_and_save(w_path, b_path, label_path, save_path)

