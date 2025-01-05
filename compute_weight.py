import os
import pandas as pd
import ast
from scipy import stats
import numpy as np
import argparse
import yaml

from GBDT_new_train import Cat_train_ex_2
from MBT_new_train import FT_train_ex_2

def compute_g(x, y_min, y_max):

    x = np.array(x)
    denominator = np.sum(x)
    result = (x / denominator) * (y_max - y_min) + y_min
    return result

def compute_softmax(x):
    """
    Compute the softmax of a list or numpy array x.

    Parameters:
    x (list or numpy array): Input values.

    Returns:
    numpy array: Softmax of the input values.
    """
    e_x = np.exp(x - np.max(x))  # Shift for numerical stability
    return e_x / e_x.sum()

def compute_w(R, P, alpha, y_min, y_max):

    # Compute g(R_i) for each R_i in R
    g_R = [compute_g(r, y_min, y_max) for r in R]
    # Compute the mean of g(R_i)
    mean_g_R = np.mean(g_R, axis=0)
    # Compute g(P)
    g_P = compute_g(P, y_min, y_max)
    # Combine according to the formula
    combined = alpha * mean_g_R + (1 - alpha) * g_P
    # Apply the softmax
    w = compute_softmax(combined)
    return w

def generate_aggregation_results(root_files,dl_ml):
    all_files = os.listdir(root_files)
    all_files = sorted(all_files)
    # print(all_files)
    dfs = [pd.read_csv(os.path.join(root_files,file_name)) for file_name in all_files]
    intersection_results = []
    final_dict = dict()
    avg_final_dict = dict()
    single_final_dict = dict()

    # dfs.append(pd.read_csv(os.path.join(root_files,all_files[-1])))
    for i, df_i in enumerate(dfs):
        for col_i in df_i.columns:
            if('avg' in col_i or 'rank' in col_i or 'Unnamed: 0'==col_i):continue
            avg_tmp = 0
            sp_res = 0
            for j, df_j in enumerate(dfs):
                if i == j:
                    continue
                for col_j in df_j.columns:
                    if(col_j!=col_i):continue
                    intersection = set([list(ast.literal_eval(tmp).keys())[0] for tmp in df_i[col_i][:100]]).intersection(set([list(ast.literal_eval(tmp).keys())[0] for tmp in df_j[col_j][:100]]))
                    first_l = []
                    second_l = []
                    for k in intersection:
                        for kk in df_i[col_i][:100]:
                            if(list(ast.literal_eval(kk).keys())[0]==k):first_l.append(list(ast.literal_eval(kk).values())[0])
                        for kk in df_j[col_j][:100]:
                            if(list(ast.literal_eval(kk).keys())[0]==k):second_l.append(list(ast.literal_eval(kk).values())[0])
                    # print(first_l,"   ",second_l)
                    # print(stats.spearmanr(first_l,second_l).statistic)
                    sp_res += stats.spearmanr(first_l,second_l).statistic
                    avg_tmp+=len(intersection)
            if(i==5):
                avg_final_dict[col_i] = [(avg_tmp/5),sp_res/5]
            else:
                final_dict[str(i)+'_'+col_i] = [(avg_tmp/5),sp_res/5]
                if(col_i not in single_final_dict.keys()):
                    single_final_dict[col_i] = [(avg_tmp/5),sp_res/5]
                else:
                    tmp_0 = single_final_dict[col_i][0]
                    tmp_1 = single_final_dict[col_i][1]
                    if(tmp_0<avg_tmp/5):single_final_dict[col_i] = [avg_tmp/5,sp_res/5]

    # print(avg_final_dict)
    res_1 = []
    res_2 = []
    for _,v in avg_final_dict.items():
        res_1.append(v[0])
        res_2.append(v[1])
    if('ft' in dl_ml):return [res_1[int(len(res_1)/2):len(res_1)],res_2[int(len(res_2)/2):len(res_2)]]
    else:return [res_1[0:int(len(res_1)/2)],res_2[0:int(len(res_2)/2)]]

def z_score_standardization(list_data):
    """
    Z-score
    """
    data = []
    for list_tmp in list_data:
        list_data_tmp = []
        for val in list_tmp:
            list_data_tmp.append(list(ast.literal_eval(val).values())[0])
        data.append(list_data_tmp)
    data = np.array(data).T
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    res =  (data - mean) / std
    return res.T

def save_weighted_KO(out_file,ex_file,weight_list,mode_type):

    T2D_pd = pd.read_csv(ex_file)
    rank_c_2 = []
    list_type = []
    list_type.append(list(T2D_pd['cat_importance']) if(mode_type=='CatBoost') else list(T2D_pd['ft_igs']))
    if(mode_type=='CatBoost'):list_type.extend([list(T2D_pd['cat_lime']),list(T2D_pd['cat_shap'])])
    else:list_type.extend([list(T2D_pd['ft_saliency']),list(T2D_pd['ft_shap'])])
    res = z_score_standardization(list_type)
    for i,val in enumerate(list_type[0]):
        weight_tmp = ast.literal_eval(val)
        KO = list(weight_tmp.keys())[0]
        # weight_value = weight_list[0]*list(weight_tmp.values())[0]
        weight_value = weight_list[0]*res[0][i]
        sum_weight = weight_value
        for j,tmp_inpret in enumerate(list_type[1:]):
            begin_0 = 0
            for k,res_tmp in enumerate(tmp_inpret):
                if(list(ast.literal_eval(res_tmp).values())[0]==0 and begin_0==0):begin_0 = k
                if(KO==list(ast.literal_eval(res_tmp).keys())[0]):
                    # sum_weight = sum_weight+weight_list[j+1]*list(ast.literal_eval(res_tmp).values())[0]
                    sum_weight = sum_weight+weight_list[j+1]*res[j][k]
                    break
        rank_c_2.append({KO:sum_weight/3})
        rank_c_2 = sorted(rank_c_2, key=lambda x: list(x.values())[0], reverse=True)
    if(mode_type=='CatBoost'):T2D_pd['cat_CIAE'] = rank_c_2
    else:T2D_pd['ft_CIAE'] = rank_c_2
    T2D_pd.to_csv(out_file,index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process input file paths and output locations.')
    parser.add_argument('-d', '--disease', type=str, default='EW-T2D', help='disease')
    parser.add_argument('-in', '--in_file', type=str, default='./data/EW_KO_95.csv', help='in_file')
    parser.add_argument('--config_file', type=str, required=True, help='Path to the configuration file.')
    parser.add_argument('--root_files', type=str, required=True, help='Path to the root_file.')
    parser.add_argument('--explanation_file', type=str, required=True, help='Path to the explanation_file.')
    parser.add_argument('-m', '--model_type', type=str, default='CatBoost', help='model type')
    parser.add_argument('-a', '--alpha', type=float, default=0.5, help='float type alpha')
    parser.add_argument('-mi', '--y_min', type=float, default=0, help='float type y_min')
    parser.add_argument('-ma', '--y_max', type=float, default=1, help='float type y_max')
    parser.add_argument('-wk', '--CIAE_weighted_KO', type=str, default='./result/KO/EW_CIAE_weighted_KO_Cat.csv', help='out_file')
    args = parser.parse_args()
    root_files = args.root_files
    input_file = args.explanation_file
    config_file = args.config_file
    if(args.model_type not in ['CatBoost','FT']):raise argparse.ArgumentTypeError(f'{args.model_type}'+ ":error feature type!")
    else:
        dl_ml = 'cat' if args.model_type=='CatBoost' else 'ft'
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    paras_cat_C = config["paras_cat_C"]
    paras_cat_EW = config["paras_cat_EW"]
    paras_ft_C = config["paras_ft_C"]
    paras_ft_EW = config["paras_ft_EW"]
    R_list = generate_aggregation_results(root_files=root_files,dl_ml=dl_ml)
    if(args.disease not in ['EW-T2D','C-T2D']):raise argparse.ArgumentTypeError(f'{args.disease}'+ ":error disease!")
    if args.model_type == "FT":
        if(args.disease=='EW-T2D'):
            paras = {
                'batch_size': paras_ft_EW['batch_size'],
                'lr': paras_ft_EW['lr'],
                'n_blocks': paras_ft_EW['n_blocks']
            }
        else: 
            paras = {
                'batch_size': paras_ft_C['batch_size'],
                'lr': paras_ft_C['lr'],
                'n_blocks': paras_ft_C['n_blocks']
            }
    elif args.model_type == "CatBoost":
        if(args.disease=='EW-T2D'):
            paras = {
                'learning_rate': paras_cat_EW['learning_rate'],
                'depth': paras_cat_EW['depth'],
                'colsample_bylevel': paras_cat_EW['colsample_bylevel']
            }
        else:
            paras = {
                'learning_rate': paras_cat_C['learning_rate'],
                'depth': paras_cat_C['depth'],
                'colsample_bylevel': paras_cat_C['colsample_bylevel']
            }     

    if args.model_type != "FT":colums = [f for f in list(pd.read_csv(input_file)) if('cat' in f)]
    else:colums = [f for f in list(pd.read_csv(input_file)) if('cat' not in f)]

    P_list = list()
    for col in colums:
        p_tmp_list = list()
        for seed in [392, 412, 432, 452, 472]:
            if(args.model_type == "CatBoost"):
                res = Cat_train_ex_2(args.disease, seed,args.model_type,args.in_file,input_file,col,100,**paras)
            else:
                res = FT_train_ex_2(args.disease, seed, True, args.in_file,input_file,col,100,**paras)
            p_tmp_list.append(res['AUC'])
        P_list.append(np.mean(p_tmp_list))
    # print(R_list)
    # print(P_list)
    W_list = compute_w(R_list,P_list,alpha=args.alpha,y_min=args.y_min,y_max=args.y_max)
    print(W_list)
    save_weighted_KO(args.CIAE_weighted_KO,args.explanation_file,W_list,args.model_type)