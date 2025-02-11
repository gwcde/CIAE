import os
import argparse
import pandas as pd
from GBDT_new_train import Cat_train,Cat_train_ex
from MBT_new_train import FT_train,FT_train_ex

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    # gpu_id
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    # disease
    parser.add_argument('-d', '--disease', type=str, default='EW-T2D', help='disease')
    # feature
    parser.add_argument('-f', '--feature', type=str, default='KO', help='feature')
    # using config
    # parser.add_argument('-uc', '--use_config', action='store_true', help='using config')

    # model type
    parser.add_argument('-m', '--model_type', type=str, default='CatBoost', help='model type')

    # FT-Transformer params
    # batch size
    parser.add_argument('-bs', '--batch_size', type=int, default=4, help='batch size')
    # learning rate
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='learning rate')
    # n_blocks
    parser.add_argument('-nb', '--n_blocks', type=int, default=2, help='number of blocks')

    #CatBoost params
    #depth
    parser.add_argument('-dp', '--depth', type=int, default=6, help='depth')
    #colsample_bylevel
    parser.add_argument('-cb', '--colsample_bylevel', type=float, default=0.9, help='colsample_bylevel')

    #data
    parser.add_argument('-in', '--in_file', type=str, default='./data/EW_KO_95.csv', help='in_file')

    parser.add_argument('-ot', '--out_file', type=str, default='./result/KO/EW_T2D_KO_Cat.csv', help='out_file')

    parser.add_argument('-uc', '--use_config', action='store_true', help='using explanations')
    parser.add_argument('-ex', '--ex_file', type=str, default='./result/KO/EW-T2D/EW_weight_KO.csv', help='explanations')

    args = parser.parse_args()

    if(args.disease not in ['EW-T2D','C-T2D']):raise argparse.ArgumentTypeError(f'{args.disease}'+ ":error disease!")
    if(args.feature not in ['species','KO']):raise argparse.ArgumentTypeError(f'{args.feature}'+ ":error feature type!")
    if(args.model_type not in ['CatBoost','FT']):raise argparse.ArgumentTypeError(f'{args.model_type}'+ ":error feature type!")

    if args.model_type == "FT":
        paras = {
            'batch_size': args.batch_size,
            'lr': args.learning_rate,
            'n_blocks': args.n_blocks
        }
    elif args.model_type == "CatBoost":
        paras = {
            'learning_rate': args.learning_rate,
            'depth': args.depth,
            'colsample_bylevel': args.colsample_bylevel
        }

    else:
        assert False, f"{args.model_type} type not supported"

    # gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if(args.use_config):
        if args.model_type != "FT":colums = [f for f in list(pd.read_csv(args.ex_file)) if('cat' in f)]
        else:colums = [f for f in list(pd.read_csv(args.ex_file)) if('cat' not in f)]
        for col in colums:
            for seed in [392, 412, 432, 452, 472]:
                if(args.model_type == "CatBoost"):
                    Cat_train_ex(args.disease, args.feature, seed,args.model_type,args.in_file,args.out_file,args.ex_file,col,100,**paras)
                else:
                    FT_train_ex(args.disease, args.feature, seed, True, args.in_file, args.out_file,args.ex_file,col,100,**paras)
    else:    
        for seed in [392, 412, 432, 452, 472]:
            if(args.model_type == "CatBoost"):
                Cat_train(args.disease, args.feature, seed,args.model_type,args.in_file,args.out_file,**paras)
            else:
                FT_train(args.disease, args.feature, seed, True, args.in_file, args.out_file,**paras)




