import os
import argparse
from GBDT_new_train import Cat_train
from MBT_new_train import FT_train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    # gpu_id
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    # disease
    parser.add_argument('-d', '--disease', type=str, default='EW-T2D', help='disease')
    # feature
    parser.add_argument('-f', '--feature', type=str, default='KO', help='feature')
    # using config
    parser.add_argument('-uc', '--use_config', action='store_true', help='using config')

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
    parser.add_argument('-in', '--in_file', type=str, default='./data/EW_species_abundance.csv', help='in_file')

    parser.add_argument('-ot', '--out_file', type=str, default='./result/EW_species.csv', help='out_file')

    args = parser.parse_args()

    if args.model_type == "FT-Transformer":
        params = {
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

    for seed in [392, 412, 432, 452, 472]:
        if(args.model_type == "CatBoost"):
            Cat_train(args.disease, args.feature, seed, args.use_config,True,args.in_file,args.out_file,**paras)
        else:
            FT_train(args.disease, args.feature, seed,args.use_config, args.model_type, args.in_file, args.out_file,**paras)




