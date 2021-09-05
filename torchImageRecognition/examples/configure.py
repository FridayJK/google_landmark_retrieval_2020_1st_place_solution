
import argparse

def get_arguments():
    args = argparse.ArgumentParser()

    args.add_argument('--net_id', type=int, default=0, help="efficientnet id num")
    args.add_argument('--input_size', type=list, default=[512,512], help="input size of net")
    args.add_argument('--embdding_size', type=int, default=512)
    args.add_argument('--num_class', type=int, default=81313)
    args.add_argument('--data_argument', type=bool, default=False)

    args.add_argument('--from_scratch', type=bool, default=False)
    args.add_argument('--pre_trained_weights', type=str, default="")
    args.add_argument('--loss_type', type=str, default="softmax", choices=['softmax', 'adacos'])
    args.add_argument('--scheduler', type=str, default="stepLR", choices=['stepLR', 'multiStepLR'])
    args.add_argument('--lr', type=float, default=1e-4)
    args.add_argument('--epochs', type=int, default=50)
    args.add_argument('--batch_per_gpu', type=int, default=50)

    args.add_argument('--data_path', type=str, default="")
    args.add_argument('--data_list', type=str, default="")
    args.add_argument('--data_list_val', type=str, default="")
    args.add_argument('--model_save_path', type=str, default="")
    
    args.add_argument('--work_mode', type=str, default="train", choices=['train', 'test'])

    return args.parse_args()