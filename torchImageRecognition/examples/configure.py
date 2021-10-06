
import argparse

def get_arguments():
    args = argparse.ArgumentParser()

    args.add_argument('--net_id', type=int, default=0, help="efficientnet id num")
    args.add_argument('--input_size', nargs="+", default=[512,512], help="input size of net")
    args.add_argument('--embdding_size', type=int, default=512)
    args.add_argument('--num_class', type=int, default=81313)
    args.add_argument('--data_argument', type=bool, default=False)

    args.add_argument('--from_scratch', type=bool, default=False)
    args.add_argument('--pre_trained_weights_path', type=str, default="pre_trained_weights/EfficientNet_pytorch/")
    args.add_argument('--loss_type', type=str, default="adacos", choices=['softmax', 'adacos'])
    args.add_argument('--use_LossWeight', action="store_true", help="useLossWeight or not")
    args.add_argument('--scheduler', type=str, default="stepLR", choices=['stepLR', 'multiStepLR', "CosineAnnealing"])
    args.add_argument('--stepLR_step', type=int, default=10)
    args.add_argument('--multiStepLR_step', nargs="+", default=[10, 25])

    args.add_argument('--lr', type=float, default=0.01)
    args.add_argument('--min_lr', type=float, default=1e-5)
    args.add_argument('--momentum', type=float, default=0.9)
    args.add_argument('--weight_decay', type=float, default=1e-5)
    args.add_argument('--epochs', type=int, default=50)
    args.add_argument('--batch_size', type=int, default=16)
    args.add_argument('--mean', nargs="+", default=[0.485, 0.456, 0.406])
    args.add_argument('--std', nargs="+", default=[0.229, 0.224, 0.225])

    args.add_argument('--root_path', type=str, default="/workspace/mnt/storage/zhangjunkang/gldv1/gldv2/")
    args.add_argument('--data_list', type=str, default="dataTrain_stage1.txt")
    args.add_argument('--data_list_val', type=str, default="dataVal_stage1.txt")
    args.add_argument('--model_save_path', type=str, default="/workspace/mnt/storage/zhangjunkang/gldv2/model/pytorch/")
    
    args.add_argument('--work_mode', type=str, default="train", choices=['train', 'test'])
    args.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    args.add_argument('--nprocs', default=1, type=int, help='number of workers')
    args.add_argument('--init_method', default='tcp://127.0.0.1:23456', help="init-method")

    args.add_argument('--train_note', type=str, default="test1")

    return args.parse_args()