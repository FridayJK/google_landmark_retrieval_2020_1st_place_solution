python -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=1 --node_rank=0 \
--master_addr="127.0.0.1" --master_port=23456 \
./torchImageRecognition/examples/train_demo.py