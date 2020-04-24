# ======================
# exampler commands on miniImageNet
# ======================

# supervised pre-training
python train_supervised.py --trial pretrain --model_path /path/to/save --tb_path /path/to/tensorboard --data_root /path/to/data_root

# distillation
python train_distillation.py -r 0.5 -a 1.0 --path_t /path/to/teacher.pth --trial born1 --model_path /path/to/save --tb_path /path/to/tensorboard --data_root /path/to/data_root

# evaluation
python eval_fewshot.py --model_path /path/to/student.pth --data_root /path/to/data_root


## train
python train_supervised.py --dataset miniImageNet --model resnet12 --learning_rate 0.1  --trial debug --use_gpu 1 --num_workers -1

## train for moco
python train_moco.py --dataset miniImageNet --model resnet12 --learning_rate 0.03 --use_gpu 1 --dist-url 77 --lr_decay_epochs 60,80,120,160 --epochs 200 --mlp --moco-dim 128 --moco-k 65536 --lamda 0.5
python train_moco.py --dataset miniImageNet --model resnet12 --learning_rate 0.03 --use_gpu 0 --dist-url 78 --lr_decay_epochs 60,80,120,160 --epochs 200 --mlp --moco-dim 128 --moco-k 16384 --lamda 0.5