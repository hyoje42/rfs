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
python train_supervised.py --dataset miniImageNet --save_freq 1 --learning_rate 0.1 --model resnet12 --trial debug --use_gpu 1 --num_workers -1