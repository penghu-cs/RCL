tau=0.05
batch_size=128
noise_rate=0.2
module_name=SGR
data_name=coco_precomp
vocab_path=../data/vocab
data_path=../data/
loss='log'
margin=0.2
ratio=-1
gpu=0

model_name=runs/$data_name/checkpoint
logger_name=runs/$data_name/log

module_name=SGR
CUDA_VISIBLE_DEVICES=$gpu python train.py --data_name $data_name --num_epochs 20 --lr_update 10 --module_name $module_name --loss $loss --log_step 200 --data_path $data_pathdata --vocab_path $vocab_path --model_name $model_name --logger_name $logger_name --noise_rate $noise_rate --margin $margin --tau $tau --ratio $ratio

module_name=SAF
CUDA_VISIBLE_DEVICES=$gpu python train.py --data_name $data_name --num_epochs 20 --lr_update 10 --module_name $module_name --loss $loss --log_step 200 --data_path $data_pathdata --vocab_path $vocab_path --model_name $model_name --logger_name $logger_name --noise_rate $noise_rate --margin $margin --tau $tau --ratio $ratio

CUDA_VISIBLE_DEVICES=$gpu python evaluate_model.py --data_name $data_name --num_epochs 20 --lr_update 10 --module_name SGRAF --loss $loss --log_step 200 --data_path $data_path --vocab_path $vocab_path --model_name $model_name --logger_name $logger_name --noise_rate $noise_rate --margin $margin --tau $tau --ratio $ratio

