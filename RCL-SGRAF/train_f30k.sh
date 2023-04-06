tau=0.05
batch_size=128
noise_rate=0.2
data_name=f30k_precomp
loss='log'
margin=0.2
data_path=../data
vocab_path=../data/vocab
gpu=0
lr_update=15
num_epochs=30

model_name=runs/$data_name/checkpoint
logger_name=runs/$data_name/log

module_name=SGR
CUDA_VISIBLE_DEVICES=$gpu python train.py --data_name $data_name --num_epochs $num_epochs --loss $loss --lr_update $lr_update --module_name $module_name --log_step 200 --data_path $data_path --vocab_path $vocab_path --model_name $model_name --logger_name $logger_name --noise_rate $noise_rate --margin $margin --tau $tau --batch_size $batch_size

# evaluate SGR
# module_name=SGR
# CUDA_VISIBLE_DEVICES=$gpu python evaluate_model.py --data_name $data_name --num_epochs $num_epochs --loss $loss --lr_update $lr_update --module_name $module_name --log_step 200 --data_path $data_path --vocab_path $vocab_path --model_name $model_name --logger_name $logger_name --noise_rate $noise_rate --margin $margin --tau $tau --batch_size $batch_size

module_name=SAF
CUDA_VISIBLE_DEVICES=$gpu python train.py --data_name $data_name --num_epochs $num_epochs --loss $loss --lr_update $lr_update --module_name $module_name --log_step 200 --data_path ./data --vocab_path ./vocab --model_name $model_name --logger_name $logger_name --noise_rate $noise_rate --margin $margin --tau $tau --batch_size $batch_size

# evaluate SAF
# module_name=SAF
# CUDA_VISIBLE_DEVICES=$gpu python evaluate_model.py --data_name $data_name --num_epochs $num_epochs --loss $loss --lr_update $lr_update --module_name $module_name --log_step 200 --data_path $data_path --vocab_path $vocab_path --model_name $model_name --logger_name $logger_name --noise_rate $noise_rate --margin $margin --tau $tau --batch_size $batch_size

# evaluate SGRAF
module_name=SGRAF
CUDA_VISIBLE_DEVICES=$gpu python evaluate_model.py --data_name $data_name --num_epochs $num_epochs --loss $loss --lr_update $lr_update --module_name $module_name --log_step 200 --data_path $data_path --vocab_path $vocab_path --model_name $model_name --logger_name $logger_name --noise_rate $noise_rate --margin $margin --tau $tau --batch_size $batch_size
