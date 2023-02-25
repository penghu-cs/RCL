tau=0
# lr=0.0004
lr=0.0002
batch_size=128
noise_rate=0
module_name=SAF
data_name=cc152k_precomp
model_name=runs_final/cc152k/checkpoint
logger_name=runs_final/cc152k/log
lr_update=20
num_epochs=40
# data_path=~/hupeng/data/data
# vocab_path=~/hupeng/data/vocab
data_path=../SCAN/data/data
vocab_path=../SCAN/data/vocab
loss='log'
python train.py --data_name $data_name --num_epochs $num_epochs --loss $loss --lr_update $lr_update --module_name $module_name --log_step 200 --data_path $data_path --vocab_path $vocab_path --model_name $model_name --logger_name $logger_name --noise_rate $noise_rate --margin 0.2 --tau $tau --alpha 0. --learning_rate $lr --batch_size $batch_size

# python train.py --data_name $data_name --num_epochs $num_epochs --loss $loss --lr_update $lr_update --module_name SAF --log_step 200 --data_path ../SCAN/data/data --vocab_path ../SCAN/data/vocab --model_name $model_name --logger_name $logger_name --noise_rate $noise_rate --margin 0.2 --tau $tau --alpha 0. --learning_rate $lr --batch_size $batch_size
# python evaluate_model.py --data_name $data_name --num_epochs $num_epochs --loss $loss --lr_update $lr_update --module_name SGR --log_step 200 --data_path $data_path --vocab_path $vocab_path --model_name $model_name --logger_name $logger_name --noise_rate $noise_rate --margin 0.2 --tau $tau --alpha 0. --learning_rate $lr --batch_size $batch_size
python evaluate_model.py --data_name $data_name --num_epochs $num_epochs --loss $loss --lr_update $lr_update --module_name SGRAF --log_step 200 --data_path $data_path --vocab_path $vocab_path --model_name $model_name --logger_name $logger_name --noise_rate $noise_rate --margin 0.2 --tau $tau --alpha 0. --learning_rate $lr --batch_size $batch_size

#python evaluate_model.py --data_name f30k_precomp --num_epochs 20 --lr_update 10 --module_name SGR --log_step 200 --data_path ../SCAN/data/data --vocab_path ../SCAN/data/vocab --model_name runs_final/f30k/checkpoint --logger_name runs_final/f30k/log --noise_rate 0.2 --margin 0.6 --tau 0.1 --alpha 0. --learning_rate 0.0005
