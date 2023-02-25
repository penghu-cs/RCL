tau=0.05
batch_size=128
noise_rate=0.2
data_name=coco_precomp
loss='log'
margin=0.2
lambda_softmax=10
num_epochs=20
lr_update=5
embed_size=1024
batch_size=128
data_path=../data/
vocab_path=../data/vocab
learning_rate=.0005
gpu=1
model_name=runs/$data_name/checkpoint
logger_name=runs/$data_name/log

# GSMN-sparse
CUDA_VISIBLE_DEVICES=$gpu python train.py --data_path $data_path --data_name $data_name --vocab_path $vocab_path --logger_name $logger_name --model_name $model_name --bi_gru --max_violation --lambda_softmax=$lambda_softmax --num_epochs=$num_epochs --lr_update=$lr_update --learning_rate=$learning_rate --embed_size=$embed_size --batch_size=$batch_size --is_sparse

# GSMN-dense
CUDA_VISIBLE_DEVICES=$gpu python train.py --data_path $data_path --data_name $data_name --vocab_path $vocab_path --logger_name $logger_name --model_name $model_name --bi_gru --max_violation --lambda_softmax=$lambda_softmax --num_epochs=$num_epochs --lr_update=$lr_update --learning_rate=$learning_rate --embed_size=$embed_size --batch_size=$batch_size
