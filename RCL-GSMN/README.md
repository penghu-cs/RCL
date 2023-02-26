# RCL-GSMN
*PyTorch implementation for IEEE TPAMI 2023 paper of [**“Cross-Modal Retrieval with Partially Mismatched Pairs”**](https://doi.org/10.1109/TPAMI.2023.3247939).* 

*It is built on top of the [GSMN](https://github.com/CrossmodalGroup/GSMN).* 

## Introduction
**The framework of RCL:**
<img src="../fig/RCL.png" width = "100%" height="50%">

## Requirements and Installation
We recommended the following dependencies.

* Python 2.7
* [PyTorch](http://pytorch.org/) 1.1.0
* [NumPy](http://www.numpy.org/) (>1.12.1)
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)

## Pretrained model
If you don't want to train from scratch, you can download the pretrained GSMN model from [here](https://drive.google.com/file/d/1lacmJ-fTtLya71erXOHWE07WEE0WGpKl/view?usp=sharing)(for Flickr30K dense model under 20% noise) and [here](https://drive.google.com/file/d/1YL3ZT3wNuH_fnDccplSI_zFE6548rEJc/view?usp=sharing)(for Flickr30K sparse model under 20% noise). The performance of this pretrained single model is as follows, in which some Recall@1 values are even better than the results produced by our paper:
```bash
GSMN-dense:
rsum: 446.0
Average i2t Recall: 81.7
Image to text: 64.3 87.9 92.8 1.0 4.0
Average t2i Recall: 67.0
Text to image: 47.1 73.0 80.9 2.0 21.6

GSMN-sparse:
rsum: 453.2                                                                                                                                          
Average i2t Recall: 82.8                                                                                                                             
Image to text: 65.8 88.5 94.2 1.0 3.5                                                                                                                
Average t2i Recall: 68.2                                                                                                                             
Text to image: 47.5 74.4 82.8 2.0 13.7  
```


## Download data
Download the dataset files. We use the image feature created by SCAN, downloaded [here](https://github.com/kuanghuei/SCAN). The text feature, image bounding box and semantic dependency are precomputed, and can be downloaded from [here](https://drive.google.com/file/d/1ZVLIN7uSh3dqYAEldelyYF2ei9vicJvZ/view?usp=sharing) (for Flickr30K and MSCOCO) 

## Training

```bash
python train.py --data_path "$DATA_PATH" --data_name f30k_precomp --vocab_path "$VOCAB_PATH" --logger_name runs/log --model_name "$MODEL_PATH" --bi_gru
```

Arguments used to train Flickr30K models and MSCOCO models are similar with those of SCAN:

For Flickr30K:

| Method      | Arguments |
| :---------: | :-------: |
|  GSMN-dense   | `--max_violation --lambda_softmax=20 --num_epochs=30 --lr_update=15 --learning_rate=.0002 --embed_size=1024 --batch_size=64 `|
|  GSMN-sparse    | `--max_violation --lambda_softmax=20 --num_epochs=30 --lr_update=15 --learning_rate=.0002 --embed_size=1024 --batch_size=64 --is_sparse `| 

For MSCOCO:

| Method      | Arguments |
| :---------: | :-------: |
|  GSMN-dense   | `--max_violation --lambda_softmax=10 --num_epochs=20 --lr_update=5 --learning_rate=.0005 --embed_size=1024 --batch_size=32 `|
|  GSMN-sparse    | `--max_violation --lambda_softmax=10 --num_epochs=20 --lr_update=5 --learning_rate=.0005 --embed_size=1024 --batch_size=32 --is_sparse `|

## Evaluation

Test on Flickr30K
```bash
python test.py
```

To do cross-validation on MSCOCO, pass `fold5=True` with a model trained using 
`--data_name coco_precomp`.

```bash
python testall.py
```

To ensemble sparse model and dense model, specify the model_path in test_stack.py, and run
```bash
python test_stack.py
```

<!-- ## Reference

If you found this code useful, please cite the following paper:
```
@inproceedings{liu2020graph,
  title={Graph Structured Network for Image-Text Matching},
  author={Liu, Chunxiao and Mao, Zhendong and Zhang, Tianzhu and Xie, Hongtao and Wang, Bin and Zhang, Yongdong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10921--10930},
  year={2020}
}
``` -->

