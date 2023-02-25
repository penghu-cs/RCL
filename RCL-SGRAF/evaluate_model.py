"""
# Pytorch implementation for AAAI2021 paper from
# https://arxiv.org/pdf/2101.01368.
# "Similarity Reasoning and Filtration for Image-Text Matching"
# Haiwen Diao, Ying Zhang, Lin Ma, Huchuan Lu
#
# Writen by Haiwen Diao, 2020
"""

import os
import time
import shutil

import torch
import numpy

import data
import opts
from vocab import Vocabulary, deserialize_vocab
from model import SGRAF
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data, shard_attn_scores

import logging
import tensorboard_logger as tb_logger
from evaluation import evaluation, evalrank

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"



def main():
    opt = opts.parse_opt()
    if 'coco' in opt.data_name:
        evaluation(opt, split="testall", fold5=True)
        evaluation(opt, split="testall", fold5=False)
    else:
        evaluation(opt, split="test", fold5=False)


if __name__ == '__main__':
    main()
