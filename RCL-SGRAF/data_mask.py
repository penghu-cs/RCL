"""Data provider"""

import torch
import torch.utils.data as data

import os
import nltk
import numpy as np
import scipy.io as sio
import h5py
import json
import csv
import random

class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split, vocab, opt=None):
        self.vocab = vocab
        loc = data_path + '/'
        self.train = data_split == 'train'

        # load the raw captions
        self.captions = []
        # with open(loc+'%s_caps.txt' % data_split, 'rb') as f:
        #     for line in f.readlines():
        #         self.captions.append(line.strip())

        if 'cc152k_precomp' in data_path:
            with open(os.path.join(data_path, '%s_caps.tsv' % data_split), encoding='utf-8') as f:
                tsvreader = csv.reader(f, delimiter='\t')
                for line in tsvreader:
                    self.captions.append(line[1].strip())
        else:
            with open(loc + '%s_caps.txt' % data_split, encoding='utf-8') as f:
                for line in f.readlines():
                    self.captions.append(line.strip())
            # with open(loc + '%s_caps.txt' % data_split, 'rb') as f:
            #     for line in f:
            #         self.captions.append(line.strip())

        # load the image features
        self.images = np.load(loc+'%s_ims.npy' % data_split)
        # self.images = h5py.File(loc+'%s_ims.h5py' % data_split, 'r', libver='latest', swmr=True)[data_split]
        # self.data_split = data_split
        # with h5py.File(self.img_file, 'r', libver='latest', swmr=True) as h:
        #     images = h[data_split]

        self.img_len = self.images.shape[0]
        self.noisy_inx = np.arange(self.img_len)
        if data_split == 'train' and opt.noise_rate > 0:
            noise_file = os.path.join(loc, 'noise_inx_%g.npy' % opt.noise_rate)
            # noise_file = os.path.join('../SCAN/data/noise_index_hzy/', '%s_%g.npy' % (opt.data_name, opt.noise_rate))
            if os.path.exists(noise_file):
                self.noisy_inx = np.load(noise_file)
                print('Loaded noisy indices from %s' % noise_file)
                print('Noisy rate: %g' % opt.noise_rate)
            else:
                noise_rate = opt.noise_rate
                inx = np.arange(self.img_len)
                np.random.shuffle(inx)
                noisy_inx = inx[0: int(noise_rate * self.img_len)]
                shuffle_noisy_inx = np.array(noisy_inx)
                np.random.shuffle(shuffle_noisy_inx)
                # self.images[noisy_inx] = self.images[shuffle_noisy_inx]
                self.noisy_inx[noisy_inx] = shuffle_noisy_inx

                np.save(noise_file, self.noisy_inx)
                print('Noisy rate: %g' % noise_rate)
            # exit(0)
        self.length = len(self.captions)
        # import pdb
        # pdb.set_trace()

        # rkiros data has redundancy in images, we divide by 5
        if self.img_len != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev' and self.length >= 5000:
            self.length = 5000

    def __getitem__(self, index):
        # handle the image redundancy
        if self.noisy_inx.shape[0] == self.img_len:
            img_id = self.noisy_inx[int(index/self.im_div)]
        else:
            img_id = self.noisy_inx[index]
        # with h5py.File(self.img_file, 'r', libver='latest', swmr=True) as h:
        # images = h[self.data_split]
        # image = torch.Tensor(self.images[img_id])

        caption = self.captions[index]

        # vocab = self.vocab
        # convert caption (string) to word ids.
        # tokens = nltk.tokenize.word_tokenize(
        #     # str(caption).lower().decode('utf-8'))
        #     # str(caption, 'utf-8').lower())
        #     str(caption).lower())
        # caption = []
        # caption.append(vocab('<start>'))
        # caption.extend([vocab(token) for token in tokens])
        # caption.append(vocab('<end>'))
        # target = torch.Tensor(caption)


        target = process_caption(self.vocab, caption, self.train)
        image = self.images[img_id]
        if self.train:  # Size augmentation on region features.
            num_features = image.shape[0]
            rand_list = np.random.rand(num_features)
            image[np.where(rand_list <= 0.20)] = 1e-10
            # image = image[np.where(rand_list > 0.20)]
        image = torch.Tensor(image)
        return image, target, index, img_id
    def __len__(self):
        return self.length

def process_caption(vocab, caption, drop=False):
    if not drop:
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        caption = list()
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return target
    else:
        # Convert caption (string) to word ids.
        tokens = ['<start>', ]
        tokens.extend(nltk.tokenize.word_tokenize(caption.lower()))
        tokens.append('<end>')
        deleted_idx = []
        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.20:
                prob /= 0.20
                # 50% randomly change token to mask token
                if prob < 0.5:
                    tokens[i] = vocab.word2idx['<mask>']
                # 10% randomly change token to random token
                elif prob < 0.6:
                    tokens[i] = random.randrange(len(vocab))
                # 40% randomly remove the token
                else:
                    tokens[i] = vocab(token)
                    deleted_idx.append(i)
            else:
                tokens[i] = vocab(token)
        if len(deleted_idx) != 0:
            tokens = [tokens[i] for i in range(len(tokens)) if i not in deleted_idx]
        target = torch.Tensor(tokens)
        return target

def collate_fn(data):
    """
    Build mini-batch tensors from a list of (image, caption, index, img_id) tuples.
    Args:
        data: list of (image, target, index, img_id) tuple.
            - image: torch tensor of shape (36, 2048).
            - target: torch tensor of shape (?) variable length.
    Returns:
        - images: torch tensor of shape (batch_size, 36, 2048).
        - targets: torch tensor of shape (batch_size, padded_length).
        - lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 2D tensor to 3D tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, ids


def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    dset = PrecompDataset(data_path, data_split, vocab, opt)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn,
                                              num_workers=num_workers)
    return data_loader


def get_loaders(data_name, vocab, batch_size, workers, opt):
    # get the data path
    dpath = os.path.join(opt.data_path, data_name)

    # get the train_loader
    train_loader = get_precomp_loader(dpath, 'train', vocab, opt,
                                      batch_size, True, workers)
    # get the val_loader
    val_loader = get_precomp_loader(dpath, 'dev', vocab, opt,
                                    100, False, workers)
    return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, batch_size, workers, opt):
    # get the data path
    dpath = os.path.join(opt.data_path, data_name)

    # get the test_loader
    test_loader = get_precomp_loader(dpath, split_name, vocab, opt,
                                     100, False, workers)
    return test_loader
