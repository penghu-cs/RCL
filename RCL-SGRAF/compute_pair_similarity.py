import os
import sys
import time

import torch
import numpy as np

from data import get_test_loader, collate_fn
from vocab import Vocabulary, deserialize_vocab
from model import SGRAF
import opts
import scipy.io as sio
# from evaluation as encode_data

def encode_data_sim(model, data_loader, max_len=-1):
    max_n_word = 0
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        max_n_word = max(max_n_word, max(lengths))
    img_embs = None
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        # compute the embeddings
        images, captions = images.cuda(), captions.cuda()
        with torch.no_grad():
            img_emb, cap_emb, cap_len = model.forward_emb(images, captions, lengths)

        if img_embs is None:
            # img_embs = np.zeros((len(data_loader.dataset.images), img_emb.size(1), img_emb.size(2)), dtype=np.single)
            max_len = len(data_loader.dataset.images) if max_len <= 0 else max_len
            img_embs = np.zeros((max_len, img_emb.size(1), img_emb.size(2)), dtype=np.single)
            cap_embs = np.zeros((len(data_loader.dataset.captions), max_n_word, cap_emb.size(2)), dtype=np.single)
            cap_lens = [0] * len(data_loader.dataset)
        # cache embeddings
        img_inx = (np.array(ids) // 5).astype('int64')
        bool_inx = img_inx < max_len
        img_inx = img_inx[bool_inx]
        img_embs[img_inx] = img_emb.data.cpu().numpy().copy()[bool_inx]
        cap_embs[ids, :max(lengths), :] = cap_emb.data.cpu().numpy().copy()

        for j, nid in enumerate(ids):
            cap_lens[nid] = cap_len[j]

        del images, captions

    shard_size = 100
    n_im_shard = (len(img_embs) - 1) // shard_size + 1
    n_cap_shard = (len(cap_embs) - 1) // shard_size + 1
    sims = np.zeros((len(img_embs), len(cap_embs)), dtype=np.single)
    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(img_embs))
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_attn_scores batch (%d,%d)' % (i, j))
            ca_start, ca_end = shard_size * j, min(shard_size * (j + 1), len(cap_embs))

            with torch.no_grad():
                im = torch.from_numpy(img_embs[im_start:im_end]).float().cuda()
                ca = torch.from_numpy(cap_embs[ca_start:ca_end]).float().cuda()
                l = cap_lens[ca_start:ca_end]
                sim = model.forward_sim(im, ca, l)

            sims[im_start:im_end, ca_start:ca_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    del img_embs, cap_embs
    return sims


def encode_data_sim_neg(model, data_loader):
    """Encode all images and captions loadable by `data_loader`
    """
    # switch to evaluate mode
    model.val_start()

    # np array to keep all the embeddings
    # max_n_word = 0
    # for i, (images, captions, lengths, ids) in enumerate(data_loader):
    #     max_n_word = max(max_n_word, max(lengths))

    sims = np.zeros(len(data_loader.dataset.captions))
    negs = np.zeros(len(data_loader.dataset.captions))
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        # compute the embeddings
        with torch.no_grad():
            img_emb, cap_emb, cap_len = model.forward_emb(images, captions, lengths)
            sim = model.forward_sim(img_emb, cap_emb, cap_len)
        # sims.append(sim.diag().data.cpu().numpy())
        sim = sim.data.cpu().numpy()
        sims[ids] = sim.diagonal()
        # tmp = np.tril(sim, k=-5) + np.triu(sim, k=5)
        row = np.array(ids) // 5
        neg = sim[row.reshape([-1, 1]) != row.reshape([1, -1])].reshape([-1])
        negs[ids] = np.random.choice(neg, sim.shape[0])
        # raw_inx = (np.array(ids) // 5).astype('int64')
        # col_inx = ids
        # sims[raw_inx, col_inx] = sim
    return sims, negs

def encode_data_sim_part(model, data_loader):
    """Encode all images and captions loadable by `data_loader`
    """
    # switch to evaluate mode
    model.val_start()

    # np array to keep all the embeddings
    # max_n_word = 0
    # for i, (images, captions, lengths, ids) in enumerate(data_loader):
    #     max_n_word = max(max_n_word, max(lengths))


    img_batch, img_ids = [], []
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        with torch.no_grad():
            img, _, _ = model.forward_emb(images, captions, lengths)
        if i >= 10:
            break
        img_batch.append(torch.stack([img[i] for i in range(0, img.shape[0], 5)]))
        img_ids += [ids[i] // 5 for i in range(0, len(ids), 5)]
    img_batch = torch.cat(img_batch)
    img_ids = np.array(img_ids)
    img_batch[img_ids] = img_batch
    sims = np.zeros((img_batch.shape[0], len(data_loader.dataset.captions)))
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        # compute the embeddings
        with torch.no_grad():
            img_emb, cap_emb, cap_len = model.forward_emb(images, captions, lengths)
            sim = model.forward_sim(img_batch, cap_emb, cap_len)
        # sims.append(sim.diag().data.cpu().numpy())
        sim = sim.data.cpu().numpy()
        sims[:, np.array(ids)] = sim
    return sims

def evaluation(config, model_path=None, data_path=None, split='test', fold5=False):
    if model_path is None:
        # if config.module_name == 'SGRAF':
        #     model_path = [config.model_name + '/' + ('%s_%s_model_best_%g.pth.tar' % (config.data_name, module_name, config.noise_rate)) for module_name in ['SAF', 'SGR']]
        #     module_names = ['SAF', 'SGR', 'SGRAF']
        # else:
        #     model_path = [config.model_name + '/' + ('%s_%s_model_best_%g.pth.tar' % (config.data_name, config.module_name, config.noise_rate))]
        #     module_names = [config.module_name]
        model_path = config.model_path
        module_names = config.module_names
    else:
        model_path = [model_path]
        module_names = []

    # load model and options
    path = model_path[0]
    checkpoint = torch.load(path)
    opt = checkpoint['opt']
    if len(module_names) == 0:
        module_names.append(opt.module_name)
    save_epoch = checkpoint['epoch']
    print(opt)
    # if data_path is not None:
    #     opt.data_path = data_path
    # else:
    module = opt.module_name
    opt = config
    opt.module_name = module
    # opt.data_path = config.data_path

    # load vocabulary used by the model
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
    opt.vocab_size = len(vocab)

    # construct model
    model = SGRAF(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])
    model.val_start()

    print('Loading dataset')
    noise_rate = opt.noise_rate
    opt.noise_rate = 0
    data_loader = get_test_loader(split, opt.data_name, vocab,
                                    opt.batch_size, opt.workers, opt)
    print("=> loaded checkpoint_epoch {}".format(save_epoch))
    opt.noise_rate = noise_rate

    print('Computing results...')
    start = time.time()
    # sims, negs = encode_data_sim(model, data_loader)
    sims = encode_data_sim(model, data_loader, max_len=100)
    # sims = encode_data_sim_part(model, data_loader)
    end = time.time()
    print("calculate similarity time:", end-start)
    # sio.savemat('%s_SGR_all_sims.mat' % opt.func, {'sims': np.array(sims), 'negs': np.array(negs)})
    sio.savemat('%s_SGR_train_part_sims.mat' % opt.func, {'sims': np.array(sims)})
    print("calculate similarity time:", end - start)

    # noisy_inx = np.load('../SCAN/data/data/coco_precomp/noise_inx_0.6.npy')
    # clean_index = noisy_inx[noisy_inx == np.arange(noisy_inx.shape[0])]
    # clean_index = np.stack([clean_index * 5 + i for i in range(5)], axis=1).reshape([-1])
    # noisy_index = noisy_inx[noisy_inx != np.arange(noisy_inx.shape[0])]
    # noisy_index = np.stack([noisy_index * 5 + i for i in range(5)], axis=1).reshape([-1])

def main():
    opt = opts.parse_opt()
    evaluation(opt, split="train", fold5=False)

if __name__ == '__main__':
    main()