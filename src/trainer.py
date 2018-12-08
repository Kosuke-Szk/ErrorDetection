from __future__ import division

from functions import fill_batch, make_dict, take_len
from BiLSTM import BiLSTM
import numpy as np
import collections

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim

import pickle
import generators as gens
import random

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

train_txt = "../data/train.kaneko.txt"
dev_txt = "../data/dev.kaneko.txt"
test_txt = "../data/test.kaneko.txt"
load_model  = "../ptmodel/ptBLSTM.model0"
vocab_dict = "../ptmodel/BLSTMVocab.pkl"

vocab_size = take_len(train_txt)
batch_size = 256
embed_size = 300
output_size = 2
hidden_size = 200
extra_hidden_size = 50
epoch = 30

def forward(batchs, tags, model, word2id, mode):
    softmax = nn.Softmax()
    cross_entropy_loss = nn.CrossEntropyLoss()

    x = Variable(torch.LongTensor(
        [[word2id[word] if word in word2id else word2id['<unk>'] for word in sen] for sen in batchs])).t()

    if torch.cuda.is_available():
        x = x.cuda()

    pres = model(x)
    sortmax_pres = [softmax(pre) for pre in pres]
    condition = x.data != -1

    if mode:  # are we calculating the loss??
        accum_loss = Variable(torch.zeros(1))  # initialize the loss count to zero
        _tags = np.array(tags, dtype=np.int64)
        tags = Variable(torch.from_numpy(_tags)).t()

        if torch.cuda.is_available():
            accum_loss = accum_loss.cuda()
            tags = tags.cuda()

        for tag, pre in zip(tags, pres):
            accum_loss += cross_entropy_loss(pre, tag)

        return accum_loss, sortmax_pres, condition

    return sortmax_pres, condition


def evaluate(model, word2id):
    c_p = 0
    correct_p = 0
    c_r = 0
    correct_r = 0

    gen1 = gens.word_list(dev_txt)
    gen2 = gens.batch(gens.sorted_parallel(gen1, embed_size * batch_size), batch_size)
    batchs = [b for b in gen2]
    for batch in batchs:
        tag0 = batch[:]
        tags = [a[:-1] for a in tag0]
        batch = [b[1:] for b in batch]
        batch = fill_batch([b[-1].split() for b in batch])
        tags = fill_batch(tags, token=-1)
        pres, cons = forward(batch, tags, model, word2id, mode=False)
        a, b, c, d = precision_recall_f(pres, tags, cons)
        c_p += a
        correct_p += b
        c_r += c
        correct_r += d
    try:
        precision = correct_p / c_p
        recall = correct_r / c_r
        f_measure = (1 + 0.5 ** 2) * precision * recall / (0.5 ** 2 * precision + recall)
        print('Precision:\t{}'.format(precision))
        print('Recall:\t{}'.format(recall))
        print('F-value\t{}'.format(f_measure))
    except ZeroDivisionError:
        precision = 0
        recall = 0
        f_measure = 0
        print('Precision:\tnothing')
        print('Recall:\tnothing')
        print('F-value\tnothing')
    return precision, recall, f_measure


def train():
    id2word = {}
    word2id = {}
    word_freq = collections.defaultdict(lambda: 0)
    id2word[0] = "<unk>"
    word2id["<unk>"] = 0
    id2word[1] = "<s>"
    word2id["<s>"] = 1
    id2word[2] = "</s>"
    word2id["</s>"] = 2

    word2id, id2word, word_list, word_freq = make_dict(train_txt, word2id, id2word, word_freq)
    model = BiLSTM(vocab_size, embed_size, hidden_size, output_size, extra_hidden_size)
    model.initialize_embed('../data/embedding.txt', word2id)
    if torch.cuda.is_available():
        model.cuda()
    opt = optim.Adam(model.parameters(), lr=0.001)

    for i in range(1, epoch + 1):
        print("\nepoch {}".format(i))
        total_loss = 0
        gen1 = gens.word_list(train_txt)
        gen2 = gens.batch(gens.sorted_parallel(gen1, embed_size * batch_size), batch_size)
        batchs = [b for b in gen2]
        bl = list(range(len(batchs)))
        random.shuffle(bl)
        for n, j in enumerate(bl):
            tag0 = batchs[j][:]
            tags = [[int(c) for c in a[:-1]] for a in tag0]
            batch = fill_batch([b[-1].split() for b in batchs[j]])
            tags = fill_batch(tags, token=0)
            accum_loss, pres, cons = forward(batch, tags, model, word2id, mode=True)
            accum_loss.backward()
            opt.step()
            total_loss += accum_loss.data[0]
        print("total_loss {}".format(total_loss))
        torch.save(model.state_dict(), "{}{}".format(load_model, i))

    torch.save(model.state_dict(), load_model)
    with open(vocab_dict, mode='wb') as f:
        pickle.dump(word2id, f)

def test():
    word2id = pickle.load(open(vocab_dict, 'rb'))
    model = BiLSTM(vocab_size, embed_size, hidden_size, output_size, extra_hidden_size)
    model.load_state_dict(torch.load(load_model))
    if torch.cuda.is_available():
        model = model.cuda()

    for i in range(1, epoch + 1):
        print("\nepoch {}".format(i))
        total_loss = 0
        gen1 = gens.word_list(test_txt)
        gen2 = gens.batch(gens.sorted_parallel(gen1, embed_size*batch_size), batch_size)
        batchs = [b for b in gen2]
        bl = list(range(len(batchs)))
        random.shuffle(bl)
        for n, j in enumerate(bl):
            tag0 = batchs[j][:]
            tags = [[int(c) for c in a[:-1]] for a in tag0]
            batch = fill_batch([b[-1].split() for b in batchs[j]])
            tags = fill_batch(tags, token=0)
            accum_loss, pres, cons = forward(batch, tags, model, word2id, mode=True)
            print(pres)
            total_loss += accum_loss.data[0]
        print("total_loss {}".format(total_loss))

if __name__ == '__main__':
    import sys

    mode = sys.argv[1] # train / test
    if mode == 'train':
        train()
    else:
        test()