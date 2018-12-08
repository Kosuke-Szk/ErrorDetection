import torch
import torch.nn as nn
from torch.autograd import Variable
import torchwordemb
import warnings

class BiLSTM(nn.Module):

    def __init__(self, _vocab_size, embed_size, hidden_size, output_size, extra_hidden_size):
        super(BiLSTM, self).__init__()
        self.vocab_size = _vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(_vocab_size, embed_size)
        self.linear1 = nn.Linear(hidden_size*2, extra_hidden_size)
        self.linear2 = nn.Linear(extra_hidden_size, output_size)

        self.blstm = nn.LSTM(embed_size, hidden_size, num_layers=1, bidirectional=True)
        self.relu = nn.ReLU()

        self.blstm_hidden = self._init_hidden(2)

    def _init_hidden(self, layers=1):
        h = Variable(torch.zeros(layers, 1, self.hidden_size))
        cell = Variable(torch.zeros(layers, 1, self.hidden_size))
        if torch.cuda.is_available():
            h = h.cuda()
            cell = cell.cuda()
        return (h, cell)

    def _reset_state(self):
        self.blstm_hidden = self._init_hidden(2)
        self.zero_grad()

    def initialize_embed(self, word2vec_model, word2id):
        w2v_vocab, w2v_vectors = torchwordemb.load_word2vec_text(word2vec_model)
        for word, i in word2id.items():
            # ignore the unicode conversion/comparison warning
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                if word in w2v_vocab.keys():
                    self.embedding.weight.data[i].copy_(w2v_vectors[w2v_vocab[word]])


    def forward(self, x):
        self._reset_state()
        h_states = []
        o_states = []

        # list of tensors of size batch_size x embed_size
        e_states = [self.relu(self.embedding(w)) for w in x]
        for e in e_states:
            inp = e.view(len(e), 1, -1)
            out, self.blstm_hidden = self.blstm(inp, self.blstm_hidden)
            out = out.view(len(e), -1)
            h_states.append(out)

        for h in h_states:
            o_states.append(self.linear2(self.relu(self.linear1(h))).view(-1, self.output_size))

        return o_states