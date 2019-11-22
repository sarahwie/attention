import torch
import torch.nn as nn

from attention.model.modelUtils import isTrue
from allennlp.common import Registrable
from allennlp.nn.activations import Activation

class Encoder(nn.Module, Registrable) :
    def forward(self, **kwargs) :
        raise NotImplementedError("Implement forward Model")

@Encoder.register('rnn')
class EncoderRNN(Encoder) :
    def __init__(self, vocab_size, embed_size, hidden_size, pre_embed=None) :
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        if pre_embed is not None :
            print("Setting Embedding")
            weight = torch.Tensor(pre_embed)
            weight[0, :].zero_()

            self.embedding = nn.Embedding(vocab_size, embed_size, _weight=weight, padding_idx=0)
        else :
            self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)

        self.output_size = self.hidden_size * 2

    def forward(self, data) :
        seq = data.seq
        lengths = data.lengths
        embedding = self.embedding(seq) #(B, L, E)
        packseq = nn.utils.rnn.pack_padded_sequence(embedding, lengths, batch_first=True, enforce_sorted=False)
        output, (h, c) = self.rnn(packseq)
        output, lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, padding_value=0)

        data.hidden = output
        data.last_hidden = torch.cat([h[0], h[1]], dim=-1)

        if isTrue(data, 'keep_grads') :
            data.embedding = embedding
            data.embedding.retain_grad()
            data.hidden.retain_grad()
            

@Encoder.register("average")
class EncoderAverage(Encoder) :
    def __init__(self,  vocab_size, embed_size, projection, hidden_size=None, activation:Activation=Activation.by_name('linear'), pre_embed=None) :
        super(EncoderAverage, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        if pre_embed is not None :
            print("Setting Embedding")
            weight = torch.Tensor(pre_embed)
            weight[0, :].zero_()

            self.embedding = nn.Embedding(vocab_size, embed_size, _weight=weight, padding_idx=0)
        else :
            self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        if projection :
            self.projection = nn.Linear(embed_size, hidden_size)
            self.output_size = hidden_size
        else :
            self.projection = lambda s : s
            self.output_size = embed_size

        self.activation = activation

    def forward(self, data) :
        seq = data.seq
        lengths = data.lengths
        embedding = self.embedding(seq) #(B, L, E)

        output = self.activation(self.projection(embedding))
        h = output.mean(1)

        data.hidden = output
        data.last_hidden = h

        if isTrue(data, 'keep_grads') :
            data.embedding = embedding
            data.embedding.retain_grad()
            data.hidden.retain_grad()
