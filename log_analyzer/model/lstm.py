## LSTM LM model
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor    

def initialize_weights(net, initrange = 1.0):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            initrange *= 1.0/np.sqrt(m.weight.data.shape[1])
            m.weight.data = initrange * truncated_normal_(m.weight.data, mean = 0.0, std =1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Embedding):
            truncated_normal_(m.weight.data, mean = 0.0, std =1)


def build_stacked_lstm(layers, embedding_dim, bid):
    lstm_lst = []
    for i, unit_size in enumerate(layers):
        if i == 0:
            lstm_lst.append(('LSTM_fwd'+str(i), nn.LSTM(embedding_dim, unit_size, batch_first = True, bidirectional = bid)))
        else:
            lstm_lst.append(('LSTM_fwd'+str(i), nn.LSTM(layers[i-1], unit_size, batch_first = True, bidirectional = bid)))
    return nn.Sequential(OrderedDict(lstm_lst))      


class Fwd_LSTM(nn.Module):
    def __init__(self, layers, vocab_size, embedding_dim, jagged = False, tiered =False, context_vector_size = 0):
        super().__init__()
        self.layers = layers 
        self.jagged = jagged
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.tiered = tiered
        self.bid = False

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        if self.tiered:
            self.embedding_dim += context_vector_size  

        self.stacked_lstm = build_stacked_lstm(self.layers, self.embedding_dim, self.bid)
        self.hidden2tag = nn.Linear(self.layers[-1], self.vocab_size)
        self.tanh = nn.Tanh()
        initialize_weights(self)

    def forward(self, sequences, lengths = None, context_vectors = None):
        
        x_lookups = self.embeddings(sequences) 
        if self.tiered:
            x_lookups = torch.cat(x_lookups, context_vectors, dim=2)

        lstm_in = x_lookups
        if self.jagged:
            lstm_in = pack_padded_sequence(lstm_in, lengths, batch_first=True)
            
        lstm_out, (hx, cx)  = self.stacked_lstm(x_lookups)
        if self.jagged: 
            lstm_out = pad_packed_sequence(lstm_out, batch_first=True)

        tag_size = self.hidden2tag(lstm_out)
        output = self.tanh(tag_size)


        return output, hx
    
class Bid_LSTM(nn.Module):
    def __init__(self, layers, vocab_size, embedding_dim, jagged = False, tiered =False, context_vector_size = 0):
        super().__init__()
        
        self.layers = layers 
        self.jagged = jagged
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.tiered = tiered
        self.bid = True
        
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        if self.tiered:
            self.embedding_dim += context_vector_size

        self.stacked_bid_lstm = build_stacked_lstm(self.layers, self.embedding_dim, self.bid)
        self.hidden2tag = nn.Linear(self.layers[-1] * 2, self.vocab_size)
        self.tanh = nn.Tanh()

        initialize_weights(self)

    def forward(self, sequences, lengths = None, context_vectors = None):   
        x_lookups = self.embeddings(sequences) #batch size, sequence length, embedded dimension
        if self.tiered:
            x_lookups = torch.cat(x_lookups, context_vectors, dim=2)

        lstm_in = x_lookups
        if self.jagged:
            lstm_in = pack_padded_sequence(lstm_in, lengths, batch_first=True)            
            
        lstm_out, (hx, cx)  = self.stacked_bid_lstm(x_lookups)
        if self.jagged:
            lstm_out = pad_packed_sequence(lstm_out, batch_first=True)
            
        tag_size = self.hidden2tag(lstm_out)
        output = self.tanh(tag_size)
        return tag_size, hx

if __name__ == "__main__":
    # I tried to make this code self-explanatory, but if there is any difficulty to understand ti or possible improvements, please tell me.
    test_layers = [10, 20, 10]
    test_vocab_size = 96
    test_embedding_dim= 30

    fwd_lstm_model = Fwd_LSTM(test_layers, test_vocab_size, test_embedding_dim, jagged = False, tiered =False, context_vector_size = 0)
    print(fwd_lstm_model)

    bid_lstm_model = Bid_LSTM(test_layers, test_vocab_size, test_embedding_dim, jagged = False, tiered =False, context_vector_size = 0)
    print(bid_lstm_model)