## LSTM LM model
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import log_analyzer.model.auxiliary as auxiliary

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
            
class Fwd_LSTM(nn.Module):
    def __init__(self, layers, vocab_size, embedding_dim, jagged = False, tiered =False, context_vector_size = 0):
        super().__init__()
        # Parameter setting
        self.layers = layers 
        self.jagged = jagged
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.tiered = tiered
        self.bid = False

        # Layers
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        if self.tiered:
            self.embedding_dim += context_vector_size  
        self.stacked_lstm = nn.LSTM(self.embedding_dim, self.layers[0], len(self.layers), batch_first = True, bidirectional = self.bid)
        self.tanh = nn.Tanh()
        self.hidden2tag = nn.Linear(self.layers[-1], self.vocab_size)

        # Weight initialization
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

        output = self.tanh(lstm_out)
        tag_size = self.hidden2tag(output)

        return tag_size, lstm_out, hx
    
class Bid_LSTM(nn.Module):
    def __init__(self, layers, vocab_size, embedding_dim, jagged = False, tiered =False, context_vector_size = 0):
        super().__init__()
        # Parameter setting
        self.layers = layers 
        self.jagged = jagged
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.tiered = tiered
        self.bid = True
        
        # Layers
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        if self.tiered:
            self.embedding_dim += context_vector_size
        self.stacked_bid_lstm = nn.LSTM(self.embedding_dim, self.layers[0], len(self.layers), batch_first = True, bidirectional = self.bid)
        self.tanh = nn.Tanh()
        self.hidden2tag = nn.Linear(self.layers[-1] * 2, self.vocab_size)

        # Weight initialization
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
        
        output = self.tanh(lstm_out)
        tag_size = self.hidden2tag(output)
           
        return tag_size, lstm_out, hx

class Context_LSTM(nn.Module):
    def __init__(self, ctxt_lv_layers, input_dim):
        super().__init__()
        # Parameter setting
        self.ctxt_lv_layers = ctxt_lv_layers 
        self.input_dim = input_dim

        # Layers
        self.context_lstm_layers = nn.LSTM(input_dim, self.ctxt_lv_layers[0], len(ctxt_lv_layers), batch_first = True, bidirectional = bid)

        # Weight initialization
        initialize_weights(self)

    def forward(self, lower_lv_outputs, final_hidden, context_h, context_c, seq_len = None):   

        if seq_len is not None:
            mean_hidden = torch.sum(lower_lv_outputs, dim = 1) / seq_len
        else:
            mean_hidden = torch.mean(lower_lv_outputs, dim = 1)
        synthetic_input = torch.cat(mean_hidden, final_hidden, dim=1)
        output, (context_hx, context_cx) = self.context_lstm_layers(synthetic_input, (context_h, context_c))
        
        return output, context_hx, context_cx 
    
class Tiered_LSTM(nn.Module):
    def __init__(self, low_lv_layers, ctxt_lv_layers, vocab_size, embedding_dim, context_vector_size, 
                 jagged = False, bid = False):
        
        super().__init__()
        # Parameter setting
        self.bid = bid
        if self.bid:
            self.model = Bid_LSTM
        else:
            self.model = Fwd_LSTM
        self.low_lv_layers = low_lv_layers 
        self.ctxt_lv_layers = ctxt_lv_layers
        self.jagged = jagged
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Layers        
        self.low_lv_lstm = self.model(self.low_lv_layers, self.vocab_size, self.embedding_dim, 
                                      jagged = self.jagged, tiered = True, context_vector_size = self.ctxt_lv_layers[-1])
        self.ctxt_lv_lstm = Context_LSTM(self.ctxt_lv_layers, low_lv_layers[-1] * 2)

        # Weight initialization
        initialize_weights(self)

    def forward(self, user_sequences, context_vectors, context_h, context_c, lengths = None):
        self.ctxt_vector = context_vectors
        self.ctxt_h = context_h
        self.ctxt_c = context_c
        self.tag_output = []
        for sequences in range(user_sequences): #number of steps (e.g., 3), number of users (e.g., 64), lengths of sequences (e.g., 10)
            tag_size, low_lv_lstm_outputs, final_hidden = self.low_lv_lstm(sequences, lengths = lengths, context_vectors = self.ctxt_vector)
            self.ctxt_vector, (self.ctxt_h, self.ctxt_c) = self.ctxt_lv_lstm(low_lv_lstm_outputs, final_hidden, self.ctxt_h, self.ctxt_c, seq_len = lengths)
            self.tag_output.append(tag_size)
        return self.tag_output, self.ctxt_vector, self.ctxt_h, self.ctxt_c

if __name__ == "__main__":
    # I tried to make this code self-explanatory, but if there is any difficulty to understand ti or possible improvements, please tell me.
    test_layers = [10, 10] #each layer has to have the same hidden units.
    test_vocab_size = 96
    test_embedding_dim= 30

    fwd_lstm_model = Fwd_LSTM(test_layers, test_vocab_size, test_embedding_dim, jagged = False, tiered =False, context_vector_size = 0)
    print(fwd_lstm_model)

    bid_lstm_model = Bid_LSTM(test_layers, test_vocab_size, test_embedding_dim, jagged = False, tiered =False, context_vector_size = 0)
    print(bid_lstm_model)