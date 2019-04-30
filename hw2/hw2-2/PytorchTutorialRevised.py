# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 17:12:08 2019

@author: u8815
"""
#from chatbot tutorial on pytorch:https://pytorch.org/tutorials/beginner/chatbot_tutorial.html

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
import os
import numpy as np
MAX_LENGTH = 20
USE_CUDA = torch.cuda.is_available()
device = torch.device( "cuda")
word2idx=np.load("word2idx.npy")
idx2word=np.load("idx2word.npy")
train_x=np.load("train_x.npy").astype(np.int)
train_y=np.load("train_y.npy").astype(np.int)
class Voc:
    def __init__(self,name,word2index,index2word):
        self.name = name
        self.word2index = word2index
        self.index2word  = index2word
        self.num_words=71475
voc=Voc("chatbot",word2idx,idx2word)
def binaryMatrix(l, value=0):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == 0:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

def inputVar(l, voc):
    indexes_batch = [x for x in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    #padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(indexes_batch)
    return torch.transpose(padVar,0,1), lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [s for s in l]
    indexes_batch2=np.array(indexes_batch).T
    max_target_len=0
    for i in range(0,len(indexes_batch2)):
        if np.sum(indexes_batch2[i]!=0):
            max_target_len+=1
        else:
            break
    #padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(indexes_batch)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(indexes_batch)
    return torch.transpose(padVar,0,1), torch.transpose(mask,0,1), max_target_len
def batch2TrainData(voc, train_x,train_y):
    #pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for i in range(len(train_x)):
        input_batch.append(train_x[i])
        output_batch.append(train_y[i])
    inp, lengths = inputVar(input_batch, voc)
    output,mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len


# Example for validation
small_batch_size = 5
selected=[]
for k in range(small_batch_size):    
    selected.append(random.randint(0,len(train_x)-1))
batches = batch2TrainData(voc, [train_x[c] for c in selected],[train_y[c] for c in selected])
input_variable, lengths, target_variable, mask, max_target_len = batches

print("input_variable:", input_variable)
print("lengths:", lengths)
print("target_variable:", target_variable)
print("mask:", mask)
print("max_target_len:", max_target_len)

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        embedded=embedded.type(torch.cuda.FloatTensor)
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden
# Luong attention layer
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)
class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        #print(last_hidden.shape)
        
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        embedded=embedded.type(torch.cuda.FloatTensor)
        #print (embedded.shape)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        #print (output.shape)
        # Return output and final hidden state
        return output, hidden
def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()
def inverse_sigmoid(x,k=1):
    return 1-1/(1+np.exp(x/k))
def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip,epoch,check,BOS_length=0,max_length=MAX_LENGTH):

    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    if not check:
        decoder_input = torch.LongTensor([[1 for _ in range(batch_size)]])
    else:
        decoder_input = torch.LongTensor([[1 for _ in range(BOS_length)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]
    #print(encoder_hidden.shape)

    # Determine if we are using teacher forcing this iteration
    #Schedule Sampling with inverse sigmoid
    teacher_forcing_ratio = inverse_sigmoid((epoch-75)/25)
    if epoch<=50:
        use_teacher_forcing = True
    else:
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            #print (mask[t].shape)
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return sum(print_losses) / n_totals
def trainIters(model_name, voc, train_x, train_y, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, save_dir, epochs, batch_size, print_every, save_every, clip, corpus_name, loadFilename):

    # Load batches for each iteration
        
    
        # Initializations
        print('Initializing...')
        training_batch=[]
        for k in range(0,train_x.shape[0]//batch_size+1):
            a=batch_size*k
            b=a+batch_size
            if b>train_x.shape[0]:
                b=train_x.shape[0]
            training_batch.append(batch2TrainData(voc, [train_x[i] for i in range(a,b)],[train_y[i] for i in range(a,b)]))
                #print (training_batch[0].shape)
        print('Training ...')
        for epoch in range(epochs):
            print_loss = 0
            for l in range(0,train_x.shape[0]//batch_size+1):
                check=0
                if loadFilename:
                    epoch = checkpoint['epoch'] + 1
            
                # Training loop

                # Extract fields from batch
                input_variable, lengths, target_variable, mask, max_target_len = training_batch[l]
                BOS_length=batch_size
                if (l==train_x.shape[0]//batch_size):
                    check=1
                    BOS_length= input_variable.shape[1]
                # Run a training iteration with batch
                loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                             decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip,epoch,check,BOS_length)
                print_loss += loss
        
                # Print progress
                if (l+1) % print_every == 0:
                    print_loss_avg = print_loss / print_every
                    print("Epoch: {};Batch: {}  Average loss: {:.4f}".format(epoch+1,l+1, print_loss_avg))
                    print_loss = 0
        
                # Save checkpoint
            if ((epoch+1) % save_every == 0 ):
                directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
                if not os.path.exists(directory):
                    os.makedirs(directory)
                torch.save({
                    'epoch': epoch,
                    'en': encoder.state_dict(),
                    'de': decoder.state_dict(),
                    'en_opt': encoder_optimizer.state_dict(),
                    'de_opt': decoder_optimizer.state_dict(),
                    'loss': loss,
                    'voc_dict': voc.__dict__,
                    'embedding': embedding.state_dict()
                }, os.path.join(directory, '{}_{}.tar'.format(epoch, 'checkpoint')))
model_name = 'cb_model'
#attn_model = 'dot'
#attn_model = 'general'
attn_model = 'concat'
hidden_size = 250
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0
batch_size = 512

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = None
#checkpoint_iter = 
#loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                            '{}_checkpoint.tar'.format(checkpoint_iter))


# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']


print('Building encoder and decoder ...')
# Initialize word embeddings
temp=np.load("embedding_matrix_normalized.npy")
embedding = nn.Embedding.from_pretrained(torch.from_numpy(temp))
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')
clip = 50.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
epoch=100
print_every = 100
save_every = 1

save_dir="./model"
corpus_name="chat_bot"
# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# Run training iterations
print("Starting Training!")
trainIters(model_name, voc, train_x,train_y, encoder, decoder, encoder_optimizer, decoder_optimizer,
           embedding, encoder_n_layers, decoder_n_layers, save_dir, epoch, batch_size,
           print_every, save_every, clip, corpus_name, loadFilename)
