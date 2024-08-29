import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HangmanTransformerWithSinCos(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model, nhead, num_layers, dim_feedforward):
        super(HangmanTransformerWithSinCos, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self.create_sin_cos_positional_encoding(max_seq_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def create_sin_cos_positional_encoding(self, max_seq_len, d_model):
        # Initialize the positional encoding matrix with zeros
        positional_encoding = torch.zeros(max_seq_len, d_model)
        
        # Calculate the positional encoding using the sine and cosine functions
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        
        # Add an extra dimension to positional_encoding for batch processing
        positional_encoding = positional_encoding.unsqueeze(0)
        
        return nn.Parameter(positional_encoding, requires_grad=False)
    
    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)
        embedded += self.positional_encoding[:, :x.size(1), :]
        encoded = self.transformer_encoder(embedded)
        logits = self.fc_out(encoded)
        
        return logits

def mask_tokens(inputs, tokenizer, mask_token_id, vocab_size, mask_probability=0.15):
    """
    Prepare masked tokens inputs/labels for masked language modeling (MLM).
    - inputs: Tensor of token ids [batch_size, seq_len]
    - mask_token_id: Token ID to use for masking (usually for the MASK token)
    - vocab_size: Total number of tokens in the vocabulary
    """
    labels = inputs.clone()
    
    # Create a mask with `mask_probability` proportion of True values (tokens to be masked)
    probability_matrix = torch.full(labels.shape, mask_probability)
    mask = torch.bernoulli(probability_matrix).bool()
    
    # Don't mask padding tokens (assuming padding tokens have id 0)
    mask[inputs == 0] = False
    
    # Mask tokens - replace with mask_token_id
    inputs[mask] = mask_token_id
    
    # Set labels for masked tokens, leave labels for non-masked tokens as -100 (ignored in loss computation)
    labels[~mask] = -100
    
    return inputs, labels

def train_mlm(model, data_loader, criterion, optimizer, mask_token_id, vocab_size, num_epochs):
    model.train()
    
    for epoch in range(num_epochs):
        for batch in data_loader:
            inputs, _ = batch
            masked_inputs, labels = mask_tokens(inputs, None, mask_token_id, vocab_size)
            
            optimizer.zero_grad()
            outputs = model(masked_inputs)
            
            # Compute loss only for masked tokens
            loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Parameters
vocab_size = 28  # 26 letters + 1 for unknown + 1 for padding
max_seq_len = 10  # Maximum length of the word
d_model = 128  # Embedding dimension
nhead = 8  # Number of attention heads
num_layers = 3  # Number of transformer layers
dim_feedforward = 512  # Dimension of feedforward layers
mask_token_id = vocab_size - 1  # Assume last ID is used for mask token
num_epochs = 10
learning_rate = 0.001

# Initialize model, criterion, and optimizer
model = HangmanTransformerWithSinCos(vocab_size, max_seq_len, d_model, nhead, num_layers, dim_feedforward)
criterion = nn.CrossEntropyLoss(ignore_index=-100)  # Ignore padding and non-masked tokens in the loss computation
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Assume data_loader is defined with batches of Hangman game states
# train_mlm(model, data_loader, criterion, optimizer, mask_token_id, vocab_size, num_epochs)
