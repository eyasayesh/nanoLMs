from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import tiktoken
import time

#-----------------------------------------------------------------------


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        #key, query, value projections for all heads
        # instead of having 3 // FFN you can have stack them into one FFN
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        #output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization?
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # mask -> called 'bias' in openAI/HF implementations (sticking to that naming convertion)
        # basically forces the model to be causal
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size,config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        
    def forward(self,x):
            B, T, C = x.size() #batch size, sequence length (time), embedding dimensions (channels) = n_embd*
            #calculate the query, key, and values for all heads in the batch and concatenate them together
            # nh is the number of heads, hs is the head size, C = nh * hs
            qkv = self.c_attn (x)
            q,k,v = qkv.split(self.n_embd, dim = 2)
            k = k.view(B,T,self.n_head, C // self.n_head).transpose(1,2) #(B, nh, T, hs)
            q = q.view(B,T,self.n_head, C // self.n_head).transpose(1,2) #(B, nh, T, hs)
            v = v.view(B,T,self.n_head, C // self.n_head).transpose(1,2) #(B, nh, T, hs)
            # attention (materializes the large (T,T) matrix for all the queries and keys)
            #scaled self attention
            att = (q @ k.transpose(-2,-1)) *(1.0/ math.sqrt(k.size(-1))) #(B, nh, T, T)
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim = -1)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) = (B, nh, T, hs)
            y = y.transpose(1,2).contiguous().view(B,T,C) #concatenate all head dimensions
            #output
            y = self.c_proj(y)
            return y


class MLP(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate= 'tanh')
        self.c_proj = nn.Linear(config.n_embd*4, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        #attention
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        #feed forqard
        self.mlp = MLP(config)

    def forward(self, x):
        #adding the x is the residual connection, layer normalization occurs
        #before passing into the next subnetwork
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            #word token embedding
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            #word positional embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd),
            #n layers of Blocks
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            #layer norm
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        #go from a embedded space to token space
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        #weight sharing scheme (consistent with the original attention is all you need paeper)
        self.transformer.wte.weight = self.lm_head.weight

        # init params 
        self.apply(self._init_weights)
    
    #default initialization follows the gpt2 paper
    def _init_weights(self,module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std += (2*self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)

    def forward(self,idx, targets = None):
        #idx is of shape (B,T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"
        #forward the token and the position embeddings
        pos = torch.arange(0,T, dtype=torch.long, device = idx.device)
        pos_emb = self.transformer.wpe(pos) #positional embeddings of shape (T, n_embed)
        tok_emb = self.transformer.wte(idx) #token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb #broadcast pos embedding by adding a B =1 dimension
        #forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        #forward the final layer norm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))
        return logits, loss



    
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 Model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        #n_layer, n_head, and n_embd are determined 
        config_args = {
            'gpt2' : dict(n_layer = 12, n_head = 12, n_embd = 768), #124M params,
            'gpt2-medium' : dict(n_layer = 24, n_head = 16, n_embd = 1024), #124M params,
            'gpt2-large' : dict(n_layer = 36, n_head = 20, n_embd = 1280), #124M params,
            'gpt2-xl' : dict(n_layer = 48, n_head = 25, n_embd = 1600), #124M params
        }[model_type]
        config_args['vocab_size'] = 50257 #always 5027 for GPT model checkpoints
        config_args['block_size'] = 1024 #always 1024 for GPT model checkpoints
        #create a from scratch initialized gpt model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] #discard this mask / buffer

        #init a huggingface transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and mathc in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias') ]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        #hard coded transpose of certain weights

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                #vanilla copy over other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
            
        return model


class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
    
        #at initload tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T) } batches")

        #state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B*T + 1]
        x = (buf[:-1]).view(B,T)
        y = (buf[1:]).view(B,T)

        self.current_position += B*T
        #if loading position would be out of bounds
        if self.current_position + B*T + 1 > len(self.tokens):
            self.current_position = 0

        return x, y

#----------------------------------------
#autodetect device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(42)
torch.cuda.manual_seed(42)

torch.set_float32_matmul_precision('high')

#init model
model = GPT(GPTConfig())
model.to(device)
model.eval()

model = torch.compile(model)

train_loader = DataLoaderLite(B=16, T = 1024)

#optimize
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    t0 = time.time()
    x,y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    with torch.autocast(device_type = device, dtype = torch.bfloat16):
        logits, loss = model(x,y)
        #import code; code.interact(local = locals())
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1-t0)*1000 #time diff in ms
    tokens_per_sec = (train_loader.B * train_loader.T)/ (dt/1000)
    print(f"step {i}, loss: {loss.item()}, dt = {dt:.2f} msm tok/s = {tokens_per_sec:.2f}")

import sys; sys.exit()


tokens = enc.encode("Hello, I am a langauge model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) #(5,8)
x = tokens.to(device)

#generate! right now x is (B,T) where B = 5, T = 8
#set the seed to 42

while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x) #(B,T,n_embd)
        #take the logits at the last positions
        logits = logits[:,-1, :] # (B, vocab_size)
        probs = F.softmax(logits, dim = -1)
        #do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5,50), topl_indices is (5,50)
        topk_probs, topk_indices = torch.topk(probs,50,dim=-1)
        #select a token from the top-k probabilites
        ix = torch.multinomial(topk_probs, 1) # (B,1)
        #gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
