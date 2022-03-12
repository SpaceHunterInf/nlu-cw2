import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2seq import utils


class TransformerEncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = MultiHeadAttention(
            self.embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True
        )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = args.attention_dropout
        self.activation_dropout = args.activation_dropout

        self.fc1 = generate_linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = generate_linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, state, encoder_padding_mask):
        """Forward pass of a single Transformer Encoder Layer"""
        residual = state.clone()

        '''
        ___QUESTION-6-DESCRIBE-D-START___
        1.  Add tensor shape annotation to EVERY TENSOR below (NOT just the output tensor)
        2.  What is the purpose of encoder_padding_mask? 
        3.  What will the output shape of `state' Tensor be after multi-head attention?
        '''

        state, _ = self.self_attn(query=state, key=state, value=state, key_padding_mask=encoder_padding_mask)

        #1. encoder_padding_mask.size = [batch_size, seq_len]. seq_len is a fix number
        #which is length you want to unify each sequence as the same.
        # state.size (before self.self_attn) = [src_time_steps, batch_size, num_features]
        #2. The preprocessing includes padding the input sequence until
        #they have the same length. The encoder_padding_mask ensures the model
        #does not pay any attention to these paddings.
        #3. The size after would be [src_time_steps, batch_size, num_features]
        '''
        ___QUESTION-6-DESCRIBE-D-END___
        '''

        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.self_attn_layer_norm(state)

        residual = state.clone()
        state = F.relu(self.fc1(state))
        state = F.dropout(state, p=self.activation_dropout, training=self.training)
        state = self.fc2(state)
        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.final_layer_norm(state)

        return state


class TransformerDecoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout = args.attention_dropout
        self.activation_dropout = args.activation_dropout
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        self.self_attn = MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_attn_heads=args.decoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True
        )

        self.encoder_attn = MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_attn_heads=args.decoder_attention_heads,
            kdim=args.encoder_embed_dim,
            vdim=args.encoder_embed_dim,
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
        )

        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = generate_linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = generate_linear(args.decoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.need_attn = True

    def forward(self,
                state,
                encoder_out=None,
                encoder_padding_mask=None,
                incremental_state=None,
                prev_self_attn_state=None,
                self_attn_mask=None,
                self_attn_padding_mask=None,
                need_attn=False,
                need_head_weights=False):
        """Forward pass of a single Transformer Decoder Layer"""

        # need_attn must be True if need_head_weights
        need_attn = True if need_head_weights else need_attn

        residual = state.clone()
        state, _ = self.self_attn(query=state,
                                  key=state,
                                  value=state,
                                  key_padding_mask=self_attn_padding_mask,
                                  need_weights=False,
                                  attn_mask=self_attn_mask)
        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.self_attn_layer_norm(state)

        residual = state.clone()
        '''
        ___QUESTION-6-DESCRIBE-E-START___
        1.  Add tensor shape annotation to EVERY TENSOR below (NOT just the output tensor)
        2.  How does encoder attention differ from self attention? 
        3.  What is the difference between key_padding_mask and attn_mask? 
        4.  If you understand this difference, then why don't we need to give attn_mask here?
        '''
        state, attn = self.encoder_attn(query=state,
                                        key=encoder_out,
                                        value=encoder_out,
                                        key_padding_mask=encoder_padding_mask,
                                        need_weights=need_attn or (not self.training and self.need_attn))
        #1.state.size = [tgt_time_steps, batch_size, embed_dim]
        #attn.size = [num_heads, batch_size, tgt_time_steps, encoder_out.size(0)]
        #encoder_out = [tgt_time_steps, batch_size, embed_dim]
        #2. self-attention only exist within a decoder or encoder. However, encoder attention
        #is using the information from encoder in decoder. It's taking the key and value of encoder_out
        #and use the query of the decoder hidden state. It's trying to find out the relationship
        #between tokens in source input and the current output of the decoder.
        #3. key_padding_mask is preventing the model pay attention to the paddings we add to unify the
        #length of the sequences. attn_mask preventing th decoder to look ahead of and pay attention to 
        # future words we've not generated yet.
        #4. Since attn_mask is already used in self-attention layer before, we don't need to re-apply it.
        '''
        ___QUESTION-6-DESCRIBE-E-END___
        '''

        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.encoder_attn_layer_norm(state)

        residual = state.clone()
        state = F.relu(self.fc1(state))
        state = F.dropout(state, p=self.activation_dropout, training=self.training)
        state = self.fc2(state)
        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.final_layer_norm(state)

        return state, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention"""
    def __init__(self,
                 embed_dim,
                 num_attn_heads,
                 kdim=None,
                 vdim=None,
                 dropout=0.,
                 self_attention=False,
                 encoder_decoder_attention=False):
        '''
        ___QUESTION-7-MULTIHEAD-ATTENTION-NOTE
        You shouldn't need to change the __init__ of this class for your attention implementation
        '''
        super().__init__()
        self.embed_dim = embed_dim
        self.k_embed_size = kdim if kdim else embed_dim
        self.v_embed_size = vdim if vdim else embed_dim

        self.num_heads = num_attn_heads
        self.attention_dropout = dropout
        self.head_embed_size = embed_dim // num_attn_heads  # this is d_k in the paper
        self.head_scaling = math.sqrt(self.head_embed_size)

        self.self_attention = self_attention
        self.enc_dec_attention = encoder_decoder_attention

        kv_same_dim = self.k_embed_size == embed_dim and self.v_embed_size == embed_dim
        assert self.head_embed_size * self.num_heads == self.embed_dim, "Embed dim must be divisible by num_heads!"
        assert not self.self_attention or kv_same_dim, "Self-attn requires query, key and value of equal size!"
        assert self.enc_dec_attention ^ self.self_attention, "One of self- or encoder- attention must be specified!"

        self.k_proj = nn.Linear(self.k_embed_size, embed_dim, bias=True)
        self.v_proj = nn.Linear(self.v_embed_size, embed_dim, bias=True)
        self.q_proj = nn.Linear(self.k_embed_size, embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        # Xavier initialisation
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self,
                query,
                key,
                value,
                key_padding_mask=None,
                attn_mask=None,
                need_weights=True):

        # Get size features
        tgt_time_steps, batch_size, embed_dim = query.size()
        assert self.embed_dim == embed_dim

        '''
        ___QUESTION-7-MULTIHEAD-ATTENTION-START
        Implement Multi-Head attention  according to Section 3.2.2 of https://arxiv.org/pdf/1706.03762.pdf.
        Note that you will have to handle edge cases for best model performance. Consider what behaviour should
        be expected if attn_mask or key_padding_mask are given?
        '''

        # attn is the output of MultiHead(Q,K,V) in Vaswani et al. 2017
        # attn must be size [tgt_time_steps, batch_size, embed_dim]
        # attn_weights is the combined output of h parallel heads of Attention(Q,K,V) in Vaswani et al. 2017
        # attn_weights must be size [num_heads, batch_size, tgt_time_steps, key.size(0)]
        # TODO: REPLACE THESE LINES WITH YOUR IMPLEMENTATION ------------------------ CUT
        #attn = torch.zeros(size=(tgt_time_steps, batch_size, embed_dim))
        #attn_weights = torch.zeros(size=(self.num_heads, batch_size, tgt_time_steps, -1)) if need_weights else None
        # print('inside multihead attention')
        # print(key.size(0))
        # print(key.size())
        # print(tgt_time_steps)

        #Step 1 Projection of Q, K, V Matrices

        projected_k = self.k_proj(key) # projected_k.size = [tgt_time_steps, batch_size, embed_dim]
        projected_v = self.v_proj(value) # projected_v.size = [tgt_time_steps, batch_size, embed_dim]
        projected_q = self.q_proj(query) # projected_q.size = [tgt_time_steps, batch_size, embed_dim]
        # print(projected_k.size())
        projected_k = projected_k.contiguous().view(-1, batch_size, self.num_heads, self.head_embed_size)
        projected_v = projected_v.contiguous().view(-1, batch_size, self.num_heads, self.head_embed_size)
        projected_q = projected_q.contiguous().view(-1, batch_size, self.num_heads, self.head_embed_size)
        #Split concatenated embeddings into each heads

        keys = projected_k.contiguous().view(-1, batch_size * self.num_heads, self.head_embed_size)
        values = projected_v.contiguous().view(-1, batch_size * self.num_heads, self.head_embed_size)
        query = projected_q.contiguous().view(-1, batch_size * self.num_heads, self.head_embed_size)
        #Regroup batch size and heads for matrix multiplication

        transposed_keys = keys.transpose(0,1).contiguous().transpose(1,2).contiguous() #transposed_keys.size = [batch_size * self.num_heads, self.head_embed_size, tgt_time_steps]
        query = query.transpose(0,1).contiguous() # query.size = [batch_size * self.num_heads, tgt_time_steps, self.head_embed_size]
        values = values.transpose(0,1).contiguous() # values.size = [batch_size * self.num_heads, tgt_time_steps, self.head_embed_size]

        #Step 2 Calculate Attention weightes with regards to masks
        attention_kqv = torch.bmm(query, transposed_keys) / self.head_scaling #attention_kqv.size = [batch_size * self.num_heads, tgt_time_steps, tgt_time_steps]
        # print('detect nan in attention_kqv before mask')
        # print(torch.isnan(attention_kqv).any())
        if key_padding_mask != None:
            #key_padding.size = [batch_size, tgt_time_step]
            key_padding_mask = key_padding_mask.unsqueeze(1).repeat(self.num_heads, 1, 1) # key_padding_mask.size = [self.num_heads* batch_size, 1, tgt_time_steps]
            attention_kqv.masked_fill(key_padding_mask, float('-inf'))
            # print('detect nan in attention_kqv after key mask')
            # print(torch.isnan(attention_kqv).any())

        if attn_mask != None:
            #attn_mask.size = [tgt_time_steps, tgt_time_steps]
            attn_mask = attn_mask.unsqueeze(0)# attn_mask = [1,tgt_time_steps, tgt_time_steps]
            attention_kqv = torch.add(attention_kqv, attn_mask)
            # print('detect nan in attention_kqv after key attn_mask')
            # print(torch.isnan(attention_kqv).any())

        #Step 3 Calcualte final attentional outputs
        # print(attention_kqv.size())
        # print(values.size())
        # print('dected nan in value')
        # print(torch.isnan(values).any())
        attention_kqv = torch.softmax(attention_kqv, -1)
        multi_head = torch.bmm(attention_kqv, values) #multi_head.size = [batch_size * self.num_heads, tgt_time_steps, self.head_embed_size]
        # print('detect nan in attn in value product')
        # print(torch.isnan(multi_head).any())
        if torch.isnan(multi_head).any():
            # print(values)
            # print(attention_kqv)
            # print(multi_head)
            raise SystemExit('error in code want to exit')
        multi_head = multi_head.transpose(0,1) #multi_head.size = [tgt_time_steps, batch_size * self.num_heads, self.head_embed_size]
        #print(multi_head.size())
        multi_head = multi_head.view(-1, batch_size, self.num_heads, self.head_embed_size)
        multi_head = multi_head.contiguous().view(-1, batch_size, embed_dim)
        multi_head = self.out_proj(multi_head) #multi_head.size = [tgt_time_steps, batch_size, embed_dim]

        if need_weights:
            attn_weights = attention_kqv.view(self.num_heads, batch_size, -1, key.size(0))
            # print('detect nan in weights')
            # print(torch.isnan(attn_weights).any())
            #print('attn_weights check')
            #print(attn_weights.size())
        else:
            attn_weights = None
        
        attn = multi_head
        # TODO: --------------------------------------------------------------------- CUT

        '''
        ___QUESTION-7-MULTIHEAD-ATTENTION-END
        '''
        # print('detect nan in attn')
        # print(torch.isnan(attn).any())
        return attn, attn_weights


class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        self.weights = PositionalEmbedding.get_embedding(init_size, embed_dim, padding_idx)
        self.register_buffer("_float_tensor", torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embed_dim, padding_idx=None):
        half_dim = embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embed_dim % 2 == 1:
            # Zero pad in specific mismatch case
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0.
        return emb

    def forward(self, inputs, incremental_state=None, timestep=None):
        batch_size, seq_len = inputs.size()
        max_pos = self.padding_idx + 1 + seq_len

        if self.weights is None or max_pos > self.weights.size(0):
            # Expand embeddings if required
            self.weights = PositionalEmbedding.get_embedding(max_pos, self.embed_dim, self.padding_idx)
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            #   Positional embed is identical for all tokens during single step decoding
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            return self.weights.index_select(index=self.padding_idx + pos, dim=0).unsqueeze(1).repeat(batch_size, 1, 1)

        # Replace non-padding symbols with position numbers from padding_idx+1 onwards.
        mask = inputs.ne(self.padding_idx).int()
        positions = (torch.cumsum(mask, dim=1).type_as(inputs) * mask).long() + self.padding_idx

        # Lookup positional embeddings for each position and return in shape of input tensor w/o gradient
        return self.weights.index_select(0, positions.view(-1)).view(batch_size, seq_len, -1).detach()


def LayerNorm(normal_shape, eps=1e-5):
    return torch.nn.LayerNorm(normalized_shape=normal_shape, eps=eps, elementwise_affine=True)


def fill_with_neg_inf(t):
    return t.float().fill_(float('-inf')).type_as(t)


def generate_embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def generate_linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m
