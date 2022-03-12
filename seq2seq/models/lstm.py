import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2seq import utils
from seq2seq.models import Seq2SeqModel, Seq2SeqEncoder, Seq2SeqDecoder
from seq2seq.models import register_model, register_model_architecture


@register_model('lstm')
class LSTMModel(Seq2SeqModel):
    """ Defines the sequence-to-sequence model class. """

    def __init__(self,
                 encoder,
                 decoder):

        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--encoder-embed-dim', type=int, help='encoder embedding dimension')
        parser.add_argument('--encoder-embed-path', help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-hidden-size', type=int, help='encoder hidden size')
        parser.add_argument('--encoder-num-layers', type=int, help='number of encoder layers')
        parser.add_argument('--encoder-bidirectional', help='bidirectional encoder')
        parser.add_argument('--encoder-dropout-in', help='dropout probability for encoder input embedding')
        parser.add_argument('--encoder-dropout-out', help='dropout probability for encoder output')

        parser.add_argument('--decoder-embed-dim', type=int, help='decoder embedding dimension')
        parser.add_argument('--decoder-embed-path', help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-hidden-size', type=int, help='decoder hidden size')
        parser.add_argument('--decoder-num-layers', type=int, help='number of decoder layers')
        parser.add_argument('--decoder-dropout-in', type=float, help='dropout probability for decoder input embedding')
        parser.add_argument('--decoder-dropout-out', type=float, help='dropout probability for decoder output')
        parser.add_argument('--decoder-use-attention', help='decoder attention')
        parser.add_argument('--decoder-use-lexical-model', help='toggle for the lexical model')

    @classmethod
    def build_model(cls, args, src_dict, tgt_dict):
        """ Constructs the model. """
        base_architecture(args)
        encoder_pretrained_embedding = None
        decoder_pretrained_embedding = None

        # Load pre-trained embeddings, if desired
        if args.encoder_embed_path:
            encoder_pretrained_embedding = utils.load_embedding(args.encoder_embed_path, src_dict)
        if args.decoder_embed_path:
            decoder_pretrained_embedding = utils.load_embedding(args.decoder_embed_path, tgt_dict)

        # Construct the encoder
        print('Encoder layer:{}'.format(args.encoder_num_layers))
        print('Decoder layer:{}'.format(args.decoder_num_layers))
        encoder = LSTMEncoder(dictionary=src_dict,
                              embed_dim=args.encoder_embed_dim,
                              hidden_size=args.encoder_hidden_size,
                              num_layers=args.encoder_num_layers,
                              bidirectional=args.encoder_bidirectional,
                              dropout_in=args.encoder_dropout_in,
                              dropout_out=args.encoder_dropout_out,
                              pretrained_embedding=encoder_pretrained_embedding)

        # Construct the decoder
        decoder = LSTMDecoder(dictionary=tgt_dict,
                              embed_dim=args.decoder_embed_dim,
                              hidden_size=args.decoder_hidden_size,
                              num_layers=args.decoder_num_layers,
                              dropout_in=args.decoder_dropout_in,
                              dropout_out=args.decoder_dropout_out,
                              pretrained_embedding=decoder_pretrained_embedding,
                              use_attention=bool(eval(args.decoder_use_attention)),
                              use_lexical_model=bool(eval(args.decoder_use_lexical_model)))
        return cls(encoder, decoder)


class LSTMEncoder(Seq2SeqEncoder):
    """ Defines the encoder class. """

    def __init__(self,
                 dictionary,
                 embed_dim=64,
                 hidden_size=64,
                 num_layers=1,
                 bidirectional=True,
                 dropout_in=0.25,
                 dropout_out=0.25,
                 pretrained_embedding=None):

        super().__init__(dictionary)
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.output_dim = 2 * hidden_size if bidirectional else hidden_size

        if pretrained_embedding is not None:
            self.embedding = pretrained_embedding
        else:
            self.embedding = nn.Embedding(len(dictionary), embed_dim, dictionary.pad_idx)

        dropout_lstm = dropout_out if num_layers > 1 else 0.
        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout_lstm,
                            bidirectional=bool(bidirectional))

    def forward(self, src_tokens, src_lengths):
        """ Performs a single forward pass through the instantiated encoder sub-network. """
        # Embed tokens and apply dropout
        batch_size, src_time_steps = src_tokens.size()
        #print('batch size:{}'.format(batch_size))
        src_embeddings = self.embedding(src_tokens)
        _src_embeddings = F.dropout(src_embeddings, p=self.dropout_in, training=self.training)

        # Transpose batch: [batch_size, src_time_steps, num_features] -> [src_time_steps, batch_size, num_features]
        src_embeddings = _src_embeddings.transpose(0, 1)

        # Pack embedded tokens into a PackedSequence
        packed_source_embeddings = nn.utils.rnn.pack_padded_sequence(src_embeddings, src_lengths.data.tolist())

        # Pass source input through the recurrent layer(s)
        if self.bidirectional:
            state_size = 2 * self.num_layers, batch_size, self.hidden_size
        else:
            state_size = self.num_layers, batch_size, self.hidden_size

        hidden_initial = src_embeddings.new_zeros(*state_size)
        context_initial = src_embeddings.new_zeros(*state_size)

        packed_outputs, (final_hidden_states, final_cell_states) = self.lstm(packed_source_embeddings,
                                                                             (hidden_initial, context_initial))

        # Unpack LSTM outputs and optionally apply dropout (dropout currently disabled)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, padding_value=0.)
        lstm_output = F.dropout(lstm_output, p=self.dropout_out, training=self.training)
        assert list(lstm_output.size()) == [src_time_steps, batch_size, self.output_dim]  # sanity check

        '''
        ___QUESTION-1-DESCRIBE-A-START___
        1.  Add tensor shape annotation to each of the output tensor
            output tensor means tensors on the left hand side of "="
            e.g., 
                sent_tensor = create_sentence_tensor(...) 
                # sent_tensor.size = [batch, sent_len, hidden]
        2.  Describe what happens when self.bidirectional is set to True. 
        3.  What is the difference between final_hidden_states and final_cell_states?
        '''
        if self.bidirectional:
            def combine_directions(outs):
                return torch.cat([outs[0: outs.size(0): 2], outs[1: outs.size(0): 2]], dim=2)
            final_hidden_states = combine_directions(final_hidden_states)
            final_cell_states = combine_directions(final_cell_states)
        
        #1.final_hidden_states.size = [D*self.num_layers, batch_size, self.hidden_size]
        # final_cell_state.size = [D*self.num_layers, batch_size, self.hidden_size]. D is 2 if 
        # self.bidirectional is True, otherwise D = 1.
        #2.When self.bidirectional is True the nn.LSTM model become a bidirectional LSTM, we double
        # the initial cell state and hidden state (1 for left to right, the other for right to left)
        # this will also double the output hidden state and cell state size, and we'll concatenate and
        # flatten them.
        #3.Hidden states are states outputed by layers of LSTM cells, Cell states are states that LSTM cells
        # retains and passed on to other cells as long-term memory.
        '''___QUESTION-1-DESCRIBE-A-END___'''

        # Generate mask zeroing-out padded positions in encoder inputs
        src_mask = src_tokens.eq(self.dictionary.pad_idx)

        return {'src_embeddings': _src_embeddings.transpose(0, 1),
                'src_out': (lstm_output, final_hidden_states, final_cell_states),
                'src_mask': src_mask if src_mask.any() else None}


class AttentionLayer(nn.Module):
    """ Defines the attention layer class. Uses Luong's global attention with the general scoring function. """
    def __init__(self, input_dims, output_dims):
        super().__init__()
        # Scoring method is 'general'
        self.src_projection = nn.Linear(input_dims, output_dims, bias=False)
        self.context_plus_hidden_projection = nn.Linear(input_dims + output_dims, output_dims, bias=False)

    def forward(self, tgt_input, encoder_out, src_mask):
        # tgt_input has shape = [batch_size, input_dims]
        # encoder_out has shape = [src_time_steps, batch_size, output_dims]
        # src_mask has shape = [src_time_steps, batch_size]
        # Get attention scores
        # [batch_size, src_time_steps, output_dims]
        encoder_out = encoder_out.transpose(1, 0)

        # [batch_size, 1, src_time_steps]
        attn_scores = self.score(tgt_input, encoder_out)

        '''
        ___QUESTION-1-DESCRIBE-B-START___
        1.  Add tensor shape annotation to each of the output tensor
            output tensor means tensors on the left hand side of "="
            e.g., 
                sent_tensor = create_sentence_tensor(...) 
                # sent_tensor.size = [batch, sent_len, hidden]
        2.  Describe how the attention context vector is calculated. 
        3.  Why do we need to apply a mask to the attention scores?
        '''
        if src_mask is not None:
            #print(src_mask.size())
            src_mask = src_mask.unsqueeze(dim=1)
            attn_scores.masked_fill_(src_mask, float('-inf'))

            #print(src_mask.size())
            #print(encoder_out.size())

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_context = torch.bmm(attn_weights, encoder_out).squeeze(dim=1)
        context_plus_hidden = torch.cat([tgt_input, attn_context], dim=1)
        attn_out = torch.tanh(self.context_plus_hidden_projection(context_plus_hidden))
        #1.src_mask.size = [batch_size, 1, src_time_steps]
        # atten_weights.size = [batch_size, 1, src_time_steps]
        # atten_context.size = [batch_size, output_dims]
        # context_plus_hidden = [batch_size, input_dims + output_dims]
        # attn_out = [batch_size, output_dims]
        #2. Firstly, we compute the alignment vector for each batch by scoring
        # current encoder output and current target input against current encoder
        # output and target input over all time steps. Then we take softmax gives us
        # a distribution in (0,1]. Then use torch.bmm we obtained context vector as
        # a weighted average for each batches.  
        #3. Remember the fact we are using decoder output as the training input of our
        #model, if we do not apply the mask, our model could 'cheat' for best output
        #weights by directly copying from golden label, which is an undesired behaviour.
        '''___QUESTION-1-DESCRIBE-B-END___'''

        return attn_out, attn_weights.squeeze(dim=1)

    def score(self, tgt_input, encoder_out):
        """ Computes attention scores. """

        '''
        ___QUESTION-1-DESCRIBE-C-START___
        1.  Add tensor shape annotation to each of the output tensor
        2.  How are attention scores calculated? 
        3.  What role does batch matrix multiplication (i.e. torch.bmm()) play in aligning encoder and decoder representations?
        '''
        projected_encoder_out = self.src_projection(encoder_out).transpose(2, 1)
        attn_scores = torch.bmm(tgt_input.unsqueeze(dim=1), projected_encoder_out)
        #1.projected_encoder_out.size = [batch_size, output_dims, src_time_steps]
        # tgt_input.unsqueeze.size = [batch_size, 1, input_dims]
        # input_dim == output_dim ==> attn_score.size = [batch_size, 1, src_time_steps]
        #2.The score is computed using general scoring function from Luong's paper.
        #3.The src_projection acts as W_a matrix and then the projected_encoder_out 
        #is the product of this matrix and encoder output vector at current time step.
        #We do this for every time step, we obtain a matrix containing projected 
        #encoder output over all timesteps
        #Thus we need to use torch.bmm to multiply target hidden state vector to obtain
        #final score as a vector over all time step to compute softmax later.
        '''___QUESTION-1-DESCRIBE-C-END___'''

        return attn_scores


class LSTMDecoder(Seq2SeqDecoder):
    """ Defines the decoder class. """

    def __init__(self,
                 dictionary,
                 embed_dim=64,
                 hidden_size=128,
                 num_layers=1,
                 dropout_in=0.25,
                 dropout_out=0.25,
                 pretrained_embedding=None,
                 use_attention=True,
                 use_lexical_model=False):

        super().__init__(dictionary)

        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size

        if pretrained_embedding is not None:
            self.embedding = pretrained_embedding
        else:
            self.embedding = nn.Embedding(len(dictionary), embed_dim, dictionary.pad_idx)

        # Define decoder layers and modules
        self.attention = AttentionLayer(hidden_size, hidden_size) if use_attention else None

        self.layers = nn.ModuleList([nn.LSTMCell(
            input_size=hidden_size + embed_dim if layer == 0 else hidden_size,
            hidden_size=hidden_size)
            for layer in range(num_layers)])

        self.final_projection = nn.Linear(hidden_size, len(dictionary))

        self.use_lexical_model = use_lexical_model
        if self.use_lexical_model:
            print('using lexical model')
            # __QUESTION-5: Add parts of decoder architecture corresponding to the LEXICAL MODEL here
            # TODO: --------------------------------------------------------------------- CUT
            self.hidden_proj = nn.Linear(embed_dim, embed_dim, bias=False) # W*f_t
            self.lexical_out_proj = nn.Linear(embed_dim, len(dictionary), bias=True) # W*h^l_t + b^l
            # TODO: --------------------------------------------------------------------- /CUT

    def forward(self, tgt_inputs, encoder_out, incremental_state=None):
        """ Performs the forward pass through the instantiated model. """
        # Optionally, feed decoder input token-by-token
        if incremental_state is not None:
            tgt_inputs = tgt_inputs[:, -1:]

        # __QUESTION-5 : Following code is to assist with the LEXICAL MODEL implementation
        # Recover encoder input
        src_embeddings = encoder_out['src_embeddings']

        src_out, src_hidden_states, src_cell_states = encoder_out['src_out']
        src_mask = encoder_out['src_mask']
        src_time_steps = src_out.size(0)

        # Embed target tokens and apply dropout
        batch_size, tgt_time_steps = tgt_inputs.size()
        tgt_embeddings = self.embedding(tgt_inputs)
        tgt_embeddings = F.dropout(tgt_embeddings, p=self.dropout_in, training=self.training)

        # Transpose batch: [batch_size, tgt_time_steps, num_features] -> [tgt_time_steps, batch_size, num_features]
        tgt_embeddings = tgt_embeddings.transpose(0, 1)

        # Initialize previous states (or retrieve from cache during incremental generation)
        '''
        ___QUESTION-1-DESCRIBE-D-START___
        1.  Add tensor shape annotation to each of the output tensor
        2.  Describe how the decoder state is initialized. 
        3.  When is cached_state == None? 
        4.  What role does input_feed play?  
        '''
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            tgt_hidden_states, tgt_cell_states, input_feed = cached_state
        else:
            tgt_hidden_states = [torch.zeros(tgt_inputs.size()[0], self.hidden_size) for i in range(len(self.layers))]
            tgt_cell_states = [torch.zeros(tgt_inputs.size()[0], self.hidden_size) for i in range(len(self.layers))]
            input_feed = tgt_embeddings.data.new(batch_size, self.hidden_size).zero_()
        
        #1. cached_state is not a tensor, just 3 variables below
        # tgt_hidden_states = [len(self.layers), batch_size, self.hidden_size]
        # tgt_cell_states = [len(self.layers), batch_size, self.hidden_size]
        # input_feed = [batch_size, self.hidden_size]
        #2. The decoder states are intialized from cached_states which is the 
        # previous decoder state. If there is no previous decoder states, cache_state
        # will return None, then we intialize all decoder states to zero tensors
        #3. cache_state is none if incremental_state is none, or full_key is not in
        #incremental_state, which means there is no cached previous decoder statesd
        #4. input_feed is the decoder hidden state of previous time step if attention
        # is not used, otherwise it's the attentional hidden state of previous time step
        #by using the attention layer we've defined. Then current token embedding will be  
        #concatenated with it for further computation.
        '''___QUESTION-1-DESCRIBE-D-END___'''

        # Initialize attention output node
        attn_weights = tgt_embeddings.data.new(batch_size, tgt_time_steps, src_time_steps).zero_()
        rnn_outputs = []

        # __QUESTION-5 : Following code is to assist with the LEXICAL MODEL implementation
        # Cache lexical context vectors per translation time-step
        lexical_contexts = []

        for j in range(tgt_time_steps):
            # Concatenate the current token embedding with output from previous time step (i.e. 'input feeding')
            lstm_input = torch.cat([tgt_embeddings[j, :, :], input_feed], dim=1)

            for layer_id, rnn_layer in enumerate(self.layers):
                # Pass target input through the recurrent layer(s)
                tgt_hidden_states[layer_id], tgt_cell_states[layer_id] = \
                    rnn_layer(lstm_input, (tgt_hidden_states[layer_id], tgt_cell_states[layer_id]))

                # Current hidden state becomes input to the subsequent layer; apply dropout
                lstm_input = F.dropout(tgt_hidden_states[layer_id], p=self.dropout_out, training=self.training)

            '''
            ___QUESTION-1-DESCRIBE-E-START___
            1.  Add tensor shape annotation to each of the output tensor
            2.  How is attention integrated into the decoder? 
            3.  Why is the attention function given the previous target state as one of its inputs? 
            4.  What is the purpose of the dropout layer?
            '''
            if self.attention is None:
                input_feed = tgt_hidden_states[-1]
            else:
                input_feed, step_attn_weights = self.attention(tgt_hidden_states[-1], src_out, src_mask)
                attn_weights[:, j, :] = step_attn_weights

                if self.use_lexical_model:
                    # __QUESTION-5: Compute and collect LEXICAL MODEL context vectors here
                    # TODO: --------------------------------------------------------------------- CUT
                    #src_embedding.size = [src_time_steps, batch_size, num_features]
                    src_transposed = src_embeddings.transpose(0,1)
                    #src_embedding.size = [batch_size,src_time_steps, num_features]
                    f_lt = torch.tanh(torch.bmm(step_attn_weights.unsqueeze(1), src_transposed))
                    #f_t.size = [batch_size, 1, num_features]
                    f_lt = f_lt.squeeze(1)
                    #f_t.size = [batch_size, num_features]
                    h_lt = torch.tanh(self.hidden_proj(f_lt)) + f_lt
                    #h_lt.size = [batch_size, num_features]
                    lexical_contexts.append(h_lt)
                    #lexical_context.size will be [tgt_time_steps, batch_size, num_features]
                    # TODO: --------------------------------------------------------------------- /CUT

            input_feed = F.dropout(input_feed, p=self.dropout_out, training=self.training)
            rnn_outputs.append(input_feed)
            #1.input_feed.size = [batch_size, self.hidden_size]
            # step_attn_weights.size = [batch_size, src_time_steps]
            #2.Attention is integerated by the fact that input_feed is
            #calculated as a attention vector and will be concatenated 
            #with next target embeddings and feed into decoder.
            #3. The previous target state is given to calculate the 
            #attention vector of this time step over all time steps
            #in the encoder hidden state output.
            #4. Using dropout layer we randomly change entries in the 
            #attention vector to 0. This could prevent model from overfitting
            #and prediction performance more robust without relying too much
            #on previous states.
            '''___QUESTION-1-DESCRIBE-E-END___'''

        # Cache previous states (only used during incremental, auto-regressive generation)
        utils.set_incremental_state(
            self, incremental_state, 'cached_state', (tgt_hidden_states, tgt_cell_states, input_feed))

        # Collect outputs across time steps
        decoder_output = torch.cat(rnn_outputs, dim=0).view(tgt_time_steps, batch_size, self.hidden_size)

        # Transpose batch back: [tgt_time_steps, batch_size, num_features] -> [batch_size, tgt_time_steps, num_features]
        decoder_output = decoder_output.transpose(0, 1)

        # Final projection
        decoder_output = self.final_projection(decoder_output)

        if self.use_lexical_model:
            # __QUESTION-5: Incorporate the LEXICAL MODEL into the prediction of target tokens here
            # TODO: --------------------------------------------------------------------- CUT
            decoder_output = decoder_output + self.lexical_out_proj(torch.stack(lexical_contexts).transpose(0,1))
            # TODO: --------------------------------------------------------------------- /CUT

        return decoder_output, attn_weights


@register_model_architecture('lstm', 'lstm')
def base_architecture(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 64)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', 64)
    args.encoder_num_layers = getattr(args, 'encoder_num_layers', 1)
    args.encoder_bidirectional = getattr(args, 'encoder_bidirectional', 'True')
    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', 0.25)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', 0.25)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 64)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', 128)
    args.decoder_num_layers = getattr(args, 'decoder_num_layers', 1)
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', 0.25)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', 0.25)
    args.decoder_use_attention = getattr(args, 'decoder_use_attention', 'True')
    args.decoder_use_lexical_model = getattr(args, 'decoder_use_lexical_model', 'False')
