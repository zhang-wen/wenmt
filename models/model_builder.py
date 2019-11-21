import copy
import wargs
import torch.nn as nn
from tools.utils import wlog

''' NMT model with encoder and decoder '''
class NMTModel(nn.Module):

    def __init__(self, encoder, decoder, generator, bowMapper=None, multigpu=False):

        self.multigpu = multigpu
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bowMapper = bowMapper
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(src, src_mask)

    def decode(self, trg, enc_output, src_mask):
        return self.decoder(trg, enc_output, src_mask)

    def generate(self, model_output):
        return self.generator(model_output)

    def forward(self, src, trg, src_mask=None, trg_mask=None, ss_prob=1.):
        '''
            src:        [batch_size, src_len]
            trg:        [batch_size, trg_len]
            src_mask:   [batch_size, src_len]
            trg_mask:   [batch_size, trg_len]
        Returns:
            * decoder output:   [trg_len, batch_size, hidden]
            * attention:        [batch_size, trg_len, src_len]
        '''

        contexts = None
        if wargs.encoder_type == 'gru':
            enc_output = self.encoder(src, src_mask)    # batch_size, max_L, hidden_size
            results = self.decoder(enc_output, trg, src_mask, trg_mask, ss_prob=ss_prob)
            logits, attends, contexts = results['logit'], results['attend'], results['context']
        if wargs.encoder_type == 'att':
            enc_output, _ = self.encoder(src, src_mask)
            logits, _, attends = self.decoder(trg, enc_output, src_mask)
            attends = attends.mean(dim=1)
            contexts = (attends[:, :, :, None] * enc_output[:, None, :, :]).sum(2)
            contexts = contexts * trg_mask[:, :, None]
            #print('contexts: {}'.format(contexts.size()))

        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None

        return {
            'logit': logits,
            'attend': attends,
            'context': contexts
        }


def build_encoder(src_emb):

    if wargs.encoder_type == 'gru':
        from models.gru_encoder import StackedGRUEncoder
        return StackedGRUEncoder(src_emb = src_emb,
                                 enc_hid_size = wargs.d_enc_hid,
                                 dropout_prob = wargs.rnn_dropout,
                                 n_layers = wargs.n_enc_layers)
    if wargs.encoder_type == 'att':
        from models.self_att_model import SelfAttEncoder, SelfAttEncoderLayer, \
                PositionwiseFeedForward, clones
        from models.attention import MultiHeadedAttention
        c = copy.deepcopy
        attn = MultiHeadedAttention(h=wargs.n_head, d_model=wargs.d_model, dropout=wargs.att_dropout)
        ff = PositionwiseFeedForward(d_model=wargs.d_model, d_ff=wargs.d_ff_filter,
                                     dropout=wargs.relu_dropout)
        return SelfAttEncoder(src_emb=src_emb,
                              layer=SelfAttEncoderLayer(wargs.d_model, c(attn), c(ff),
                                                        dropout=wargs.residual_dropout),
                              N=wargs.n_enc_layers)

def build_decoder(trg_emb):

    if wargs.encoder_type == 'gru':
        from models.gru_decoder import StackedGRUDecoder
        return StackedGRUDecoder(trg_emb = trg_emb,
                                 enc_hid_size = wargs.d_enc_hid,
                                 dec_hid_size = wargs.d_dec_hid,
                                 n_layers = wargs.n_dec_layers,
                                 attention_type = wargs.attention_type,
                                 rnn_dropout_prob = wargs.rnn_dropout,
                                 out_dropout_prob = wargs.output_dropout)
    if wargs.decoder_type == 'att':
        from models.self_att_model import SelfAttDecoder, SelfAttDecoderLayer, \
                PositionwiseFeedForward, clones
        from models.attention import MultiHeadedAttention
        c = copy.deepcopy
        attn = MultiHeadedAttention(h=wargs.n_head, d_model=wargs.d_model, dropout=wargs.att_dropout)
        wlog('clones -> {}'.format(2))
        ff = PositionwiseFeedForward(d_model=wargs.d_model, d_ff=wargs.d_ff_filter,
                                     dropout=wargs.relu_dropout)
        return SelfAttDecoder(trg_emb=trg_emb,
                              layer=SelfAttDecoderLayer(wargs.d_model, c(attn), c(attn), c(ff),
                                                        dropout=wargs.residual_dropout),
                              N=wargs.n_enc_layers)

def build_NMT(src_emb, trg_emb):

    encoder = build_encoder(src_emb)
    decoder = build_decoder(trg_emb)

    from models.self_att_model import Generator
    generator = Generator(wargs.d_model if wargs.decoder_type == 'att' else 2 * wargs.d_enc_hid, trg_emb)
    if wargs.bow_loss is True:
        from models.self_att_model import BowMapper
        bowMapper = BowMapper(wargs.d_model if wargs.decoder_type == 'att' else 2 * wargs.d_enc_hid, trg_emb)
    else: bowMapper = None

    return NMTModel(encoder, decoder, generator, bowMapper)


