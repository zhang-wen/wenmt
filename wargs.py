# Maximal sequence length in training data
max_seq_len = 256
worse_counter = 0
# 'toy', 'zhen', 'ende', 'deen', 'uyzh'
dataset, model_config = 'ende', 't2t_base'
batch_type = 'token'    # 'sents' or 'tokens', sents is default, tokens will do dynamic batching
#gpu_ids = [7, 4,5,6]
gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]
#gpu_ids = [0, 1]
batch_size = 40 if batch_type == 'sents' else 4096
#gpu_id = None
n_co_models = 1
s_step_decay = 300 * n_co_models
e_step_decay = 3000 * n_co_models
import os
def getdir():
    return os.path.abspath('.') + '/'
print('Working in {}'.format(getdir()))
work_dir=getdir()
''' directory to save model, validation output and test output '''
dir_model, dir_valid, dir_tests = work_dir+'wmodel', work_dir+'wvalid', work_dir+'wtests'
''' vocabulary '''
n_src_vcb_plan, n_trg_vcb_plan, share_vocab = 30000, 30000, False
small, epoch_eval, src_char, char_bleu = False, False, False, False
cased, with_bpe, ref_bpe, use_multi_bleu = False, False, False, True
opt_mode = 'adam'       # 'adadelta', 'adam' or 'sgd'
lr_update_way, param_init_D, learning_rate = 'chen', 'U', 0.001  # 'noam' or 'chen'
beta_1, beta_2, weight_decay, u_gain, adam_epsilon, warmup_steps, chunk_size = 0.9, 0.98, 0., 0.01, 1e-08, 500, 20
max_grad_norm = 5.      # the norm of the gradient exceeds this, renormalize it to max_grad_norm
d_dec_hid, d_model = 512, 512
input_dropout, rnn_dropout, output_dropout = 0.5, 0.3, 0.5
encoder_normalize_before, decoder_normalize_before, max_epochs, max_update = False, False, 15, 30000

if model_config == 't2t_base':
    encoder_type, decoder_type = 'att', 'att'   # 'att', 'gru'
    lr_update_way = 'invsqrt'  # 'noam' or 'chen' or 'invsqrt'
    param_init_D = 'X'      # 'U': uniform , 'X': xavier, 'N': normal
    d_src_emb, d_trg_emb, d_model, d_ff_filter, n_head, n_enc_layers, n_dec_layers = 512, 512, 512, 2048, 8, 6, 6
    input_dropout, att_dropout, relu_dropout, residual_dropout = 0.3, 0., 0., 0.3
    learning_rate, warmup_steps, u_gain, beta_2 = 0.0005, 4000, 0.08, 0.98
    warmup_init_lr, min_lr = 1e-07, 1e-09
    chunk_size, max_grad_norm = 20, 0.
    small = True
if model_config == 't2t_big':
    encoder_type, decoder_type = 'att', 'att'   # 'att', 'gru'
    lr_update_way, param_init_D = 'noam', 'X'
    d_src_emb, d_trg_emb, d_model, d_ff_filter, n_head, n_enc_layers, n_dec_layers = 1024, 1024, 1024, 4096, 16, 6, 6
    input_dropout, att_dropout, relu_dropout, residual_dropout = 0.3, 0.1, 0.1, 0.3
    learning_rate, warmup_steps, u_gain, beta_2 = 0.2, 8000, 0.08, 0.997
    chunk_size, max_grad_norm = 1, 0.
if model_config == 'gru_base':
    encoder_type, decoder_type, attention_type = 'gru', 'gru', 'multihead_additive'
    d_src_emb, d_trg_emb, d_enc_hid, d_dec_hid, n_enc_layers, n_dec_layers = 512, 512, 512, 512, 2, 2
    learning_rate, u_gain, beta_2, adam_epsilon = 0.001, 0.08, 0.999, 1e-6
    s_step_decay, e_step_decay, warmup_steps = 8000, 96000, 8000
    small = True
if model_config == 'gru_big':
    encoder_type, decoder_type, attention_type = 'gru', 'gru', 'multihead_additive'
    d_src_emb, d_trg_emb, d_enc_hid, d_dec_hid, n_enc_layers, n_dec_layers = 1024, 1024, 1024, 1024, 2, 2
    learning_rate, u_gain, beta_2, adam_epsilon = 0.001, 0.08, 0.999, 1e-6
    s_step_decay, e_step_decay, warmup_steps = 8000, 64000, 8000

''' training data '''
dir_data = work_dir+'data/'
if dataset == 'toy':
    val_tst_dir = './data/'
    val_src_suffix, val_ref_suffix, val_prefix, tests_prefix = 'zh', 'en', 'devset1_2.lc', ['devset3.lc']
    #tests_prefix = None
    max_epochs = 50
elif dataset == 'deen':
    dir_data = work_dir+'data.iwslt/'
    #val_tst_dir = '/home/wen/3.data/iwslt14_deen/prep/'
    #val_tst_dir = '/ceph_nmt/wenzhang/2.data/mt/iwslt14_deen/iwslt14.tokenized.de-en/'
    val_tst_dir = '/home/wen/3.data/iwslt14_deen/iwslt14.tokenized.de-en/'
    val_src_suffix, val_ref_suffix, val_prefix, tests_prefix = 'de', 'en', 'valid', ['test']
    d_src_emb, d_trg_emb, d_model, d_ff_filter, n_head, n_enc_layers, n_dec_layers = 512, 512, 512, 1024, 4, 6, 6
    input_dropout, att_dropout, relu_dropout, residual_dropout = 0.3, 0.1, 0.1, 0.3
    learning_rate, warmup_steps, beta_2, adam_epsilon = 0.001, 4000, 0.98, 1e-08
    warmup_init_lr, min_lr = 1e-07, 1e-09
    max_grad_norm = 25.      # the norm of the gradient exceeds this, renormalize it to max_grad_norm
    eval_valid_from, eval_valid_freq = 8000, 1000
    #n_src_vcb_plan, n_trg_vcb_plan = 32009, 22822
    share_vocab = True
    max_epochs, with_bpe, ref_bpe, cased, max_update = 1000, True, True, True, 20000    # False: Case-insensitive BLEU  True: Case-sensitive BLEU
    batch_size = 40 if batch_type == 'sents' else 4096
elif dataset == 'zhen':
    dir_data = work_dir+'data.zhen/'
    val_tst_dir = '/ceph_nmt/wenzhang/2.data/mt/nist_zhen/mfd_1.25M/nist_test_new/'
    #dev_prefix = 'nist02'
    val_src_suffix, val_ref_suffix = 'src.BPE', 'trg.tok.sb'
    val_prefix, tests_prefix = 'mt06_u8', ['mt02_u8', 'mt03_u8', 'mt04_u8', 'mt05_u8', 'mt08_u8']
    input_dropout, att_dropout, relu_dropout, residual_dropout = 0.3, 0.1, 0.1, 0.3
    learning_rate, warmup_steps, beta_2, adam_epsilon = 0.0007, 4000, 0.98, 1e-08
    warmup_init_lr, min_lr = 1e-07, 1e-09
    max_grad_norm = 0.      # the norm of the gradient exceeds this, renormalize it to max_grad_norm
    eval_valid_from, eval_valid_freq = 8000, 1000
    tests_prefix = None
    n_src_vcb_plan, n_trg_vcb_plan = 50000, 50000
    max_epochs, with_bpe, ref_bpe, max_update = 1000, True, False, 30000
elif dataset == 'uyzh':
    #val_tst_dir = '/home5/wen/2.data/mt/uy_zh_300w/devtst/'
    val_tst_dir = '/home/wen/3.corpus/mt/uy_zh_300w/devtst/'
    val_src_suffix, val_src_suffix, val_prefix, tests_prefix = '8kbpe.src', 'uy.src', 'dev700', ['tst861']
elif dataset == 'ende':
    dir_data = work_dir+'data.wmt14ende/'
    #val_tst_dir = '/home/wen/3.corpus/ende37kbpe/'
    val_tst_dir = '/ceph_nmt/wenzhang/2.data/mt/wmt14_ende/ende37kbpe/'
    val_src_suffix, val_ref_suffix = 'tc.en.37kbpe', 'tc.de'
    val_prefix, tests_prefix = 'newstest2013', ['newstest2014']
    input_dropout, att_dropout, relu_dropout, residual_dropout = 0.1, 0.1, 0.1, 0.1
    learning_rate, warmup_steps, beta_2, adam_epsilon = 0.001, 4000, 0.98, 1e-8
    d_src_emb, d_trg_emb, d_model, d_ff_filter, n_head, n_enc_layers, n_dec_layers = 512, 512, 512, 1024, 8, 6, 3
    eval_valid_from, eval_valid_freq = 50000, 10000
    warmup_init_lr, min_lr, chunk_size = 1e-07, 1e-09, 10
    max_grad_norm = 0.      # the norm of the gradient exceeds this, renormalize it to max_grad_norm
    n_src_vcb_plan, n_trg_vcb_plan = 50000, 50000
    share_vocab = True
    max_epochs, cased, with_bpe, ref_bpe, max_update = 1000, True, True, False, 200000
    batch_size = 40 if batch_type == 'sents' else 4096
    #s_step_decay, e_step_decay, warmup_steps = 200000, 1200000, 8000

src_vcb, trg_vcb = dir_data + 'src.vcb', dir_data + 'trg.vcb'
train_prefix, train_src_suffix, train_trg_suffix = 'train', 'src', 'trg'
proj_share_weight = True
position_encoding = True if (encoder_type in ('att') and decoder_type in ('att')) else False

''' validation data '''
dev_max_seq_len = 10000000
inputs_data = dir_data + 'inputs.pt'

''' training '''
epoch_shuffle_train, epoch_shuffle_batch = False, False
sort_k_batches = 500      # 0 for all sort, 1 for no sort
save_one_model = False
select_model_by, test_in_training = 'loss', False   # loss or bleu
start_epoch = 1
trg_bow, emb_loss, bow_loss = True, False, False
grad_accum_count = 1    # accumulate gradient for batch_size * accum_count batches (Transformer)
loss_norm = 'tokens'    # 'sents' or 'tokens', normalization method of the gradient
label_smoothing = 0.1
model_prefix = dir_model + '/model'
best_model = dir_valid + '/best.model.pt' if dir_valid else 'best.model.pt'

''' whether use pretrained model '''
pre_train = None
#pre_train = best_model
fix_pre_params = False
''' display settings '''
n_look, fix_looking = 5, False

''' decoder settings '''
search_mode = 1
len_norm = 2    # 0: no noraml, 1: length normal, 2: alpha-beta
beam_size, alpha_len_norm, beta_cover_penalty = 5, 0.6, 0.
valid_batch_size, test_batch_size = 128, 128
print_att = True

''' Scheduled Sampling of Samy bengio's paper '''
greed_sampling = False
greed_gumbel_noise = 0.5     # None: w/o noise
bleu_sampling = False
bleu_gumbel_noise = 0.5     # None: w/o noise
ss_type = None     # 1: linear decay, 2: exponential decay, 3: inverse sigmoid decay
ss_prob_begin, ss_k = 1., 12.     # k < 1. for exponential decay, k >= 1. for inverse sigmoid decay
if ss_type == 1:
    ss_prob_end = 0.
    ss_decay_rate = (ss_prob_begin - ss_prob_end) / 10.
if ss_type == 2: assert ss_k < 1., 'requires ss_k < 1.'
if ss_type == 3: assert ss_k >= 1., 'requires ss_k >= 1.'

sampling = 'length_limit'     # truncation, length_limit, gumbeling

display_freq = 100 if small else 1000
look_freq = 4000 if small else 5000
batch_size = batch_size * len(gpu_ids)


