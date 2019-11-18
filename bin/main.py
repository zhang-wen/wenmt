#!/usr/bin/env python

import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import torch as tc
from torch import cuda

import wargs
from tools.inputs_handler import *
from tools.inputs import Input
from tools.optimizer import Optim
from models.losser import LabelSmoothingCriterion, BOWLossCriterion, EMBLossCriterion, MultiGPULossCompute
from models.embedding import WordEmbedding
from models.model_builder import build_NMT
from tools.utils import init_dir, wlog

device_ids = wargs.gpu_ids

if device_ids is not None:
    device = tc.device('cuda:{}'.format(device_ids[0]) if cuda.is_available() else 'cpu')
    wlog('Set device {}, will use {} GPUs {}'.format(device_ids[0], len(device_ids), device_ids))

# Check if CUDA is available
if cuda.is_available():
    wlog('CUDA is available, specify device by gpu_ids argument (i.e. gpu_ids=[0, 1, 2])')
else:
    if len(device_ids) > 1:
        wlog('Can not train on multi-gpus, device count: {}'.format(cuda.device_count()))
        sys.exit(0)
    else: wlog('Warning: CUDA is not available, train on CPU')

from trainer import *

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.enabled = True

tc.manual_seed(1111)

def main():

    #if wargs.ss_type is not None: assert wargs.model == 1, 'Only rnnsearch support schedule sample'
    init_dir(wargs.dir_model)
    init_dir(wargs.dir_valid)

    src = os.path.join(wargs.dir_data, '{}.{}'.format(wargs.train_prefix, wargs.train_src_suffix))
    trg = os.path.join(wargs.dir_data, '{}.{}'.format(wargs.train_prefix, wargs.train_trg_suffix))
    src, trg = os.path.abspath(src), os.path.abspath(trg)
    vocabs = {}
    wlog('\nPreparing source vocabulary from {} ... '.format(src))
    src_vocab = extract_vocab(src, wargs.src_vcb, wargs.n_src_vcb_plan,
                              wargs.max_seq_len, char=wargs.src_char)
    wlog('\nPreparing target vocabulary from {} ... '.format(trg))
    trg_vocab = extract_vocab(trg, wargs.trg_vcb, wargs.n_trg_vcb_plan, wargs.max_seq_len)
    n_src_vcb, n_trg_vcb = src_vocab.size(), trg_vocab.size()
    wlog('Vocabulary size: |source|={}, |target|={}'.format(n_src_vcb, n_trg_vcb))
    vocabs['src'], vocabs['trg'] = src_vocab, trg_vocab

    wlog('\nPreparing training set from {} and {} ... '.format(src, trg))
    trains = {}
    train_src_tlst, train_trg_tlst = wrap_data(wargs.dir_data, wargs.train_prefix,
                                               wargs.train_src_suffix, wargs.train_trg_suffix,
                                               src_vocab, trg_vocab, shuffle=True,
                                               sort_k_batches=wargs.sort_k_batches,
                                               max_seq_len=wargs.max_seq_len,
                                               char=wargs.src_char)
    '''
    list [torch.LongTensor (sentence), torch.LongTensor, torch.LongTensor, ...]
    no padding
    '''
    batch_train = Input(train_src_tlst, train_trg_tlst, wargs.batch_size,
                        batch_type=wargs.batch_type, bow=wargs.trg_bow, batch_sort=False,
                        gpu_ids=device_ids)
    wlog('Sentence-pairs count in training data: {}'.format(len(train_src_tlst)))

    batch_valid = None
    if wargs.val_prefix is not None:
        val_src_file = os.path.join(wargs.val_tst_dir, '{}.{}'.format(wargs.val_prefix, wargs.val_src_suffix))
        val_trg_file = os.path.join(wargs.val_tst_dir, '{}.{}'.format(wargs.val_prefix, wargs.val_ref_suffix))
        val_src_file, val_trg_file = os.path.abspath(val_src_file), os.path.abspath(val_trg_file)
        wlog('\nPreparing validation set from {} and {} ... '.format(val_src_file, val_trg_file))
        valid_src_tlst, valid_trg_tlst = wrap_data(wargs.val_tst_dir, wargs.val_prefix,
                                                   wargs.val_src_suffix, wargs.val_ref_suffix,
                                                   src_vocab, trg_vocab, shuffle=False,
                                                   max_seq_len=wargs.dev_max_seq_len,
                                                   char=wargs.src_char)
        batch_valid = Input(valid_src_tlst, valid_trg_tlst,
                            batch_size=wargs.valid_batch_size, batch_sort=False,
                            gpu_ids=device_ids)

    batch_tests = None
    if wargs.tests_prefix is not None:
        assert isinstance(wargs.tests_prefix, list), 'Test files should be list.'
        init_dir(wargs.dir_tests)
        batch_tests = {}
        for prefix in wargs.tests_prefix:
            init_dir(wargs.dir_tests + '/' + prefix)
            test_file = '{}{}.{}'.format(wargs.val_tst_dir, prefix, wargs.val_src_suffix)
            test_file = os.path.abspath(test_file)
            wlog('\nPreparing test set from {} ... '.format(test_file))
            test_src_tlst, _ = wrap_tst_data(test_file, src_vocab, char=wargs.src_char)
            batch_tests[prefix] = Input(test_src_tlst, None,
                                        batch_size=wargs.test_batch_size,
                                        batch_sort=False, gpu_ids=device_ids)
    wlog('\n## Finish to Prepare Dataset ! ##\n')

    src_emb = WordEmbedding(n_src_vcb, wargs.d_src_emb, wargs.input_dropout,
                            wargs.position_encoding, prefix='Src')
    trg_emb = WordEmbedding(n_trg_vcb, wargs.d_trg_emb, wargs.input_dropout,
                            wargs.position_encoding, prefix='Trg')
    # share the embedding matrix - preprocess with share_vocab required.
    if wargs.embs_share_weight:
        if n_src_vcb != n_trg_vcb:
            raise AssertionError('The `-share_vocab` should be set during '
                                 'preprocess if you use share_embeddings!')
        src_emb.we.weight = trg_emb.we.weight

    nmtModel = build_NMT(src_emb, trg_emb)

    if device_ids is not None:
        wlog('push model onto GPU {} ... '.format(device_ids[0]), 0)
        nmtModel_par = nn.DataParallel(nmtModel, device_ids=device_ids)
        nmtModel_par.to(device)
    else:
        wlog('push model onto CPU ... ', 0)
        nmtModel.to(tc.device('cpu'))
    wlog('done.')

    if wargs.pre_train is not None:
        assert os.path.exists(wargs.pre_train)
        from tools.utils import load_model
        _dict = load_model(wargs.pre_train)
        # initializing parameters of interactive attention model
        class_dict = None
        if len(_dict) == 5:
            #model_dict, e_idx, e_bidx, n_steps, optim = _dict['model'], _dict['epoch'], _dict['batch'], _dict['steps'], _dict['optim']
            model_dict, e_idx, e_bidx, n_steps, optim = _dict
        elif len(_dict) == 4:
            model_dict, e_idx, e_bidx, optim = _dict
        for name, param in nmtModel.named_parameters():
            if name in model_dict:
                param.requires_grad = not wargs.fix_pre_params
                param.data.copy_(model_dict[name])
                wlog('{:7} -> grad {}\t{}'.format('Model', param.requires_grad, name))
            elif name.endswith('map_vocab.weight'):
                if class_dict is not None:
                    param.requires_grad = not wargs.fix_pre_params
                    param.data.copy_(class_dict['map_vocab.weight'])
                    wlog('{:7} -> grad {}\t{}'.format('Model', param.requires_grad, name))
            elif name.endswith('map_vocab.bias'):
                if class_dict is not None:
                    param.requires_grad = not wargs.fix_pre_params
                    param.data.copy_(class_dict['map_vocab.bias'])
                    wlog('{:7} -> grad {}\t{}'.format('Model', param.requires_grad, name))
            else: init_params(param, name, init_D=wargs.param_init_D, a=float(wargs.u_gain))

        #wargs.start_epoch = e_idx + 1
        optim.n_current_steps = 0

    else:
        optim = Optim(wargs.opt_mode, wargs.learning_rate, wargs.max_grad_norm)
        for n, p in nmtModel.named_parameters():
            # bias can not be initialized uniformly
            if 'norm' in n:
                wlog('ignore layer norm init ...')
                continue
            if 'emb' in n:
                wlog('ignore word embedding weight init ...')
                continue
            if 'vcb_proj' in n:
                wlog('ignore vcb_proj weight init ...')
                continue
            init_params(p, n, init_D=wargs.param_init_D, a=float(wargs.u_gain))
            #if wargs.encoder_type != 'att' and wargs.decoder_type != 'att':
            #    init_params(p, n, init_D=wargs.param_init_D, a=float(wargs.u_gain))

    wlog(nmtModel)
    wlog(optim)
    pcnt1 = len([p for p in nmtModel.parameters()])
    pcnt2 = sum([p.nelement() for p in nmtModel.parameters()])
    wlog('parameters number: {}/{}'.format(pcnt1, pcnt2))

    wlog('\n' + '*' * 30 + ' trainable parameters ' + '*' * 30)
    for n, p in nmtModel.named_parameters():
        if p.requires_grad: wlog('{:60} : {}'.format(n, p.size()))

    optim.init_optimizer(nmtModel.parameters())

    criterion = LabelSmoothingCriterion(trg_emb.n_vocab, label_smoothing=wargs.label_smoothing)
    if device_ids is not None:
        wlog('push criterion onto GPU {} ... '.format(device_ids[0]), 0)
        criterion = criterion.to(device)
        wlog('done.')                                                                                                         
    
    BOWcriterion = BOWLossCriterion(trg_emb.n_vocab) if wargs.bow_loss is True else None
    if wargs.bow_loss is True and device_ids is not None:
        wlog('push bow criterion onto GPU {} ... '.format(device_ids[0]), 0)
        BOWcriterion = BOWcriterion.to(device)
        wlog('done.')
    EMBcriterion = EMBLossCriterion(trg_emb) if wargs.emb_loss is True else None
    if wargs.emb_loss is True and device_ids is not None:
        wlog('push emb criterion onto GPU {} ... '.format(device_ids[0]), 0)
        EMBcriterion = EMBcriterion.to(device)
        wlog('done.')

    lossCompute = MultiGPULossCompute(nmtModel.generator, criterion,
            wargs.d_model if wargs.decoder_type == 'att' else 2 * wargs.d_enc_hid,
            n_trg_vcb, trg_emb, nmtModel.bowMapper, loss_norm=wargs.loss_norm,
            bow_crit=BOWcriterion, emb_crit=EMBcriterion, chunk_size=wargs.chunk_size, device_ids=device_ids)

    trainer = Trainer(nmtModel_par, batch_train, vocabs, optim, lossCompute, nmtModel, batch_valid, batch_tests)

    trainer.train()


if __name__ == '__main__':

    main()















