from __future__ import division

import sys
import copy
import time
import numpy as np
import torch as tc
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=5)

import wargs
from tools.utils import BOS, EOS, debug, init_beam, part_sort, lp_cp

class Nbs(object):

    def __init__(self, model, tvcb_i2w, k=10, ptv=None, noise=False, print_att=False, gpu_ids=None):

        self.k = k
        self.ptv = ptv
        self.noise = noise
        self.tvcb_i2w = tvcb_i2w
        self.print_att = print_att
        self.C = [0] * 4
        #self.encoder, self.decoder, self.generator = model.encoder, model.decoder, model.generator
        self.model = model
        '''
        prob_projection = nn.LogSoftmax()
        self.model.prob_projection = prob_projection.cuda()
        #self.print_att = False
        '''
        self.gpu_ids = gpu_ids

    def beam_search_trans(self, x_BL, x_mask=None):

        self.beam, self.hyps = [], []
        if isinstance(x_BL, list):
            x_BL = tc.tensor(x_BL).long().unsqueeze(0)
            x_BL = x_BL.to(tc.device('cuda', self.gpu_ids[0]))
        elif isinstance(x_BL, tuple):
            # x_BL: (idxs, tsrcs, tspos, lengths, src_mask)
            if len(x_BL) == 4: _, x_BL, lens, src_mask = x_BL
            elif len(x_BL) == 2: x_BL, src_pos_BL = x_BL
            elif len(x_BL) == 8:
                _, x_BL, src_pos, _, _, _, _, _ = x_BL
                x_BL, src_pos_BL = x_BL.t(), src_pos.t()
        #assert x_BL.size(0) == 1, 'Unsupported for batch decoding ... '
        #if wargs.gpu_id is not None and not x_BL.is_cuda: x_BL = x_BL.cuda()
        #self.maxL = wargs.max_seq_len
        self.B, self.x_len = x_BL.size(0), x_BL.size(1)
        self.maxL, self.x_BL = 2 * self.x_len, x_BL 
        if x_mask is None:
            x_mask = tc.ones((1, self.x_len), requires_grad=False)
            if self.gpu_ids != None: self.x_mask = x_mask.to(tc.device('cuda', self.gpu_ids[0]))
        else: self.x_mask = x_mask

        debug('x_BL: {}\n{}'.format(x_BL.size(), x_BL))
        debug('x_mask: {}\n{}'.format(self.x_mask.size(), self.x_mask))
        #self.enc_src0, _ = self.encoder(x_BL, self.x_mask)
        self.enc_src0, _ = self.model.encode(x_BL, self.x_mask)
        debug('enc_src0: {}\n{}'.format(self.enc_src0.size(), self.enc_src0))
        init_beam(self.beam, B=self.B, gpu_id=self.gpu_ids[0], cnt=self.maxL, cp=True)
        self.B_tran_cands = [ [] for _ in range(self.B) ]
        self.B_aprob_cands = [[] for _ in range(self.B)] if self.print_att is True else None

        if not wargs.with_batch: best_trans, best_loss = self.search()
        elif wargs.ori_search:   best_trans, best_loss = self.ori_batch_search()
        else:                    self.batch_search()
        # best_trans w/o <bos> and <eos> !!!
        #tc.cuda.empty_cache()

        '''
            [
                batch_0 : [ (cand_0: ([2,...,3], score)), (cand_1: ([2,...,3], score)), ...,  ]
                batch_1 : [ (cand_0), (cand_1), ...,  ]
                batch_2 : [ (cand_0), (cand_1), ...,  ]
                ......
                batch_39: [ (cand_0), (cand_1), ...,  ]
            ]
        '''
        # B_tran_cands: [(trans, loss, attend)]
        #for bidx in range(self.B):
        #    debug('===========================================================')
        #    debug([ (a[0], a[1]) for a in self.B_tran_cands[bidx] ])
        #    best_trans, best_loss, best_norm_loss, _ = self.B_tran_cands[bidx][0]
        #    debug('Src[{}], maskL[{}], hyp (w/o EOS)[{}], maxL[{}], loss[{}]'.format(
        #        x_mask[bidx].sum().item(), self.x_len, len(best_trans), self.maxL, best_loss))
        #debug('Average Merging Rate [{}/{}={:6.4f}]'.format(self.C[1], self.C[0], self.C[1] / self.C[0]))
        #debug('Average location of bp [{}/{}={:6.4f}]'.format(self.C[3], self.C[2], self.C[3] / self.C[2]))
        #debug('Step[{}] stepout[{}]'.format(*self.C[4:]))

        return self.B_tran_cands

    #@exeTime
    def batch_search(self):

        # s0: (1, trg_nhids), self.enc_src0: (B, x_len, src_nhids*2)
        enc_size, L = self.enc_src0.size(-1), self.x_len
        debug('Last layer output of encoder: {}'.format(self.enc_src0.size()))
        for i in range(1, self.maxL + 1):
            B_prevbs = self.beam[i - 1]
            #debug('\n{} Look Beam-{} {}'.format('-'*20, i - 1, '-'*20))
            #for bidx, sent_prevb in enumerate(B_prevbs):
            #    debug('Sent {} '.format(bidx))
            #    for b in sent_prevb:    # do not output state
            #        debug(b[0:1] + (None if b[1] is None else b[1].size(),
            #                        None if b[-4] is None else b[-4].size()) + b[-3:])
            #debug('\n{} Step-{} {}'.format('#'*20, i, '#'*20))

            n_remainings = len(B_prevbs)
            if all(len(cands) == self.k for cands in self.B_tran_cands) is True or n_remainings == 0:
                #debug('Early stop~ Normal beam search or sampling in this batch finished.')
                return
            xs_mask, enc_src, prebs_sz, self.true_bidx, y_part_seqs, hyp_scores = [], [], [], [], [], []
            #debug('n_remainings: {}'.format(n_remainings))
            for bidx in range(n_remainings):
                prevb = B_prevbs[bidx]
                preb_sz = len(prevb)
                prebs_sz.append(preb_sz)
                hyp_scores += list(zip(*prevb))[0]
                self.true_bidx.append(prevb[0][-3])
                # (src_sent_len, B, src_nhids) -> (src_sent_len, B*preb_sz, src_nhids)
                enc_src.append(self.enc_src0[bidx].repeat(preb_sz, 1, 1))
                xs_mask.append(self.x_mask[bidx].repeat(preb_sz, 1))
                #debug(self.enc_src0[bidx].repeat(preb_sz, 1, 1).size())
                #debug(tc.stack(list(zip(*prevb))[-2]))
                y_part_seqs.append(tc.stack(list(zip(*prevb))[-2]))
            enc_src, y_part_seqs = tc.cat(enc_src, dim=0), tc.cat(y_part_seqs, dim=0)
            xs_mask = tc.cat(xs_mask, dim=0)
            #debug('enc_src: {}'.format(enc_src.size()))
            #debug('xs_mask: {}'.format(xs_mask.size()))

            cnt_bp = (i >= 2)
            #debug('hyp_scores: \n{}'.format(hyp_scores))
            hyp_scores = tc.stack(hyp_scores)
            #debug('hyp_scores: {} \n{}'.format(hyp_scores.size(), hyp_scores))
            # -- Preparing decoded data seq -- #
            #y_part_seqs = track_ys(i) # (preb_sz, trg_part_L)
            #y_part_seqs = tc.tensor(y_part_seqs, requires_grad=False).view(-1, i)
            #if self.gpu_ids is not None: y_part_seqs = y_part_seqs.cuda()
            # self.enc_src0: (B, len_q, d_model)
            # (B, x_len, enc_size) -> (B, preb_sz, x_len, enc_size) -> (B*preb_sz, x_len, enc_size)
            enc_srcs = enc_src.reshape(-1, L, enc_size)
            #debug('Whole x reprs -> {}'.format(enc_srcs.size()))
            #y_part_seqs = y_part_seqs.cuda(self.gpu_ids[0]).type(tc.int64)
            # -- Decoding -- #
            #debug('Part y seq: {}'.format(y_part_seqs.size()))
            #dec_output, _, nlayer_attns = self.decoder(y_part_seqs, x_BL, enc_srcs)
            #dec_output, _, alpha_ij = self.decoder(y_part_seqs, x_BL, enc_srcs)
            # y_part_seqs: (B*preb_sz, part_L)
            #dec_output, _, alpha_ij = self.decoder(y_part_seqs, enc_srcs, xs_mask)
            dec_output, _, alpha_ij = self.model.decode(y_part_seqs, enc_srcs, xs_mask)
            #alpha_ij = nlayer_attns[-1]
            #debug('History decoder output: {}'.format(dec_output.size()))
            #debug('Previous decoder output alpha_ij: {}'.format(alpha_ij.size()))
            # (preb_sz, part_Len, d_model) -> (preb_sz, d_model)
            dec_output = dec_output[:, -1, :] # (preb_sz, d_model) previous decoder hidden state
            #debug('Previous decoder output: {}'.format(dec_output.size()))
            # alpha_ij: (B, n_heads, trgL, x_len) -> (x_len, B)
            alpha_ij = alpha_ij[:, -1, -1, :].permute(1, 0)
            if self.B_aprob_cands is not None: self.B_aprob_cands.append(alpha_ij)
            self.C[2] += 1
            self.C[3] += 1
            #next_ces = -self.generator(dec_output)[-1]  # negative log-likelihood
            next_ces = -self.model.generate(dec_output)[-1]  # negative log-likelihood
            voc_size = next_ces.size(-1)
            #debug('next_ces {}'.format(next_ces.size()))
            cand_scores = hyp_scores[:, None] + next_ces
            #debug('cand_scores {}'.format(cand_scores.size()))
            next_ces_B_prevb = tc.split(cand_scores, prebs_sz, dim=0) # [B: (prevb, vocab)]
            #debug(next_ces_B_prevb)
            #debug(self.B_tran_cands)
            next_step_beam, del_batch_idx = [], []
            #debug('true_bidx: {}, n_remainings: {}'.format(self.true_bidx, n_remainings))
            for sent_id in range(n_remainings):
                next_beam_cur_sent, true_id = [], self.true_bidx[sent_id]
                if len(self.B_tran_cands[true_id]) == self.k: continue
                cand_scores = next_ces_B_prevb[sent_id]
                #debug('cand_scores {}'.format(cand_scores.size()))
                #if i < self.x_len - 1:
                #    '''here we make the score of <s> so large to avoid null translation'''
                #    cand_scores[:, EOS] = float('+inf')
                cand_scores_flat = cand_scores.flatten()
                #debug('cand_scores_flat {}'.format(cand_scores_flat.size()))
                #ranks_flat = part_sort(cand_scores_flat, self.k - len(self.hyps))
                #ranks_flat = cand_scores_flat.topk(k=self.k - len(self.hyps), dim=-1, largest=False, sorted=True)
                if wargs.len_norm == 0: norm_scores = cand_scores_flat
                elif wargs.len_norm == 1: norm_scores = cand_scores_flat / i
                elif wargs.len_norm == 2:   # alpha length normal
                    #lp, cp = lp_cp(bp, i, 0, self.beam)
                    lp, cp = lp_cp(0, i, 0, self.beam)
                    norm_scores = cand_scores_flat / lp
                norm_costs, ranks_flat = norm_scores.topk(k=self.k, dim=-1, largest=False, sorted=True)
                #debug(norm_costs)
                #debug(ranks_flat)
                #costs = cand_scores.gather(dim=1, index=ranks_flat)
                costs = cand_scores_flat[ranks_flat]
                #debug('costs \n{}'.format(costs))
                #debug('ranks_flat \n{}'.format(ranks_flat))
                #debug('voc_size {}'.format(voc_size))
                word_indices = ranks_flat % voc_size
                #debug('word_indices \n{}'.format(word_indices))
                prevb_ids = ranks_flat // voc_size
                #debug('For beam[{}], pre-beam ids: \n{}'.format(i, prevb_ids))

                for prevb_id, b in enumerate(zip(costs, norm_costs, word_indices, prevb_ids)):
                    #debug('Sent {}, Cand {}, True {}:'.format(sent_id, prevb_id, true_id))
                    cost, norm_cost, bp = b[0], b[1], b[-1].item()
                    if cnt_bp: self.C[1] += (b[-1] + 1)
                    part_sent = tc.cat([B_prevbs[sent_id][bp][-2], b[-2].unsqueeze(-1)])
                    if b[-2].item() == EOS:
                        self.B_tran_cands[true_id].append((part_sent.tolist(), cost.item(), norm_cost.item(), None))
                        #debug('OK cand: {}'.format(self.B_tran_cands[true_id][-1]))
                        if len(self.B_tran_cands[true_id]) == self.k:
                            # output sentence, early stop, best one in k
                            #debug('Early stop! see {} cands ending with EOS.'.format(self.k))
                            del_batch_idx.append(sent_id)
                            self.B_tran_cands[true_id] = sorted(self.B_tran_cands[true_id], key=lambda tup: tup[-2])
                            #for cand in sorted_cands: debug('{}'.format(cand))
                            #best_cand = sorted_cands[0]
                            #debug('Best cand length (w/ EOS)[{}]'.format(best_cand[0])) # w/ eos w/o bos
                            break
                    else:
                        # should calculate when generate item in current beam
                        next_beam_cur_sent.append((cost, None, None, true_id) + (part_sent, ) + (bp, ))
                if len(next_beam_cur_sent) > 0 and len(self.B_tran_cands[true_id]) < self.k: next_step_beam.append(next_beam_cur_sent)

            self.beam[i] = next_step_beam
            if len(del_batch_idx) < n_remainings:
                self.enc_src0 = self.enc_src0[list(filter(lambda x: x not in del_batch_idx, range(n_remainings)))]
                self.x_mask = self.x_mask[list(filter(lambda x: x not in del_batch_idx, range(n_remainings)))]
            del enc_srcs     # free the tensor
            #tc.cuda.empty_cache()

        # no early stop, back tracking
        #return back_tracking(self.beam, 0, self.no_early_best(), self.attent_probs)
        if n_remainings == 0: return
        self.no_early_best()

    def no_early_best(self):

        debug('==Start== No early stop ...')
        for prev_sent_beam in self.beam[self.maxL]:
            if len(prev_sent_beam) == 0: continue
            cost, _, _, true_id, part_sent, bp = sorted(prev_sent_beam, key=lambda tup: tup[0].item())[0]
            hyps = self.B_tran_cands[true_id]
            if len(hyps) == self.k: continue  # have finished this sentence
            elif len(hyps) == 0:
                debug('No early stop, no hyp with EOS, select k hyps length {} '.format(self.maxL))
                hyps.append((part_sent.tolist() + [EOS], cost.item(), cost.item(), None))
            self.B_tran_cands[true_id] = sorted(hyps, key=lambda tup: tup[-2])
        debug('==End== No early stop ...')

