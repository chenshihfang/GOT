# --------------------------------------------------------------------*/
# This file includes code from https://github.com/facebookresearch/detr/blob/main/models/detr.py
# --------------------------------------------------------------------*/
#

import copy
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, List
from torch import Tensor

# --- add this import to use PE's RoPE entrypoint ---
from ltr.models.transformer.position_encoding import PositionEmbeddingSine
import math

####################################################


# Transformer_ori
class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048,
                 dropout=0.1, activation="relu", normalize_before=False, return_intermediate_dec=False,
                 use_ckpt=False,
                 ckpt_impl="torch"
                 ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None

        # self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm,
                                          use_ckpt=use_ckpt, ckpt_impl=ckpt_impl)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        query_embed = query_embed.unsqueeze(1).repeat(1, src.shape[1], 1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)


        # print("src.shape", src.shape) # [H*W*ref_num+H*W*1, B, 256]
        # # print("mask.shape", mask.shape) # None
        # print("pos_embed.shape", pos_embed.shape) # [972 2 256]
        # print("query_embed.shape", query_embed.shape) # [1 2 256]

        # print("memory.shape", memory.shape) # [H*W*ref_num+H*W*1, B, 256]
        # print("tgt.shape", tgt.shape) # [1 B 256]
        # print("hs.shape", hs.shape) # [1 1 B 256]
        # print("hs.transpose(1, 2).shape", hs.transpose(1, 2).shape) # [1 B 1 256]
        # input()

        return hs.transpose(1, 2), memory

####################################################


# TransformerDecoder_ori
class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, pos=None, query_pos=None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, pos=pos, query_pos=query_pos,
                           tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)

            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


####################################################


# TransformerDecoderLayer_ori
class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                     memory_key_padding_mask=None, pos=None, query_pos=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos), key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                    memory_key_padding_mask=None, pos=None, query_pos=None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos), key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, pos=None, query_pos=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


####################################################


class TransformerDecoderInstance(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_decoder_layers=6, dim_feedforward=2048,
                 dropout=0.1, activation="relu", normalize_before=False, return_intermediate_dec=False):
        super().__init__()

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        if query_embed.dim() == 2:
            query_embed = query_embed.unsqueeze(1).repeat(1, src.shape[1], 1)

        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(tgt, src, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2)


class TransformerEncoderInstance(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048,
                 dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed):
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        return memory


# TransformerEncoder_ckpt_impl
class TransformerEncoder(nn.Module):
    """
    Transformer Encoder with optional activation checkpointing and optional BN freezing.

    Args:
        encoder_layer (nn.Module): a single encoder layer to be cloned N times
        num_layers (int): number of layers
        norm (nn.Module or None): final normalization module
        use_ckpt (bool): whether to use activation checkpointing
        ckpt_impl (str): "torch" or "deepspeed" (fallback to torch if deepspeed unavailable)
        freeze_bn (bool): if True, temporarily set BN layers inside each encoder layer to eval()
                          during the layer call (both the original forward and any recompute).
                          If False, BN behaves normally.
    """
    def __init__(
        self,
        encoder_layer,
        num_layers,
        norm=None,
        use_ckpt=False,
        ckpt_impl="torch",
        # freeze_bn: bool = True,
        freeze_bn: bool = False,
    ):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.use_ckpt = bool(use_ckpt)
        self.freeze_bn = bool(freeze_bn)

        print("TransformerEncoder self.freeze_bn", self.freeze_bn)

        impl = str(ckpt_impl).lower().strip()
        self._ckpt_fn = None

        if impl == "deepspeed":
            # Try DeepSpeed wrapper explicitly
            try:
                from deepspeed.runtime.activation_checkpointing import checkpointing as ds_ckpt
                self._ckpt_fn = getattr(ds_ckpt, "checkpoint", None)
            except Exception:
                self._ckpt_fn = None
            if self._ckpt_fn is None:
                try:
                    import deepspeed
                    self._ckpt_fn = getattr(getattr(deepspeed, "checkpointing", None), "checkpoint", None)
                except Exception:
                    self._ckpt_fn = None

        # Fallback to torch or explicit torch selection
        if self._ckpt_fn is None:
            from torch.utils.checkpoint import checkpoint as torch_ckpt
            self._ckpt_fn = torch_ckpt

    # ---------- BN freezing helpers ----------
    @staticmethod
    def _iter_bn_modules(root: nn.Module):
        """Yield all BN modules under root."""
        bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
        for m in root.modules():
            if isinstance(m, bn_types):
                yield m

    from contextlib import contextmanager
    @contextmanager
    def _temp_eval_bn(self, layer: nn.Module):
        """
        Temporarily set all BN under layer to eval(), then restore training flags.
        No-op when self.freeze_bn is False or when there is no BN.
        """
        if not self.freeze_bn:
            yield
            return

        bns = list(self._iter_bn_modules(layer))
        if not bns:
            yield
            return

        # Save original training flags and switch to eval()
        orig = []
        for bn in bns:
            orig.append((bn, bn.training))
            bn.eval()
        try:
            yield
        finally:
            for bn, was_training in orig:
                bn.train(was_training)

    # ---------- Checkpoint wrapper ----------
    def _ckpt_layer(self, layer, x, mask, key_pad_mask, pos):
        """
        Wrap a single encoder layer with checkpointing, freezing BN during the call if enabled.
        """
        def _fw(_x, _mask, _kpm, _pos):
            # Executed on the initial forward and on recompute
            with self._temp_eval_bn(layer):
                return layer(_x, src_mask=_mask, src_key_padding_mask=_kpm, pos=_pos)

        return self._ckpt_fn(_fw, x, mask, key_pad_mask, pos)

    # ---------- Forward ----------
    def forward(self, src, mask=None, src_key_padding_mask=None, pos=None):
        """
        src: shape (S, N, E) for standard PyTorch Transformer layouts
        mask: optional attention mask
        src_key_padding_mask: optional key padding mask
        pos: optional positional encoding or embeddings to add or pass through
        """
        output = src
        for layer in self.layers:
            use_ckpt_now = self.use_ckpt and torch.is_grad_enabled() and (
                output.requires_grad or any(p.requires_grad for p in layer.parameters())
            )

            if use_ckpt_now:
                output = self._ckpt_layer(layer, output, mask, src_key_padding_mask, pos)
            else:
                with self._temp_eval_bn(layer):
                    output = layer(
                        output,
                        src_mask=mask,
                        src_key_padding_mask=src_key_padding_mask,
                        pos=pos,
                    )

        if self.norm is not None:
            output = self.norm(output)
        return output


# TransformerEncoder_ori
class TransformerEncoder_ori(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, pos=None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output



# TransformerEncoderLayer_ori
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        print("TransformerEncoderLayer_ori")
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu/glu, not {activation}.")


