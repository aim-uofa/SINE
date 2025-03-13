
import torch
from torch import Tensor, nn
from torch.nn import functional as F
import numpy as np
from typing import Tuple, Optional

from detectron2.layers import get_norm

from .position_encoding import PositionEmbeddingSine

# Ref: https://github.com/NVlabs/ODISE/blob/e97b06c424c575fec9fc5368dd4b3e050d91abc4/odise/modeling/meta_arch/odise.py#L923
class MaskPooling(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x, mask):
        """
        Args:
            x: [B, C, H, W]
            mask: [B, Q, H, W]
        """
        if not x.shape[-2:] == mask.shape[-2:]:
            # reshape mask to x
            mask = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)
        with torch.no_grad():
            mask = mask.detach()
            mask = (mask > 0).to(mask.dtype)
            denorm = mask.sum(dim=(-1, -2), keepdim=True) + 1e-8

        mask_pooled_x = torch.einsum(
            "bchw,bqhw->bqc",
            x,
            mask / denorm,
        )
        return mask_pooled_x


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        pre_norm: bool = False,
        sa_on: bool = True,
        ca_on: bool = True,
        ffn_on: bool = True
    ):
        super().__init__()

        self.transformer_self_attention_layer = SelfAttentionLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dropout=0.0,
            normalize_before=pre_norm,
        ) if sa_on else None

        self.transformer_cross_attention_layer = CrossAttentionLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dropout=0.0,
            normalize_before=pre_norm,
        ) if ca_on else None

        self.transformer_ffn_layer = FFNLayer(
            d_model=embedding_dim,
            dim_feedforward=mlp_dim,
            dropout=0.0,
            normalize_before=pre_norm,
        ) if ffn_on else None

    def forward(self, output, image_feat, query_embed=None, image_pe=None, attn_mask=None):

        if self.transformer_self_attention_layer is not None:
            output = self.transformer_self_attention_layer(
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

        if self.transformer_cross_attention_layer is not None:
            output = self.transformer_cross_attention_layer(
                output, image_feat,
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=image_pe, query_pos=query_embed
            )

        # FFN
        if self.transformer_ffn_layer is not None:
            output = self.transformer_ffn_layer(output)

        return output

class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        pre_norm: bool = True,
    ):
        super().__init__()

        self.transformer_self_attention_layer = SelfAttentionLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dropout=0.0,
            normalize_before=pre_norm,
        )

        self.transformer_cross_attention_layer = CrossAttentionLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dropout=0.0,
            normalize_before=pre_norm,
        )

        self.transformer_ffn_layer1 = FFNLayer(
            d_model=embedding_dim,
            dim_feedforward=mlp_dim,
            dropout=0.0,
            normalize_before=pre_norm,
        )

        self.transformer_ffn_layer2 = FFNLayer(
            d_model=embedding_dim,
            dim_feedforward=mlp_dim,
            dropout=0.0,
            normalize_before=pre_norm,
        )

    def forward(self, output, image_feat, query_embed=None, image_pe=None, sa_attn_mask=None, ca_attn_mask=None):

        [output_id, output_ins, output_sem] = output
        [query_embed_id, query_embed_ins, query_embed_sem] = query_embed

        nm, nq, nc = query_embed_id.shape[0], query_embed_ins.shape[0], query_embed_sem.shape[0]

        output = torch.cat([output_id, output_ins, output_sem], dim=0)
        query_embed = torch.cat([query_embed_id, query_embed_ins, query_embed_sem], dim=0)

        # sa
        output = self.transformer_self_attention_layer(
            output, tgt_mask=sa_attn_mask,
            tgt_key_padding_mask=None,
            query_pos=query_embed
        )

        output_id, output_ins, output_sem = torch.split(output, split_size_or_sections=(nm, nq, nc), dim=0)
        output = torch.cat([output_id, output_ins], dim=0)
        query_embed = torch.cat([query_embed_id, query_embed_ins], dim=0)

        # ca
        output = self.transformer_cross_attention_layer(
            output, image_feat,
            memory_mask=ca_attn_mask,
            memory_key_padding_mask=None,  # here we do not apply masking on padded region
            pos=image_pe, query_pos=query_embed
        )

        output_id, output_ins = torch.split(output, split_size_or_sections=(nm, nq), dim=0)

        # ffn
        output_id = self.transformer_ffn_layer1(output_id)
        output_ins = self.transformer_ffn_layer1(output_ins)
        output_sem = self.transformer_ffn_layer2(output_sem)

        return [output_id, output_ins, output_sem]


def get_classification_logits(x, text_classifier, logit_scale):
    # x in shape of [B, *, C]
    # text_classifier in shape of [num_classes, C]
    # logit_scale is a learnable scalar https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/model.py#L201
    # return: [B, *, num_classes]
    x = F.normalize(x, dim=-1)
    text_classifier = F.normalize(text_classifier, dim=-1)
    logit_scale = torch.clamp(logit_scale.exp(), max=100)
    pred_logits = torch.einsum("bqc,bnc->bqn", x, text_classifier)  # B, *, N + 1
    pred_logits = logit_scale * pred_logits # B, *, N + 1

    return pred_logits


class MFormer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        mask_dim: int,
        fusion_layer_depth: int = 1,
        temp: float = 30.,
        num_queries: int = 20,
        pre_norm: bool = True,
        mask_classification = True
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = embedding_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        self.num_heads = num_heads
        self.num_layers = depth
        self.embedding_dim = embedding_dim
        self.mlp_dim = mlp_dim
        self.temp = nn.Parameter(torch.tensor(temp))

        self.num_fusion_layers = fusion_layer_depth
        self.fusion_layers  = nn.ModuleList()
        for _ in range(self.num_fusion_layers):
            self.fusion_layers.append(
                TransformerEncoderLayer(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                )
            )

        self.transformer_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.transformer_layers.append(
                TransformerDecoderLayer(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    pre_norm=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(embedding_dim)
        self.pre_norm = pre_norm

        self.mask_feat_layer = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, embedding_dim // 2, kernel_size=2, stride=2),
            get_norm("LN", embedding_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(embedding_dim // 2, embedding_dim, kernel_size=2, stride=2),
        )

        self.mask_pooling = MaskPooling()

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, embedding_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, embedding_dim)

        self.query_grained_embed = nn.Embedding(3, embedding_dim)

        self.decoder_norm = nn.LayerNorm(embedding_dim)

        # output FFNs
        if self.mask_classification:
            self.id_class_embed = nn.Linear(embedding_dim, 2)
            self.neg_class_embed = nn.Embedding(1, embedding_dim)

        self.mask_embed = MLP(embedding_dim, embedding_dim, mask_dim, 3)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):

        id_output, ins_output, sem_output = [self.decoder_norm(out).transpose(0, 1) for out in output]

        outputs_class_id = self.id_class_embed(id_output) # bs, nm, 2
        outputs_class_ins = get_classification_logits(ins_output, sem_output, self.logit_scale)
        outputs_class = [outputs_class_id, outputs_class_ins]

        id_mask_embed = self.mask_embed(id_output)
        ins_mask_embed = self.mask_embed(ins_output)
        mask_embed = [id_mask_embed, ins_mask_embed]

        id_outputs_mask = torch.einsum("bqc,bchw->bqhw", id_mask_embed, mask_features) # bs, nm, h, w
        ins_outputs_mask = torch.einsum("bqc,bchw->bqhw", ins_mask_embed, mask_features) # bs, nq, h ,w
        nm, nq = id_outputs_mask.shape[1], ins_outputs_mask.shape[1]
        outputs_mask = torch.cat([id_outputs_mask, ins_outputs_mask], dim=1) # bs, nm+nq, h, w

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B, h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        outputs_mask = torch.split(outputs_mask, split_size_or_sections=(nm, nq), dim=1)

        return outputs_class, outputs_mask, attn_mask, mask_embed

    def apply_gaussian_kernel(self, corr, spatial_height, spatial_width, sigma=10):
        bsz, nm, _ = corr.size()

        # Get max correlation index for each query
        center = corr.max(dim=2)[1]
        
        # Compute y, x coordinates from the flattened index
        center_y = center // spatial_width  # Row index (height)
        center_x = center % spatial_width  # Column index (width)

        # Create coordinate grid
        x = torch.arange(0, spatial_width).float().to(corr.device)
        y = torch.arange(0, spatial_height).float().to(corr.device)

        # Compute y-distance and x-distance from the center
        y = y.view(1, 1, spatial_height).repeat(bsz, nm, 1) - center_y.unsqueeze(2)
        x = x.view(1, 1, spatial_width).repeat(bsz, nm, 1) - center_x.unsqueeze(2)

        # Expand dimensions for proper broadcasting
        y = y.unsqueeze(3).repeat(1, 1, 1, spatial_width)  # (B, nm, H, W)
        x = x.unsqueeze(2).repeat(1, 1, spatial_height, 1)  # (B, nm, H, W)

        # Compute Gaussian kernel with different H and W
        gauss_kernel = torch.exp(-(x.pow(2) + y.pow(2)) / (2 * sigma ** 2))

        # Reshape correlation map to match (B, nm, H, W)
        filtered_corr = gauss_kernel * corr.view(bsz, nm, spatial_height, spatial_width)

        # Restore shape back to (B, nm, H*W)
        filtered_corr = filtered_corr.view(bsz, nm, spatial_height * spatial_width)

        return filtered_corr

    def forward_per_image(
        self,
        ref_feature: Tensor,
        tar_feature: Tensor,
        ref_mask: Tensor,
        ref_label: Tensor
    ):

        bs, c, h, w = tar_feature.shape
        tar_mask_feature = self.mask_feat_layer(tar_feature)
        _, _, mh, mw = tar_mask_feature.shape

        nq = self.num_queries
        nm = ref_mask.shape[0]
        mask_for_pooling = F.interpolate(ref_mask.to(ref_feature.dtype), size=tar_mask_feature.shape[-2:],
                                         mode='bilinear', align_corners=False).to(ref_feature.dtype)
        ref_feat_for_pooling = F.interpolate(ref_feature, size=mask_for_pooling.shape[-2:],
                                         mode='bilinear', align_corners=False).to(ref_feature.dtype)

        # id-level query
        pooled_ref_feat_for_id = self.mask_pooling(ref_feat_for_pooling, mask_for_pooling)  # bs, nm, c
        pooled_ref_feat_for_id = pooled_ref_feat_for_id.permute(1, 0 ,2) # bs, nm, c

        # ins-level query
        ins_labels = torch.unique(ref_label)
        uni_ref_label = ins_labels.cpu().tolist()
        nc = len(uni_ref_label) # number of category
        ref_feat_nshot = torch.split(ref_feat_for_pooling, split_size_or_sections=1, dim=0)
        ref_feat_nshot = torch.cat(ref_feat_nshot, dim=-1) # bs, c, h, w*nshot
        ref_mask_nshot = torch.split(mask_for_pooling, split_size_or_sections=1, dim=0)
        ref_mask_nshot = torch.cat(ref_mask_nshot, dim=-1) # bs, nc=1, h, w*nshot
        pooled_ref_feat_for_ins = self.mask_pooling(ref_feat_nshot, ref_mask_nshot)  # bs, nc, c

        # fusion reference and target
        image_feat = tar_feature.flatten(2) # B, C, HW
        image_feat = image_feat.permute(2, 0, 1)  # HW, B, C
        pooled_ref_feats = torch.cat([pooled_ref_feat_for_id, pooled_ref_feat_for_ins], dim=1) # B, Q, C
        pooled_ref_feats = pooled_ref_feats.permute(1, 0, 2) # Q, B, C
        for i in range(self.num_fusion_layers):
            pooled_ref_feats_new = self.fusion_layers[i](
                output=pooled_ref_feats,
                image_feat=image_feat,
            )
            image_feat_new = self.fusion_layers[i](
                output=image_feat,
                image_feat=pooled_ref_feats,
            )
            pooled_ref_feats = pooled_ref_feats_new + pooled_ref_feats
            image_feat = image_feat_new + image_feat

        image_feat = image_feat.permute(1, 0, 2) # B, HW, C
        pooled_ref_feats = pooled_ref_feats.permute(1, 0, 2) # B, nm+nc, C

        id_query_feat, sem_ref_feat = pooled_ref_feats[:, :nm], pooled_ref_feats[:, nm:]

        image_pe = self.pe_layer(tar_feature, None).flatten(2).permute(0, 2, 1)  # BxCxHxW -> BxCxHW == B x HW x C

        # id
        # cal id p.e.
        image_feat_norm = F.normalize(image_feat, dim=-1, p=2)
        id_query_feat_norm = F.normalize(id_query_feat, dim=-1, p=2)
        corr_matrix = torch.einsum('nac,nbc->nab', id_query_feat_norm, image_feat_norm)  # 1, nm, HW

        id_corr_matrix = self.apply_gaussian_kernel(corr_matrix, h, w)
        id_dist = torch.softmax(id_corr_matrix * self.temp, dim=-1)
        id_embed = torch.einsum('nab,nbc->nac', id_dist, image_pe)
        output_id = id_query_feat # bs, nm, c

        # ins
        output_ins = self.query_feat.weight.unsqueeze(0).repeat(bs, 1, 1)
        ins_embed = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)

        # sem
        neg_class_feat = self.neg_class_embed.weight[None,].repeat(bs, 1, 1)
        output_sem = torch.cat([sem_ref_feat, neg_class_feat], dim=1) # nc + 1
        sem_embed = torch.zeros_like(output_sem)

        id_grained_embed, ins_grained_embed, sem_grained_embed = self.query_grained_embed.weight[0], self.query_grained_embed.weight[1], self.query_grained_embed.weight[2]
        id_grained_embed = id_grained_embed.unsqueeze(0).unsqueeze(0)
        ins_grained_embed = ins_grained_embed.unsqueeze(0).unsqueeze(0)
        sem_grained_embed = sem_grained_embed.unsqueeze(0).unsqueeze(0)

        output_id = output_id + id_grained_embed     # nm
        output_ins = output_ins + ins_grained_embed  # nq
        output_sem = output_sem + sem_grained_embed  # nc + 1

        # ref labels
        id_labels = ref_label[None, :] # 1, nm
        sem_labels = ins_labels[None, :]  # 1, nc

        output = [output_id, output_ins, output_sem]
        query_embed = [id_embed, ins_embed, sem_embed]

        image_feat = image_feat.permute(1, 0, 2) # HW, B, C
        image_pe = image_pe.permute(1, 0, 2)
        output = [out.permute(1, 0, 2) for out in output]
        query_embed = [qe.permute(1, 0, 2) for qe in query_embed]

        sa_attn_mask = torch.ones((nm+nq+nc+1, nm+nq+nc+1), device=image_feat.device)
        sa_attn_mask[:nm, :nm] = 0
        sa_attn_mask[nm:, nm:] = 0
        sa_attn_mask = (sa_attn_mask.unsqueeze(0).unsqueeze(0).repeat(1, self.num_heads, 1, 1).flatten(0, 1) == 1).bool()
        sa_attn_mask = sa_attn_mask.detach()

        preds_id_class = []
        preds_ins_class = []
        preds_id_mask = []
        preds_ins_mask = []

        outputs_class, outputs_mask, ca_attn_mask, _ = self.forward_prediction_heads(
            output,
            mask_features=tar_mask_feature,
            attn_mask_target_size=(h, w)
        )

        outputs_class_id, outputs_class_ins = outputs_class
        preds_id_class.append(outputs_class_id)
        preds_ins_class.append(outputs_class_ins)

        outputs_mask_id, outputs_mask_ins = outputs_mask
        preds_id_mask.append(outputs_mask_id)
        preds_ins_mask.append(outputs_mask_ins)

        for i in range(self.num_layers):
            if ca_attn_mask is not None:
                ca_attn_mask[torch.where(ca_attn_mask.sum(-1) == ca_attn_mask.shape[-1])] = False

            output = self.transformer_layers[i](
                output=output,
                image_feat=image_feat,
                query_embed=query_embed,
                image_pe=image_pe,
                sa_attn_mask=sa_attn_mask,
                ca_attn_mask=ca_attn_mask
            )

            outputs_class, outputs_mask, attn_mask, mask_embed = self.forward_prediction_heads(
                output,
                mask_features=tar_mask_feature,
                attn_mask_target_size=(h, w)
            )

            outputs_class_id, outputs_class_ins = outputs_class
            preds_id_class.append(outputs_class_id)
            preds_ins_class.append(outputs_class_ins)

            outputs_mask_id, outputs_mask_ins = outputs_mask
            preds_id_mask.append(outputs_mask_id)
            preds_ins_mask.append(outputs_mask_ins)


        id_hidden_states, ins_hidden_states = mask_embed
        id_hidden_states = id_hidden_states
        ins_hidden_states = ins_hidden_states

        out = {
            'ref_id_labels': id_labels,
            'ref_sem_labels': sem_labels,
            'pred_id_logits': preds_id_class[-1],
            'pred_ins_logits': preds_ins_class[-1],
            'pred_id_masks': preds_id_mask[-1],
            'pred_ins_masks': preds_ins_mask[-1],
            'id_hidden_states': id_hidden_states,
            'ins_hidden_states': ins_hidden_states,
            'aux_outputs': self._set_aux_loss(preds_id_class, preds_ins_class, preds_id_mask, preds_ins_mask)
        }
        return out

    @torch.jit.unused
    def _set_aux_loss(self, preds_id_class, preds_ins_class, preds_id_mask, preds_ins_mask):

        if self.mask_classification:
            return [
                {"pred_id_logits": a, "pred_ins_logits": b, "pred_id_masks": c, "pred_ins_masks": d}
                for a, b, c, d in zip(preds_id_class[:-1], preds_ins_class[:-1], preds_id_mask[:-1], preds_ins_mask[:-1])
            ]
        else:
            return [
                {"pred_id_masks": c, "pred_ins_masks": d}
                for c, d in zip(preds_id_mask[:-1], preds_ins_mask[:-1])
            ]

    def forward(
        self,
        ref_features: Tensor,
        tar_features: Tensor,
        ref_masks: Tensor,
        ref_labels: Tensor
    ) -> Tuple[Tensor, Tensor]:

        outputs = []
        out = self.forward_per_image(
            ref_features, tar_features, ref_masks, ref_labels
        )
        outputs.append(out)

        return outputs


def build_mformer(args):

    return MFormer(
        depth=args.transformer_depth,
        embedding_dim=args.feat_chans,
        num_heads=args.transformer_nheads,
        mlp_dim=args.transformer_mlp_dim,
        mask_dim=args.transformer_mask_dim,
        fusion_layer_depth=args.transformer_fusion_layer_depth,
        num_queries=args.transformer_num_queries,
        pre_norm=args.transformer_pre_norm
    )

