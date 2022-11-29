from torch import nn
import torch.nn.functional
from Constants import *
import math
import numpy as np

class DepthwiseSeparableConvolution(nn.Module):
    def __init__(self, in_ch, out_ch, k, dim=1, bias=True):
        super().__init__()
        if dim == 1:
            self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                            padding=k // 2, bias=bias)
            self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
        elif dim == 2:
            self.depthwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                            padding=k // 2, bias=bias)
            self.pointwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
        else:
            raise Exception("Wrong dimension for Depthwise Separable Convolution!")
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.depthwise_conv.bias, 0.0)
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.pointwise_conv.bias, 0.0)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class POSEncoder(nn.Module):
    def __init__(self, length):
        super().__init__()
        freqs = torch.Tensor(
            [10000 ** (-i / d_model) if i % 2 == 0 else -10000 ** ((1 - i) / d_model) for i in range(d_model)]).unsqueeze(dim=1)
        phases = torch.Tensor([0 if i % 2 == 0 else math.pi / 2 for i in range(d_model)]).unsqueeze(dim=1)
        pos = torch.arange(length).repeat(d_model, 1).to(torch.float)
        self.pos_encoding = nn.Parameter(torch.sin(torch.add(torch.mul(pos, freqs), phases)), requires_grad=False)

    def forward(self, x):
        x = x + self.pos_encoding
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, d_model)
        self.a = 1 / math.sqrt(d_k)

    def forward(self, x, mask):
        bs, _, l_x = x.size()
        x = x.transpose(1,2)
        k = self.k_linear(x).view(bs, l_x, n_head, d_k)
        q = self.q_linear(x).view(bs, l_x, n_head, d_k)
        v = self.v_linear(x).view(bs, l_x, n_head, d_k)
        q = q.permute(2, 0, 1, 3).contiguous().view(bs*n_head, l_x, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(bs*n_head, l_x, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(bs*n_head, l_x, d_k)
        mask = mask.unsqueeze(1).expand(-1, l_x, -1).repeat(n_head, 1, 1)
        
        attn = torch.bmm(q, k.transpose(1, 2)) * self.a
        attn = mask_logits(attn, mask)
        attn = torch.nn.functional.softmax(attn, dim=2)
        attn = self.dropout(attn)
            
        out = torch.bmm(attn, v)
        out = out.view(n_head, bs, l_x, d_k).permute(1,2,0,3).contiguous().view(bs, l_x, d_model)
        out = self.fc(out)
        out = self.dropout(out)
        return out.transpose(1,2)

class EncoderBlock(nn.Module):
    def __init__(self, conv_num: int, ch_num: int, k: int, length: int):
        super().__init__()
        self.convs = nn.ModuleList([DepthwiseSeparableConvolution(ch_num, ch_num, k) for _ in range(conv_num)])
        self.self_att = MultiHeadAttention()
        self.fc = nn.Linear(ch_num, ch_num, bias=True)
        self.pos_encoder_layer = POSEncoder(length)
        self.normb = nn.LayerNorm([d_model, length])
        self.norms = nn.ModuleList([nn.LayerNorm([d_model, length]) for _ in range(conv_num)])
        self.norme = nn.LayerNorm([d_model, length])
        self.L = conv_num

    def forward(self, x, mask):
        out = self.pos_encoder_layer(x)
        res = out
        out = self.normb(out)
        for i, conv in enumerate(self.convs):
            out = conv(out)
            out = torch.nn.functional.relu(out)
            out = out + res
            if (i + 1) % 2 == 0:
                p_drop = dropout * (i + 1) / self.L
                out = torch.nn.functional.dropout(out, p=p_drop, training=self.training)
            res = out
            out = self.norms[i](out)
        out = self.self_att(out, mask)
        out = out + res
        out = torch.nn.functional.dropout(out, p=dropout, training=self.training)
        res = out
        out = self.norme(out)
        out = self.fc(out.transpose(1, 2)).transpose(1, 2)
        out = torch.nn.functional.relu(out)
        out = out + res
        out = torch.nn.functional.dropout(out, p=dropout, training=self.training)
        return out

class ContextQueryAttention(nn.Module):
    def __init__(self):
        super().__init__()
        w = torch.empty(d_model * 3)
        lim = 1 / d_model
        nn.init.uniform_(w, -math.sqrt(lim), math.sqrt(lim))
        self.w = nn.Parameter(w)

    def forward(self, C, Q, cmask, qmask):
        C = C.transpose(1, 2)
        Q = Q.transpose(1, 2)
        cmask = cmask.unsqueeze(2)
        qmask = qmask.unsqueeze(1)
        shape = (C.size(0), C.size(1), Q.size(1), C.size(2))
        Ct = C.unsqueeze(2).expand(shape)
        Qt = Q.unsqueeze(1).expand(shape)
        CQ = torch.mul(Ct, Qt)
        S = torch.cat([Ct, Qt, CQ], dim=3)
        S = torch.matmul(S, self.w)
        S1 = torch.nn.functional.softmax(mask_logits(S, qmask), dim=2)
        S2 = torch.nn.functional.softmax(mask_logits(S, cmask), dim=1)
        A = torch.bmm(S1, Q)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)
        out = torch.nn.functional.dropout(out, p=dropout, training=self.training)
        return out.transpose(1, 2)

class Pointer(nn.Module):
    def __init__(self):
        super().__init__()
        w1 = torch.empty(d_model * 2)
        w2 = torch.empty(d_model * 2)
        lim = 3 / (2 * d_model)
        nn.init.uniform_(w1, -math.sqrt(lim), math.sqrt(lim))
        nn.init.uniform_(w2, -math.sqrt(lim), math.sqrt(lim))
        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.linear_layer_1 = nn.Linear(len_c, len_c)
        self.linear_layer_2 = nn.Linear(len_c, len_c)

    def forward(self, M1, M2, M3, mask):
        X1 = torch.cat([M1, M2], dim=1)
        X2 = torch.cat([M1, M3], dim=1)
        Y1 = torch.matmul(self.w1, X1)
        Y2 = torch.matmul(self.w2, X2)
        p1 = self.linear_layer_1(Y1)
        p2 = self.linear_layer_2(Y2)
        Y1 = mask_logits(Y1, mask)
        Y2 = mask_logits(Y2, mask)
        return p1, p2



class ModelA(nn.Module):
    def __init__(self):
        super(ModelA, self).__init__()

        self.default_config = AutoConfig.from_pretrained('bert-base-uncased', output_hidden_states=False)
        self.bert_model = AutoModel.from_pretrained("bert-base-uncased", config=self.default_config)

        self.qa_outputs = nn.Linear(self.default_config.hidden_size, 2)

        self.context_conv = DepthwiseSeparableConvolution(d_word, d_model, 5)
        self.question_conv = DepthwiseSeparableConvolution(d_word, d_model, 5)
        self.context_embedding_encoder_block = EncoderBlock(conv_num=4, ch_num=d_model, k=7, length=len_c)
        self.question_embedding_encoder_block = EncoderBlock(conv_num=4, ch_num=d_model, k=7, length=len_q)
        self.context_query_attention_layer = ContextQueryAttention()
        self.context_query_resizer = DepthwiseSeparableConvolution(d_model * 4, d_model, 5)
        enc_blk = EncoderBlock(conv_num=2, ch_num=d_model, k=5, length=len_c)
        # stacked embedding encoder blocks
        self.stacked_embedding_encoder_blocks = nn.ModuleList([enc_blk] * 7)
        self.out = Pointer()

    def forward(self, input_ids, token_ids=None, attention_mask=None, start_positions=None, end_positions=None):
        # create tensors for question and context
        context_mask = token_ids * attention_mask
        question_mask = ((1-context_mask) * attention_mask)
        
        question_output = self.bert_model(input_ids, attention_mask=question_mask, token_type_ids=token_ids)[0]
        context_output = self.bert_model(input_ids, attention_mask=context_mask, token_type_ids=token_ids)[0]


        # question_output, _ = self.bert(input_ids, token_ids, question_mask, output_all_encoded_layers=False)
        # context_output, _ = self.bert(input_ids, token_ids, context_mask, output_all_encoded_layers=False)
        
        question_output = question_output.permute(0, 2, 1).float()
        context_output = context_output.permute(0, 2, 1).float()
        context_mask = context_mask.float()
        question_mask = question_mask.float()
        C = self.context_conv(context_output)  
        Q = self.question_conv(question_output)
        Ce = self.context_embedding_encoder_block(C, context_mask)
        Qe = self.question_embedding_encoder_block(Q, question_mask)
        
        X = self.context_query_attention_layer(Ce, Qe, context_mask, question_mask)
        M1 = self.context_query_resizer(X)
        for enc in self.stacked_embedding_encoder_blocks: M1 = enc(M1, context_mask)
        M2 = M1
        for enc in self.stacked_embedding_encoder_blocks: M2 = enc(M2, context_mask)
        M3 = M2
        for enc in self.stacked_embedding_encoder_blocks: M3 = enc(M3, context_mask)
        start_logits, end_logits = self.out(M1, M2, M3, context_mask)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_function = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_function(start_logits, start_positions)
            end_loss = loss_function(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits