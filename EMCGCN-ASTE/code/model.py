import torch
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import math
import copy

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features)) # (768,)
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class RefiningStrategy(nn.Module): # 细化策略 主要是对输入的边进一步细化和处理
    def __init__(self, hidden_dim, edge_dim, dim_e, dropout_ratio=0.5):
        super(RefiningStrategy, self).__init__()
        self.hidden_dim = hidden_dim # hidden_dim=300
        self.edge_dim = edge_dim # edge_dim=50
        self.dim_e = dim_e # dim_e=10
        self.dropout = dropout_ratio
        self.W = nn.Linear(self.hidden_dim * 2 + self.edge_dim * 3, self.dim_e) # self.W = nn.Linear(750,10)
        # self.W = nn.Linear(self.hidden_dim * 2 + self.edge_dim * 1, self.dim_e)

    def forward(self, edge, node1, node2):
        batch, seq, seq, edge_dim = edge.shape
        node = torch.cat([node1, node2], dim=-1)

        edge_diag = torch.diagonal(edge, offset=0, dim1=1, dim2=2).permute(0, 2, 1).contiguous()
        # torch.diagonal(input,offset,dim1,dim2) 求对角线元素
        # input:tensor ; offset:偏移量
        # permute() 实现对任意高维矩阵进行转置
        # contiguous() 如果view之前调用了transpose(),permute()等，就需要contiguous，返回一个contiguous copy
        edge_i = edge_diag.unsqueeze(1).expand(batch, seq, seq, edge_dim)
        edge_j = edge_i.permute(0, 2, 1, 3).contiguous()
        edge = self.W(torch.cat([edge, edge_i, edge_j, node], dim=-1))
        # edge = self.W(torch.cat([edge, node], dim=-1))

        return edge


class GraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, device, gcn_dim, edge_dim, dep_embed_dim, pooling='avg'):
        super(GraphConvLayer, self).__init__()
        self.gcn_dim = gcn_dim # 300
        self.edge_dim = edge_dim # 50
        self.dep_embed_dim = dep_embed_dim # 10
        self.device = device
        self.pooling = pooling
        self.layernorm = LayerNorm(self.gcn_dim)
        # self.highway = RefiningStrategy(gcn_dim, self.edge_dim, self.dep_embed_dim, dropout_ratio=0.5)
        # RefineStrategy(300,50,10)
        self.W = nn.Linear(self.gcn_dim, self.gcn_dim)
        self.W1 = nn.Linear(self.gcn_dim*2,self.dep_embed_dim)

    def forward(self, weight_prob_softmax, weight_adj, gcn_inputs, self_loop):
        batch, seq, dim = gcn_inputs.shape # gcn_inputs=tensor(batch_size, 102, 300)
        weight_prob_softmax = weight_prob_softmax.permute(0, 3, 1, 2) # 4,50,102,102

        # gcn_inputs 升维 batch_size, 50, 102, 300
        gcn_inputs = gcn_inputs.unsqueeze(1).expand(batch, self.edge_dim, seq, dim)

        weight_prob_softmax += self_loop # b,50,102,102
        Ax = torch.matmul(weight_prob_softmax, gcn_inputs) # batch_size, 50, 102, 300
        if self.pooling == 'avg':
            Ax = Ax.mean(dim=1)   # b,102,300
        elif self.pooling == 'max':
            Ax, _ = Ax.max(dim=1)
        elif self.pooling == 'sum':
            Ax = Ax.sum(dim=1)
        # Ax: [batch, seq, dim]
        gcn_outputs = self.W(Ax)
        gcn_outputs = self.layernorm(gcn_outputs)
        weights_gcn_outputs = F.relu(gcn_outputs) # 4,102,300

        node_outputs = weights_gcn_outputs
        weight_prob_softmax = weight_prob_softmax.permute(0, 2, 3, 1).contiguous()
        node_outputs1 = node_outputs.unsqueeze(1).expand(batch, seq, seq, dim) # 4,102,102,300
        node_outputs2 = node_outputs1.permute(0, 2, 1, 3).contiguous() # 4,102,102,300
        #
        # edge_outputs = self.highway(weight_adj, node_outputs1, node_outputs2) # weight_adj (4,102,102,50)
        #
        # edge_outputs (4,102,102,10)
        edge_outputs1 = self.W1(torch.cat((node_outputs1, node_outputs2), dim=-1)) # without refining strategy
        return node_outputs, edge_outputs1


class Biaffine(nn.Module):# 双仿射
    def __init__(self, args, in1_features, in2_features, out_features, bias=(True, True)):
        # bias_x , bias_y = True
        super(Biaffine, self).__init__()
        self.args = args
        self.in1_features = in1_features # 300
        self.in2_features = in2_features # 300
        self.out_features = out_features # 10
        self.bias = bias # (True,True)
        self.linear_input_size = in1_features + int(bias[0]) # 301
        self.linear_output_size = out_features * (in2_features + int(bias[1])) # 3010
        self.linear = torch.nn.Linear(in_features=self.linear_input_size,
                                    out_features=self.linear_output_size,
                                    bias=False)

    def forward(self, input1, input2):
        # batch_size, len1, len2, dim1 = input1.size()
        # batch_size, len3, len4, dim2 = input2.size()
        batch_size, len1, dim1 = input1.size() # 4,102,300
        batch_size, len2, dim2 = input2.size() # 4,102,300
        if self.bias[0]:
            ones = torch.ones(batch_size, len1, 1).to(self.args.device)
            input1 = torch.cat((input1, ones), dim=2) # 4,102,301
            dim1 += 1
        if self.bias[1]:
            ones = torch.ones(batch_size, len2, 1).to(self.args.device)
            input2 = torch.cat((input2, ones), dim=2) # 4,102,301
            dim2 += 1
        affine = self.linear(input1)
        affine = affine.view(batch_size, len1*self.out_features, dim2)
        input2 = torch.transpose(input2, 1, 2) # 4,301,102
        biaffine = torch.bmm(affine, input2) # 乘法
        biaffine = torch.transpose(biaffine, 1, 2)
        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_features)
        return biaffine # 4,102,102,10


class EMCGCN(torch.nn.Module):
    def __init__(self, args):
        super(EMCGCN, self).__init__()
        self.args = args
        # 模型加载 预训练模型
        self.bert = BertModel.from_pretrained(args.bert_model_path, return_dict = False)
        # 利用分词器构建模型输入
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
        self.dropout_output = torch.nn.Dropout(args.emb_dropout) # dropout
        # 位置信息 初始化词向量


        # self.bi_lstm = nn.LSTM(input_size=768, hidden_size=self.gcn_dim, num_layers=self.num_layers,
        #                       bidirectional=True, batch_first=True)


        self.post_emb = torch.nn.Embedding(args.post_size, args.class_num, padding_idx=0) # position while class_num = 10 , post_emb = tensor(81,10)
        # 依赖树
        self.deprel_emb = torch.nn.Embedding(args.deprel_size, args.class_num, padding_idx=0) # deprel 依存关系 deprel_emb = tensor(45,10)
        # 词性组合信息
        self.postag_emb = torch.nn.Embedding(args.postag_size, args.class_num, padding_idx=0) # postag 词性组合 postag_emb = tensor(855,10)
        self.synpost_emb = torch.nn.Embedding(args.synpost_size, args.class_num, padding_idx=0) # 句法位置 synpost_emb = tensor(7,10)
        self.triplet_biaffine = Biaffine(args, args.gcn_dim, args.gcn_dim, args.class_num, bias=(True, True)) # biaffine 双仿射注意力模块
        # triplet_biaffine = Biaffine(args,args.gcn_dim = 300 , args.gcn_dim = 300 , args.class_num = 10) in_feature=300,out_feature=3010
        self.ap_fc = nn.Linear(args.bert_feature_dim, args.gcn_dim)   # args.bert_feature_dim 768 , args.gcn_dim 300
        self.op_fc = nn.Linear(args.bert_feature_dim, args.gcn_dim)

        self.dense = nn.Linear(args.bert_feature_dim, args.gcn_dim)
        self.num_layers = args.num_layers # 1
        # nn.ModuleList()

        self.gcn_layers = nn.ModuleList()

        self.layernorm = LayerNorm(args.bert_feature_dim)

        for i in range(self.num_layers):
            self.gcn_layers.append(
                GraphConvLayer(args.device, args.gcn_dim, 5*args.class_num, args.class_num, args.pooling)) # class_num 10
                # gcn_dim=300,class_num=10

    def forward(self, tokens, masks, word_pair_position, word_pair_deprel, word_pair_pos, word_pair_synpost):
        bert_feature, _ = self.bert(tokens, masks) # 4,102,768
        bert_feature = self.dropout_output(bert_feature)

        batch, seq = masks.shape # 4,102
        tensor_masks = masks.unsqueeze(1) .expand(batch, seq, seq).unsqueeze(-1) # 4,102,102,1
        # * multi-feature
        # 生成词对表示
        word_pair_post_emb = self.post_emb(word_pair_position) # 4,102,102,10  word_pair_positon 8,102,102
        word_pair_deprel_emb = self.deprel_emb(word_pair_deprel) # 4,102,102,10 word_pair_deprel 8,102,102
        word_pair_postag_emb = self.postag_emb(word_pair_pos) # 4,102,102,10 word_pair_pos 8,102,102
        word_pair_synpost_emb = self.synpost_emb(word_pair_synpost) # 4,102,102,10 word_pair_synpost 8,102,102

        # BiAffine
        ap_node = F.relu(self.ap_fc(bert_feature)) #4,102,300
        op_node = F.relu(self.op_fc(bert_feature)) # 4,102,300
        biaffine_edge = self.triplet_biaffine(ap_node, op_node)
        gcn_input = F.relu(self.dense(bert_feature)) # 4,102,300
        gcn_outputs = gcn_input  # BATCH_SIZE , 102 , 300

        # 拼接
        weight_prob_list = [biaffine_edge, word_pair_post_emb, word_pair_deprel_emb, word_pair_postag_emb, word_pair_synpost_emb] # 4,102,102,10
        
        biaffine_edge_softmax = F.softmax(biaffine_edge, dim=-1) * tensor_masks
        word_pair_post_emb_softmax = F.softmax(word_pair_post_emb, dim=-1) * tensor_masks
        word_pair_deprel_emb_softmax = F.softmax(word_pair_deprel_emb, dim=-1) * tensor_masks
        word_pair_postag_emb_softmax = F.softmax(word_pair_postag_emb, dim=-1) * tensor_masks
        word_pair_synpost_emb_softmax = F.softmax(word_pair_synpost_emb, dim=-1) * tensor_masks

        self_loop = [] # 自环矩阵
        for _ in range(batch):
            self_loop.append(torch.eye(seq)) # torch.eye()生成对角线全1，其余部分全0的二维数组 102,102
        self_loop = torch.stack(self_loop).to(self.args.device).unsqueeze(1).expand(batch, 5*self.args.class_num, seq, seq) * tensor_masks.permute(0, 3, 1, 2).contiguous()
        # self_loop = torch.stack(self_loop).to(self.args.device).unsqueeze(1).expand(batch, 3*self.args.class_num, seq, seq) * tensor_masks.permute(0, 3, 1, 2).contiguous()

        # stack() 拼接 BATCH_SIZE 50 102 102
        weight_prob = torch.cat([biaffine_edge, word_pair_post_emb, word_pair_deprel_emb, \
            word_pair_postag_emb, word_pair_synpost_emb], dim=-1) # 4,102,102,50
        weight_prob_softmax = torch.cat([biaffine_edge_softmax, word_pair_post_emb_softmax, \
            word_pair_deprel_emb_softmax, word_pair_postag_emb_softmax, word_pair_synpost_emb_softmax], dim=-1)
        # weight_prob = torch.cat([biaffine_edge, word_pair_post_emb,  word_pair_synpost_emb], dim=-1)  # 4,102,102,30
        # weight_prob_softmax = torch.cat([biaffine_edge_softmax, word_pair_post_emb_softmax, word_pair_synpost_emb_softmax], dim=-1)

        for _layer in range(self.num_layers):
            gcn_outputs, weight_prob = self.gcn_layers[_layer](weight_prob_softmax, weight_prob, gcn_outputs, self_loop)  # [batch, seq, dim]
            weight_prob_list.append(weight_prob)
        return weight_prob_list

