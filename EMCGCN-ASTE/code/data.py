import math
import torch
import numpy as np
from collections import OrderedDict, defaultdict
from transformers import BertTokenizer


sentiment2id = {'negative': 3, 'neutral': 4, 'positive': 5}

label = ['N', 'B-A', 'I-A', 'A', 'B-O', 'I-O', 'O', 'negative', 'neutral', 'positive']
# label2id = {'N': 0, 'B-A': 1, 'I-A': 2, 'A': 3, 'B-O': 4, 'I-O': 5, 'O': 6, 'negative': 7, 'neutral': 8, 'positive': 9}
# label = ['N', 'A', 'O', 'negative', 'neutral', 'positive'] # 6
# label2id = ：{'N':0,'A':1,'O':2,'negative':3,'netural':4,'postive':5}
label2id, id2label = OrderedDict(), OrderedDict()
for i, v in enumerate(label):
    label2id[v] = i
    id2label[i] = v


def get_spans(tags):
    '''for BIO tag'''
    # 提取json中标注好的数据
    tags = tags.strip().split() # This\\O palce\\O
    length = len(tags)
    spans = []
    start = -1 # 标记位
    for i in range(length):
        if tags[i].endswith('B'): # 最后以B结尾
            if start != -1:
                spans.append([start, i - 1])
            start = i
        elif tags[i].endswith('O'):
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length - 1])
    return spans


def get_evaluate_spans(tags, length, token_range):
    '''for BIO tag'''
    spans = []
    start = -1
    for i in range(length):
        l, r = token_range[i]
        if tags[l] == -1:
            continue
        elif tags[l] == 1:
            if start != -1:
                spans.append([start, i - 1])
            start = i
        elif tags[l] == 0:
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length - 1])
    return spans


class Instance(object):
    def __init__(self, tokenizer, sentence_pack, post_vocab, deprel_vocab, postag_vocab, synpost_vocab, args):
        self.id = sentence_pack['id'] # sentence_pack 为train.json文件中的数据
        self.sentence = sentence_pack['sentence'] # 句子主体
        self.tokens = self.sentence.strip().split() # .strip() 返回删除前导和尾随空格的字符串副本 .split()将一个字符串分割为多个字符串数组
        self.postag = sentence_pack['postag'] # 词性标签
        self.head = sentence_pack['head']
        # 单词之间的依赖关系
        self.deprel = sentence_pack['deprel']
        self.sen_length = len(self.tokens)
        self.token_range = [] # 每个token的界定范围？
        self.bert_tokens = tokenizer.encode(self.sentence) # tokenizer.encode()直接输出id，需要转换为tensor
        # bert_token->list   tokenizer.tokenize,先分词然后在转换成id，直接输出tensor
        self.length = len(self.bert_tokens)
        self.bert_tokens_padding = torch.zeros(args.max_sequence_len).long() # 转化成长整数
        self.aspect_tags = torch.zeros(args.max_sequence_len).long()
        self.opinion_tags = torch.zeros(args.max_sequence_len).long()

        self.tags = torch.zeros(args.max_sequence_len, args.max_sequence_len).long() # arg.max_sequence_len = 102
        self.tags_symmetry = torch.zeros(args.max_sequence_len, args.max_sequence_len).long() # tags_symmetry 对称？

        self.mask = torch.zeros(args.max_sequence_len)

        for i in range(self.length): # 遍历整个句子的长度，填充部分全部置为0，非填充部分则为tokens的真实值
            self.bert_tokens_padding[i] = self.bert_tokens[i]
        self.mask[:self.length] = 1 # 填充部分的mask为1，非填充部分不做mask处理 102

        token_start = 1 # token_range
        for i, w, in enumerate(self.tokens):
            token_end = token_start + len(tokenizer.encode(w, add_special_tokens=False))
            # tokenizer.encode(w,add_special_tokens=False)
            # ==
            # seg = tokenizer.tokenize(w)
            # encoder = tokenizer.convert_tokens_to_ids(seg)

            self.token_range.append([token_start, token_end-1])
            token_start = token_end
        assert self.length == self.token_range[-1][-1]+2 # [CLS] [SEP] 如果不对应，则抛出异常
        # 把初始化为0的方面词和意见词
        self.aspect_tags[self.length:] = -1 #把length之外的记为-1
        self.aspect_tags[0] = -1 #排除[CLS]
        self.aspect_tags[self.length-1] = -1 #排除[SEP] aspect_tags tensor(102,)
        # opinion_tags 方式同 aspect_tags
        self.opinion_tags[self.length:] = -1
        self.opinion_tags[0] = -1
        self.opinion_tags[self.length - 1] = -1 # opinion_tags->tensor(102,)

        self.tags[:, :] = -1 # tag[102,102]
        self.tags_symmetry[:, :] = -1 # tag_symmetry[102,102]
        for i in range(1, self.length-1):
            for j in range(i, self.length-1):
                self.tags[i][j] = 0 # 其余部分 =  -1

        for triple in sentence_pack['triples']: # json中triples元组
            aspect = triple['target_tags'] # 分别提取出三元组的对应标签
            opinion = triple['opinion_tags']
            aspect_span = get_spans(aspect)
            opinion_span = get_spans(opinion)

            '''set tag for aspect'''
            # 这段代码实现了一个方面级情感分析模型中的标注功能。
            # 对于传入的aspect_span（方面词的标记），对应的词语会被标注为方面词和情感词。
            # 具体实现过程如下：
            # 对于aspect_span中每个方面词的起始位置和结束位置，找到这些位置对应的词在标注的序列中的起始和结束位置。
            #
            # 根据起始和结束位置，对于这些词的所有可能的子序列，设置相应的标注，具体的标注分为三类：
            # B - A（方面词的开始位置），I - A（方面词的中间位置），A（方面词的结束位置）。
            #
            # 对于每个方面词，将其第一个词标注为1（表示B - A），将其余词标注为2（表示I - A），
            # 并将不在这个方面词中的词的标注设为 - 1（表示该位置不需要进行标注）。
            #
            # 对于每个子词，将其标注为 - 1（表示该位置不需要进行标注），以便后续的处理。
            for l, r in aspect_span:
                start = self.token_range[l][0]
                end = self.token_range[r][1]
                for i in range(start, end+1):
                    for j in range(i, end+1):
                        # self.tags[i][j] = label2id['A'] # 对所有的方面词都标记为A
                        if j == start: # 标记方面词的标记，B-A , I-A , A
                            self.tags[i][j] = label2id['B-A']
                        elif j == i:
                            self.tags[i][j] = label2id['I-A']
                        else:
                            self.tags[i][j] = label2id['A']

                for i in range(l, r+1):
                    set_tag = 1 if i == l else 2
                    al, ar = self.token_range[i]
                    self.aspect_tags[al] = set_tag
                    self.aspect_tags[al+1:ar+1] = -1
                    '''mask positions of sub words'''
                    self.tags[al+1:ar+1, :] = -1 # tags 102，102
                    self.tags[:, al+1:ar+1] = -1

            '''set tag for opinion'''
            for l, r in opinion_span:
                start = self.token_range[l][0]
                end = self.token_range[r][1]
                for i in range(start, end+1):
                    for j in range(i, end+1):
                        # self.tags[i][j] = label2id['O']
                        if j == start:
                            self.tags[i][j] = label2id['B-O']
                        elif j == i:
                            self.tags[i][j] = label2id['I-O']
                        else:
                            self.tags[i][j] = label2id['O']

                for i in range(l, r+1):
                    set_tag = 1 if i == l else 2
                    pl, pr = self.token_range[i]
                    self.opinion_tags[pl] = set_tag
                    self.opinion_tags[pl+1:pr+1] = -1
                    self.tags[pl+1:pr+1, :] = -1
                    self.tags[:, pl+1:pr+1] = -1

            for al, ar in aspect_span:
                for pl, pr in opinion_span:
                    for i in range(al, ar+1):
                        for j in range(pl, pr+1):
                            sal, sar = self.token_range[i]
                            spl, spr = self.token_range[j]
                            self.tags[sal:sar+1, spl:spr+1] = -1
                            if args.task == 'pair':
                                if i > j:
                                    self.tags[spl][sal] = 7
                                else:
                                    self.tags[sal][spl] = 7
                            elif args.task == 'triplet':
                                if i > j:
                                    self.tags[spl][sal] = label2id[triple['sentiment']] # sentiment -> id
                                else:
                                    self.tags[sal][spl] = label2id[triple['sentiment']]

        for i in range(1, self.length-1):
            for j in range(i, self.length-1):
                self.tags_symmetry[i][j] = self.tags[i][j]
                self.tags_symmetry[j][i] = self.tags_symmetry[i][j]
             
        '''1. generate position index of the word pair'''
        self.word_pair_position = torch.zeros(args.max_sequence_len, args.max_sequence_len).long()
        for i in range(len(self.tokens)):
            start, end = self.token_range[i][0], self.token_range[i][1]
            for j in range(len(self.tokens)):
                s, e = self.token_range[j][0], self.token_range[j][1]
                for row in range(start, end + 1):
                    for col in range(s, e + 1):
                        self.word_pair_position[row][col] = post_vocab.stoi.get(abs(row - col), post_vocab.unk_index)
        
        """2. generate deprel index of the word pair"""
        # 获取词对的依赖索引
        self.word_pair_deprel = torch.zeros(args.max_sequence_len, args.max_sequence_len).long()
        for i in range(len(self.tokens)):
            start = self.token_range[i][0]
            end = self.token_range[i][1]
            for j in range(start, end + 1):
                s, e = self.token_range[self.head[i] - 1] if self.head[i] != 0 else (0, 0)
                for k in range(s, e + 1):
                    self.word_pair_deprel[j][k] = deprel_vocab.stoi.get(self.deprel[i])
                    self.word_pair_deprel[k][j] = deprel_vocab.stoi.get(self.deprel[i])
                    self.word_pair_deprel[j][j] = deprel_vocab.stoi.get('self')
        
        """3. generate POS tag index of the word pair"""
        self.word_pair_pos = torch.zeros(args.max_sequence_len, args.max_sequence_len).long()
        for i in range(len(self.tokens)):
            start, end = self.token_range[i][0], self.token_range[i][1]
            for j in range(len(self.tokens)):
                s, e = self.token_range[j][0], self.token_range[j][1]
                for row in range(start, end + 1):
                    for col in range(s, e + 1):
                        self.word_pair_pos[row][col] = postag_vocab.stoi.get(tuple(sorted([self.postag[i], self.postag[j]])))

        """4. generate synpost index of the word pair"""
        self.word_pair_synpost = torch.zeros(args.max_sequence_len, args.max_sequence_len).long()
        tmp = [[0]*len(self.tokens) for _ in range(len(self.tokens))]
        for i in range(len(self.tokens)):
            j = self.head[i]
            if j == 0:
                continue
            tmp[i][j - 1] = 1
            tmp[j - 1][i] = 1

        tmp_dict = defaultdict(list)
        for i in range(len(self.tokens)):
            for j in range(len(self.tokens)):
                if tmp[i][j] == 1:
                    tmp_dict[i].append(j)
            
        word_level_degree = [[4]*len(self.tokens) for _ in range(len(self.tokens))]

        for i in range(len(self.tokens)):
            node_set = set()
            word_level_degree[i][i] = 0
            node_set.add(i)
            for j in tmp_dict[i]:
                if j not in node_set:
                    word_level_degree[i][j] = 1
                    node_set.add(j)
                for k in tmp_dict[j]:
                    if k not in node_set:
                        word_level_degree[i][k] = 2
                        node_set.add(k)
                        for g in tmp_dict[k]:
                            if g not in node_set:
                                word_level_degree[i][g] = 3
                                node_set.add(g)
        
        for i in range(len(self.tokens)):
            start, end = self.token_range[i][0], self.token_range[i][1]
            for j in range(len(self.tokens)):
                s, e = self.token_range[j][0], self.token_range[j][1]
                for row in range(start, end + 1):
                    for col in range(s, e + 1):
                        self.word_pair_synpost[row][col] = synpost_vocab.stoi.get(word_level_degree[i][j], synpost_vocab.unk_index)


def load_data_instances(sentence_packs, post_vocab, deprel_vocab, postag_vocab, synpost_vocab, args):
    instances = list() # 用list接收instances
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_path) # 加载bert分词器，生成向量
    for sentence_pack in sentence_packs:
        instances.append(Instance(tokenizer, sentence_pack, post_vocab, deprel_vocab, postag_vocab, synpost_vocab, args))
    return instances


class DataIterator(object):
    def __init__(self, instances, args):
        self.instances = instances
        self.args = args
        self.batch_count = math.ceil(len(instances)/args.batch_size)
        # math.ceil(x) 将x向上舍入到最接近的整数

    def get_batch(self, index):
        sentence_ids = []
        sentences = []
        sens_lens = []
        token_ranges = []
        bert_tokens = []
        lengths = []
        masks = []
        aspect_tags = []
        opinion_tags = []
        tags = []
        tags_symmetry = []
        word_pair_position = []
        word_pair_deprel = []
        word_pair_pos = []
        word_pair_synpost = []

        for i in range(index * self.args.batch_size,
                       min((index + 1) * self.args.batch_size, len(self.instances))): # 范围从index*batch_size -> index+1*batch_size,步长为len(instance)
            sentence_ids.append(self.instances[i].id) # 句子id
            sentences.append(self.instances[i].sentence) # 句子
            sens_lens.append(self.instances[i].sen_length) # 句子长度
            token_ranges.append(self.instances[i].token_range) # token_range
            bert_tokens.append(self.instances[i].bert_tokens_padding) # bert token
            lengths.append(self.instances[i].length)
            masks.append(self.instances[i].mask)
            aspect_tags.append(self.instances[i].aspect_tags) # 方面词标签
            opinion_tags.append(self.instances[i].opinion_tags) # 意见词标签
            # tags and tags_symmetry
            tags.append(self.instances[i].tags)
            tags_symmetry.append(self.instances[i].tags_symmetry)

            word_pair_position.append(self.instances[i].word_pair_position) # 词对位置
            word_pair_deprel.append(self.instances[i].word_pair_deprel)
            word_pair_pos.append(self.instances[i].word_pair_pos)
            word_pair_synpost.append(self.instances[i].word_pair_synpost)

        bert_tokens = torch.stack(bert_tokens).to(self.args.device) #拼接
        lengths = torch.tensor(lengths).to(self.args.device)
        masks = torch.stack(masks).to(self.args.device)
        aspect_tags = torch.stack(aspect_tags).to(self.args.device)
        opinion_tags = torch.stack(opinion_tags).to(self.args.device)
        tags = torch.stack(tags).to(self.args.device)
        tags_symmetry = torch.stack(tags_symmetry).to(self.args.device)
        word_pair_position = torch.stack(word_pair_position).to(self.args.device)
        word_pair_deprel = torch.stack(word_pair_deprel).to(self.args.device)
        word_pair_pos = torch.stack(word_pair_pos).to(self.args.device)
        word_pair_synpost = torch.stack(word_pair_synpost).to(self.args.device)

        return sentence_ids, sentences, bert_tokens, lengths, masks, sens_lens, token_ranges, aspect_tags, tags, \
            word_pair_position, word_pair_deprel, word_pair_pos, word_pair_synpost, tags_symmetry
