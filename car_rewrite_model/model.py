import logging
import random

import re
import jieba
import json
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
import copy
numpy.random.seed(6666)
from pyltp import Postagger

from simplex_base_model import SimplexBaseModel
from simplex_sdk import SimplexClient

logger = logging.getLogger(__name__)

partten = re.compile(r'[.a-zA-Z|零|一|二|三|四|五|六|七|八|九|十|百|千|万|号|升|尺|英尺|英寸|寸|米|牛米|吨|千米|千克|克|分|转|马|码|公里|代|里|英里|厘米|年|月|天|匹|点|几|倍|挡\d]+')
digits = re.compile(r'[\d\.]+|[零一二两三四五六七八九十百千万]*')

ignore_pos_tags = ['e', 'nl', 'nh', 'o', 'q', 'wp', 'm', 'u', 'x', 'h', 'g']


class CarRewriteSynonymsReplace(SimplexBaseModel):
    """调用同义词推荐模型进行汽车评论改写"""
    def __init__(self, *args, **kwargs):
        super(CarRewriteSynonymsReplace, self).__init__(*args, **kwargs)

        # self.synonyms_recom_model = SimplexClient('BertMaskedLM', url='https://alpha-model-serving.aidigger.com/api/v1/bert-masked-l-m-batch/predict')
        self.synonyms_recom_model = SimplexClient('BertMaskedLM', url='https://model-serving.aidigger.com/api/v1/bert-masked-l-m-batch/predict')
        self.min_short_sen_length = kwargs.get("min_short_sen_length", 3)  # 最小序列长度
        self.max_sen_length = kwargs.get("max_sen_length", 100)  # 最大序列长度，超过进行分句
        self.timeout = kwargs.get("timeout", 30)
        self.pos_model_file = self.download(kwargs["pos_model_path"])
        self.postagger = Postagger()
        self.postagger.load(self.pos_model_file)
        self.remove_colon=kwargs.get('remove_phrase_before_colon',True)
        self.all_mask=kwargs.get('all_mask',False)

    def mask_sequence_v1(self, tokens, num_mask=1, have_masked_token_ids=[]):
        """mask单词，返回mask单词后的字符串序列，以及被mask的单词列表"""
        masked_tokens = []
        masked_token_ids = []

        # candidates_token_ids = numpy.random.randint(0, len(tokens), num_mask+len(ignore_pos_tags)+len(have_masked_token_ids))
        candidates_token_ids = numpy.random.permutation(len(tokens))
        for masked_token_id in candidates_token_ids:
            if have_masked_token_ids and masked_token_id in have_masked_token_ids:
                continue
            masked_token = tokens[masked_token_id]
            pos_tag = list(self.postagger.postag([masked_token]))[0]
            # print('pos_tag: {}'.format(pos_tag))
            if pos_tag in ignore_pos_tags:
                continue
            else:
                masked_tokens.append(masked_token)
                masked_token_ids.append(masked_token_id)
                tokens[masked_token_id] = '<mask>'
            if len(masked_tokens) == num_mask:
                break

        masked_tokens = [x for _, x in sorted(zip(masked_token_ids, masked_tokens))]
        masked_token_ids.sort()
        return ''.join(tokens), masked_tokens, masked_token_ids

    def mask_sequence(self, tokens, num_mask=1, have_masked_token_ids=[]):
        """mask单词，返回mask单词后的字符串序列，以及被mask的单词列表"""
        masked_tokens = []
        masked_token_ids = []

        mask_margin_threshod = min(max(1, (len(tokens)-2) // num_mask - 1), 2)  # 1 或 2

        prev_mask_token_idx = -1
        for idx, token in enumerate(tokens[:-1]):  # 不mask第一个以及最后一个token
            pos_tag = list(self.postagger.postag([token]))[0]
            if pos_tag in ignore_pos_tags:
                continue
            else:
                if idx - prev_mask_token_idx <= mask_margin_threshod:  # 和上一个mask token的间隔小于mask_margin_threshod
                    continue
                else:
                    masked_token = token
                    masked_token_id = idx
                    masked_tokens.append(masked_token)
                    masked_token_ids.append(masked_token_id)
                    tokens[masked_token_id] = '<mask>'
                    prev_mask_token_idx = idx

            if len(masked_tokens) == num_mask:
                break

        masked_tokens = [x for _, x in sorted(zip(masked_token_ids, masked_tokens))]
        masked_token_ids.sort()
        return ''.join(tokens), masked_tokens, masked_token_ids

    def predict_v1(self, data, **kwargs):
        '''
        data: [{"id":int,
                "content":string,
                "brand":string,
                "series":string,
                "spec":string,
                "spec_name":string,
                "domain":string
                }, ...]
        output: [{
                "id":int,
                "rewrite_content":string,
                "masked_words": list,
                "replaced_words": list
                }, ...]
        '''
        if data is None or len(data) == 0:
            return []

        num_mask = kwargs.get("num_mask", 5)
        results = []
        for idx, item in enumerate(data):
            id = item["id"]
            domain = item["domain"]
            scontent = item["content"]
            tokens = jieba.lcut(scontent)
            if len(tokens) < self.min_short_sen_length:
                results.append({"id": id, "domain": domain, "content": scontent, "rewrite_content": scontent, "masked_words": [], "replaced_words": []})
                continue

            i = 0
            have_masked_token_ids = []
            all_masked_tokens = []
            all_synonsyms_tokens = []
            while i < num_mask:
                content, masked_tokens, masked_token_ids = self.mask_sequence_v1(copy.deepcopy(tokens), num_mask=1,
                                                                         have_masked_token_ids=have_masked_token_ids)
                if not masked_tokens:
                    i += 1
                    continue
                have_masked_token_ids.extend(masked_token_ids)
                data = {"context": content, "maskwords": masked_tokens, "topk": 1}

                ret = self.synonyms_recom_model.predict(data)
                # try:
                #     ret = self.synonyms_recom_model.predict(data)
                # except:
                #     i = num_mask
                #     continue

                for n, result in enumerate(ret):
                    synonym_token = result["candidates"][0]["word"]
                    masked_token = result["maskword"]
                    masked_token_id = masked_token_ids[n]
                    # print('origin_token: {}, synonym_token: {}'.format(tokens[masked_token_id], synonym_token))
                    tokens[masked_token_id] = synonym_token
                    all_masked_tokens.append(masked_token)
                    all_synonsyms_tokens.append(synonym_token)

                i += 1

            rewrite_content = ''.join(tokens)

            results.append({"id": id, "domain": domain, "content": scontent, "rewrite_content": rewrite_content, "masked_words": all_masked_tokens, "replaced_words": all_synonsyms_tokens})

        return results

    def predict_v2(self, data, **kwargs):
        '''
        data: [{"id":int,
                "content":string,
                "brand":string,
                "series":string,
                "spec":string,
                "spec_name":string,
                "domain":string
                }, ...]
        output: [{
                "id":int,
                "rewrite_content":string,
                "masked_words": list,
                "replaced_words": list
                }, ...]
        '''
        if data is None or len(data) == 0:
            return []

        # num_mask = kwargs.get("num_mask", 5)
        results = []
        for idx, item in enumerate(data):
            id = item["id"]
            domain = item["domain"]
            scontent = item["content"]
            tokens = jieba.lcut(scontent)
            len_tokens = len(tokens)
            num_mask = max(1, min(10, int(len_tokens * 0.3)))  ## 最小1，最大10
            if len(tokens) < self.min_short_sen_length:
                results.append({"id": id, "domain": domain, "content": scontent, "rewrite_content": scontent, "masked_words": [], "replaced_words": []})
                continue

            content, masked_tokens, masked_token_ids = self.mask_sequence(copy.deepcopy(tokens), num_mask=num_mask, have_masked_token_ids=[])

            if not masked_tokens:
                results.append(
                    {"id": id, "domain": domain, "content": scontent, "rewrite_content": scontent, "masked_words": [],
                     "replaced_words": []})
                continue

            data = {"context": content, "maskwords": masked_tokens, "topk": 1}

            all_masked_tokens = []
            all_synonsyms_tokens = []
            ret = self.synonyms_recom_model.predict(data)
            for i, result in enumerate(ret):
                synonym_token = result["candidates"][0]["word"]
                masked_token = result["maskword"]
                masked_token_id = masked_token_ids[i]
                # print('origin_token: {}, synonym_token: {}'.format(tokens[masked_token_id], synonym_token))
                tokens[masked_token_id] = synonym_token
                all_masked_tokens.append(masked_token)
                all_synonsyms_tokens.append(synonym_token)

            rewrite_content = ''.join(tokens)

            results.append({"id": id, "domain": domain, "content": scontent, "rewrite_content": rewrite_content, "masked_words": all_masked_tokens, "replaced_words": all_synonsyms_tokens})

        return results

    def predict_one_mask(self, data, **kwargs):
        '''
        data: [{"id":int,
                "content":string,
                "brand":string,
                "series":string,
                "spec":string,
                "spec_name":string,
                "domain":string
                }, ...]
        output: [{
                "id":int,
                "rewrite_content":string,
                "masked_words": list,
                "replaced_words": list
                }, ...]
        '''
        """支持batch，一条数据，根据长度按比例设置需要mask的tokens数目，然后每次只mask一个token，重复此操作，直到mask固定数目的tokens，最后调用同义词预测模型逐一进行token替换"""
        if data is None or len(data) == 0:
            return []

        data_results = []

        num_mask_batch_syn_model_data = []  # 存放num_mask*batch_size条数据
        num_mask_batch_data_len = []  # 记录每一条data的num_mask

        batch_ids = []
        batch_domains = []
        batch_scontent = []  # 原始data里的content

        batch_tokens = []
        batch_masked_token_ids = []
        # batch_masked_tokens = []

        for idx, item in enumerate(data):
            id = item["id"]
            domain = item["domain"]
            scontent = item["content"]
            if self.remove_colon:
                scontent=self.remove_phrase_before_colon(scontent)

            tokens = jieba.lcut(scontent)
            len_tokens = len(tokens)
            num_mask = max(1, min(6, int(len_tokens * 0.3)))  ## 最小1，最大6

            if len(tokens) < self.min_short_sen_length:
                data_results.append(
                    {"id": id, "domain": domain, "content": scontent, "rewrite_content": scontent, "masked_words": [],
                     "replaced_words": []})
                continue

            i = 0
            num_mask_item_syn_model_data = []
            have_masked_token_ids = []
            while i < num_mask:
                content, masked_tokens, masked_token_ids = self.mask_sequence_v1(copy.deepcopy(tokens), num_mask=1,
                                                                          have_masked_token_ids=have_masked_token_ids)
                if not masked_tokens:   # 如果没有找到符合条件的masked_token就跳出while循环
                    i = num_mask
                    continue

                have_masked_token_ids.extend(masked_token_ids)
                num_mask_item_syn_model_data.append({"context": content, "maskwords": masked_tokens})
                i += 1

            if not have_masked_token_ids:
                data_results.append(
                    {"id": id, "domain": domain, "content": scontent, "rewrite_content": scontent, "masked_words": [],
                     "replaced_words": []})
                continue

            num_mask_batch_data_len.append(len(num_mask_item_syn_model_data))
            num_mask_batch_syn_model_data.extend(num_mask_item_syn_model_data)

            batch_ids.append(id)
            batch_domains.append(domain)
            batch_scontent.append(scontent)
            batch_tokens.append(tokens)
            batch_masked_token_ids.append(have_masked_token_ids)

        num_mask_batch_results = self.synonyms_recom_model.predict({"data": num_mask_batch_syn_model_data, "topk": 1})

        j = 0
        sum_perv_results_size = 0  # 记录已经处理的num_mask_batch_results列表的size
        batch_size = len(batch_ids)  ## 注意：batch_size 不等于 num_mask_batch_syn_model_data列表的长度
        while j < batch_size:
            id = batch_ids[j]
            domain = batch_domains[j]
            scontent = batch_scontent[j]
            tokens = batch_tokens[j]
            masked_token_ids = batch_masked_token_ids[j]
            num_mask_item = num_mask_batch_data_len[j]  # 不同数据的num_mask
            assert (len(masked_token_ids) == num_mask_item)
            num_mask_item_results = num_mask_batch_results[sum_perv_results_size: sum_perv_results_size+num_mask_item]

            all_synonsyms_tokens = []
            all_masked_tokens = []
            for idx, results in enumerate(num_mask_item_results):
                result = results[0]
                synonym_token = result["candidates"][0]["word"]
                masked_token = result["maskword"]
                masked_token_id = masked_token_ids[idx]
                tokens[masked_token_id] = synonym_token
                all_synonsyms_tokens.append(synonym_token)
                all_masked_tokens.append(masked_token)

            rewrite_content = ''.join(tokens)
            data_results.append({"id": id, "domain": domain, "content": scontent, "rewrite_content": rewrite_content,
                                 "masked_words": all_masked_tokens, "replaced_words": all_synonsyms_tokens})

            sum_perv_results_size += num_mask_item
            j += 1

        return data_results

    def predict_all_mask(self, data, **kwargs):
        '''
        data: [{"id":int,
                "content":string,
                "brand":string,
                "series":string,
                "spec":string,
                "spec_name":string,
                "domain":string
                }, ...]
        output: [{
                "id":int,
                "rewrite_content":string,
                "masked_words": list,
                "replaced_words": list
                }, ...]
        '''
        """支持batch，一条数据，根据长度按比例同时mask一定数目的tokens，然后调用同义词预测模型同时进行tokens替换"""
        if data is None or len(data) == 0:
            return []

        # num_mask = kwargs.get("num_mask", 5)
        data_results = []

        batch_masked_tokens = []
        batch_masked_token_ids = []
        batch_tokens = []
        batch_ids = []
        batch_domains = []
        batch_scontent = []  # 原始data里的content

        batch_syn_model_data = []

        for idx, item in enumerate(data):
            id = item["id"]
            domain = item["domain"]
            scontent = item["content"]
            if self.remove_colon:
                scontent=self.remove_phrase_before_colon(scontent)
            tokens = jieba.lcut(scontent)
            len_tokens = len(tokens)
            num_mask = max(1, min(6, int(len_tokens * 0.3)))  ## 最小1，最大6

            if len(tokens) < self.min_short_sen_length:
                data_results.append({"id": id, "domain": domain, "content": scontent, "rewrite_content": scontent, "masked_words": [], "replaced_words": []})
                continue

            content, masked_tokens, masked_token_ids = self.mask_sequence(copy.deepcopy(tokens), num_mask=num_mask,
                                                                          have_masked_token_ids=[])

            if not masked_tokens:
                data_results.append(
                    {"id": id, "domain": domain, "content": scontent, "rewrite_content": scontent, "masked_words": [],
                     "replaced_words": []})
                continue

            batch_ids.append(id)
            batch_domains.append(domain)
            batch_scontent.append(scontent)

            batch_tokens.append(tokens)

            batch_masked_tokens.append(masked_tokens)
            batch_masked_token_ids.append(masked_token_ids)
            batch_syn_model_data.append({"context": content, "maskwords": masked_tokens})

        batch_results = self.synonyms_recom_model.predict({"data": batch_syn_model_data, "topk": 1})

        for idx, results in enumerate(batch_results):
            id = batch_ids[idx]
            domain = batch_domains[idx]
            scontent = batch_scontent[idx]
            masked_token_ids = batch_masked_token_ids[idx]
            tokens = batch_tokens[idx]
            masked_tokens = batch_masked_tokens[idx]

            all_synonsyms_tokens = []
            for i, result in enumerate(results):
                synonym_token = result["candidates"][0]["word"]
                # masked_token = result["maskword"]
                masked_token_id = masked_token_ids[i]
                tokens[masked_token_id] = synonym_token
                all_synonsyms_tokens.append(synonym_token)

            rewrite_content = ''.join(tokens)
            data_results.append({"id": id, "domain": domain, "content": scontent, "rewrite_content": rewrite_content,
                            "masked_words": masked_tokens, "replaced_words": all_synonsyms_tokens})

        return data_results

    def predict(self, data, **kwargs):
        if self.all_mask:
            return self.predict_all_mask(data,**kwargs)

        return self.predict_one_mask(data,**kwargs)

    def remove_phrase_before_colon(self, txt):
        main = '油耗|动力|操控|舒适性|空间|外观|内饰|性价比|空调|加速|设计|行驶|底盘及悬架|座椅|乘坐空间|车内空间|悬架减震|做工车漆|高速路段|噪音控制|标准|选装|能耗|方向盘|刹车|期待的配置'.split(
            '|')
        suffix = '方面|设计|舒适性|平顺性|操控感|表现|配置'.split('|')
        pattern = r'\b(({})({})?[ \t]*)[:：]'.format('|'.join(main), '|'.join(suffix))
        return re.sub(pattern, ' ', txt)


class CarRewriteBaseKeywordsNewProcess(SimplexBaseModel):
    """基于277个新标签集合训练得到的改写模型欧拉服务"""
    def __init__(self, *args, **kwargs):
        super(CarRewriteBaseKeywordsNewProcess, self).__init__(*args, **kwargs)

        # self.multi_labels_cls_model = SimplexClient('BertCarMultiLabelsExtractTopKForRewrite', url="https://alpha-model-serving.aidigger.com/api/v1/car-multi-labels-extract/predict")  # 获取多标签
        self.multi_labels_cls_model = SimplexClient('CarMultiLabelsExtract', url='https://model-serving.aidigger.com/api/v1/car-multi-labels-extract/predict')  # 获取多标签
        self.min_short_sen_length = kwargs.get("min_short_sen_length", 1)  # 最小序列长度
        self.max_sen_length = kwargs.get("max_sen_length", 100)  # 最大序列长度，超过进行分句
        self.timeout = kwargs.get("timeout", 30)
        self.vocab_file = self.download(kwargs["vocab_path"])
        self.keywords_file = self.download(kwargs["keywords_path"])
        self.idf_json_file = self.download(kwargs["idf_json_path"])
        self.high_freq_token_file = self.download(kwargs["high_freq_token_path"])
        self.token2id = {line.strip(): idx for idx, line in enumerate(open(self.vocab_file, 'r').readlines())}
        self.eos_id = self.token2id['<eos>']
        self.sep_id = self.token2id['<sep>']
        self.high_freq_tokens = [line.strip() for idx, line in enumerate(open(self.high_freq_token_file, 'r').readlines())]
        self.keywords = [line.strip() for idx, line in enumerate(open(self.keywords_file, 'r').readlines())]
        self.idf_scores_dict = json.load(open(self.idf_json_file, 'r'))
        self.num_unit_split_pattern = re.compile(r'[\d\.]+|[零一二两三四五六七八九十百千万]*')

    def tokenize(self, tokens):
        token_ids = [self.token2id[token] if token in self.token2id else self.token2id['<unk>'] for token in tokens]
        return token_ids

    def cut_sens(self, content):
        """按特殊标点符号，将长句进行切割，返回切割后的短句列表"""
        # return re.findall(u'.*?[！|。|...|？|?|!|；|~|～|。]+', content)
        sens_li = re.findall(u'.*?[！|,|，|。|？|?|!|；|~|～|。|：：|#]+', content + '#')
        return sens_li[:-1] + [sens_li[-1].rstrip('#')]

    def do_concat_short_sens(self, short_sens):
        concat_short_sens = []  # 将合并后字符级别的序列长度不超过max_sen_length*1.5的短句合并成一个长句

        new_sen = ''
        for short_sen in short_sens[:-1]:
            if len(new_sen) >= int(self.max_sen_length * 1.5):
                concat_short_sens.append(new_sen)
                new_sen = short_sen
            elif len(short_sen) >= (self.max_sen_length * 1.5):
                if new_sen:
                    concat_short_sens.append(new_sen)
                    concat_short_sens.append(short_sen)
                    new_sen = ''
                else:
                    concat_short_sens.append(short_sen)
            else:
                new_sen += short_sen

        if new_sen:
            if len(new_sen) >= int(self.max_sen_length * 1.5):
                concat_short_sens.append(new_sen)
                concat_short_sens.append(short_sens[-1])
            else:
                new_sen += short_sens[-1]
                concat_short_sens.append(new_sen)

        else:
            concat_short_sens.append(short_sens[-1])

        return concat_short_sens

    def extract_line_keywords(self, cut_tokens, keywords_li, word2id, num_unit_split_pattern, idf_scores_dict):
        """
        抽取一句话里的关键词
        cut_tokens: 切完词的tokens
        keywords_li: 关键词词表，包含一般的关键词，数字以及数字单位
        num_unit_split_pattern:  见self.get_num_unit_split_reg_pattern()
        """
        line_keywords = []
        for token in cut_tokens:
            if token in keywords_li:
                line_keywords.append(token)
            else:  # 不在关键词表里
                # 先判断是否是数字和单位
                (split_b, split_e) = num_unit_split_pattern.match(
                    token).span()  # 将数字和单位连在一起的keyword进行拆分，比如"12kg" 拆分成"12", "kg"
                if split_e != 0:
                    try:
                        float(token[:split_e])
                        for c in token[:split_e]:  # 把数字拆分，"12"->"1", "2"
                            if c in word2id:
                                line_keywords.append(c)
                    except:  # 可能是"零一二两三四五六七八九十百千万"
                        for c in token[:split_e]:
                            if c in word2id:
                                line_keywords.append(c)

                    if split_e < len(token):  # 还有单位或其他字符，并确保单位或其他字符在词表里才当作关键词
                        for c in token[split_e:]:
                            if c in word2id:
                                line_keywords.append(c)
                else:  # 再判断是否在词表里
                    if token not in word2id:  # 不在词表，拆成字，如果字在词表里就当作关键词，否则不作为关键词
                        for c in token:
                            if c in word2id:
                                line_keywords.append(c)

        if len(line_keywords) < int(len(cut_tokens) / 2):  # 关键词太少，再根据tf-idf值抽取
            idf_scores = []  # 一句话里的token的tf基本相等，只算idf就可以，即保留一句话里的稀有词
            topk = int(len(cut_tokens) / 3)  # 取1/3的关键词，有可能会有重复
            for token in cut_tokens:
                if token in idf_scores_dict:
                    idf_scores.append(idf_scores_dict[token])
                else:
                    idf_scores.append(1e-8)

            tokens = numpy.array(cut_tokens)
            idf_sorting = numpy.argsort(idf_scores)[::-1]
            topk_keywords = list(tokens[idf_sorting[:topk]])
            for keyword in topk_keywords:
                if keyword in word2id and keyword not in line_keywords:  # 只要在词表里并没有出现在line_keywords就当作关键词
                    line_keywords.append(keyword)

        # if len(line_keywords) > int(len(cut_tokens) * 7 / 12):  # 控制line_keywords的个数
        #     line_keywords = line_keywords[:int(len(cut_tokens) * 7 / 12)]

        return line_keywords

    def process_line(self, cut_tokens, high_freq_words, num_unit_split_pattern):
        """
        重新处理分词后的一句话的tokens，比如特殊处理数字和单位
        cut_tokens: 切完词的tokens
        high_freq_words: 高频词列表
        num_unit_split_pattern:  见self.get_num_unit_split_reg_pattern()
        """
        new_line_tokens = []
        for token in cut_tokens:
            # 先判断是否是数字和单位
            (split_b, split_e) = num_unit_split_pattern.match(
                token).span()  # 将数字和单位连在一起的keyword进行拆分，比如"12kg" 拆分成"12", "kg"
            if split_e != 0:
                try:
                    float(token[:split_e])
                    for c in token[:split_e]:  # 把数字拆分，"12"->"1", "2"
                        new_line_tokens.append(c)
                except:  # 可能是"零一二两三四五六七八九十百千万"
                    for c in token[:split_e]:
                        new_line_tokens.append(c)

                if split_e < len(token):  # 还有单位或其他字符
                    for c in token[split_e:]:
                        new_line_tokens.append(c)
            else:  # 再判断是否是低频词
                if token not in high_freq_words:
                    for c in token:
                        new_line_tokens.append(c)
                else:
                    new_line_tokens.append(token)

        return new_line_tokens

    def get_tf_results(self, tokens_li, lengths, max_len):
        # tokens = tokens.split()
        # tf_data = {
        #     "signature_name": "serving_default",
        #     "instances": [{
        #         "tokens": tokens,
        #         "length": len(tokens)
        #     }]
        # }
        tf_data = {
            "signature_name": "serving_default",
            "instances": [{"tokens": tokens.split() + ['<eos>'] * (max_len - lengths[idx]), "length": lengths[idx]} for
                          idx, tokens in enumerate(tokens_li)]
        }

        ret = self._call_local_tf_service(tf_data, timeout=self.timeout).json()

        # try:
        #     ret = self._call_local_tf_service(tf_data, timeout=self.timeout).json()
        # except Exception as e:
        #     logger.warning("# Get tf results with error: {0}".format(e))
        #     ret = None

        # if ret is None:
        #     return ['']*len(lengths)

        rewrite_results = [''.join(result['tokens'][0][:result['length'][0] - 1]) for result in ret['predictions']]

        # ret_tokens = ret['predictions'][0]['tokens'][0]
        # ret_tokens_len = ret['predictions'][0]['length'][0] - 1

        # return ret_tokens[:ret_tokens_len]
        return rewrite_results

    def predict(self, data, **kwargs):
        '''
        data: [{"id":int,
                "content":string,
                "brand":string,
                "series":string,
                "spec":string,
                "spec_name":string,
                "domain":string
                }, ...]
        output: [{
                "id":int,
                "rewrite_content":string
                }, ...]
        '''
        if data is None or len(data) == 0:
            return []

        results = []

        data_ids = []
        data_pieces_lengths = []
        data_pieces_contents = []
        data_pieces_keywords = []
        data_domains = []
        data_contents = []

        for idx, item in enumerate(data):
            id = item['id']
            comment = item['content'].strip()
            if not comment:
                continue

            tokens = jieba.lcut(comment)
            if len(tokens) < self.min_short_sen_length:
                continue

            domain = '#' + item['domain'].strip() + '#'
            # domain_token_id = self.token2id[domain]

            if len(tokens) > self.max_sen_length:
                short_sens = self.cut_sens(comment)
                concat_short_sens = self.do_concat_short_sens(short_sens)
                pieces_tokens = [jieba.lcut(short_sen) for short_sen in concat_short_sens]

            else:
                concat_short_sens = [comment]
                pieces_tokens = [tokens]

            ## 保证tokens长度不超过max_sen_length
            # pieces_keywords = [self.process_line(piece_tokens[:self.max_sen_length], self.high_freq_tokens, self.num_unit_split_pattern) for piece_tokens in pieces_tokens]
            pieces_keywords = [self.extract_line_keywords(piece_tokens[:self.max_sen_length], self.keywords, self.token2id, self.num_unit_split_pattern, self.idf_scores_dict) for piece_tokens in pieces_tokens]

            # pieces_keywords_ids = [self.tokenize(piece_keywords) for piece_keywords in pieces_keywords]

            data_ids.append(id)
            data_domains.append(domain)
            data_contents.append(comment)
            data_pieces_contents.extend(concat_short_sens)
            data_pieces_lengths.append(len(pieces_keywords))
            data_pieces_keywords.append(pieces_keywords)

        # num_data = len(data_ids)
        # assert(len(data_domain_ids)==num_data)
        # assert (len(data_contents) == num_data)
        # assert (len(data_pieces_lengths) == num_data)
        # assert (len(data_pieces_keywords_ids) == num_data)

        all_multi_tags_results = self.multi_labels_cls_model.predict([{"content": content} for content in data_pieces_contents])
        all_multi_tags = [['__'+t_result['tag']+'__' for t_result in pred_result['tags']] for pred_result in all_multi_tags_results]

        ## 构建tf_serving数据格式
        data_tokens_li = []
        data_tokens_length = []

        num_data = len(data_ids)
        prev_sum = 0

        for i in range(num_data):
            domain = data_domains[i]
            pieces_keywords = data_pieces_keywords[i]
            piece_len = data_pieces_lengths[i]
            piece_multi_tags = [multi_tags for multi_tags in all_multi_tags[prev_sum:prev_sum + piece_len]]

            pieces_tokens_li = [domain + ' <sep> ' + ' '.join(piece_multi_tags[j]) + ' <sep> ' + ' '.join(pieces_keywords[j]) for j in range(piece_len)]

            data_tokens_li.extend(pieces_tokens_li)
            data_tokens_length.extend([len(piece_tokens.strip().split()) for piece_tokens in pieces_tokens_li])
            prev_sum += piece_len

        max_len = max(data_tokens_length)
        assert (len(data_tokens_li) == sum([piece_length for piece_length in data_pieces_lengths]))

        data_rewrite_results = self.get_tf_results(data_tokens_li, data_tokens_length, max_len)

        prev_sum = 0
        for i in range(num_data):
            rewrite_str = ''
            id = data_ids[i]
            domain = data_domains[i].strip('#')
            content = data_contents[i]
            piece_len = data_pieces_lengths[i]
            rewrite_results = data_rewrite_results[prev_sum: prev_sum + piece_len]
            comments_pieces = data_pieces_contents[prev_sum: prev_sum + piece_len]
            keywords = data_tokens_li[prev_sum: prev_sum + piece_len]  # 关键词
            prev_sum += piece_len
            rewrite_results = [tmp_rewrite_str if '<unk>' not in tmp_rewrite_str else comments_pieces[idx] for
                               idx, tmp_rewrite_str in enumerate(rewrite_results)]
            rewrite_str += ' '.join(rewrite_results)
            results.append({'id': id, 'domain': domain, 'content': content, 'rewrite_content': rewrite_str, 'keywords': keywords})

        return results

    def predict_v1(self, data, **kwargs):
        '''
        data: [{"id":int,
                "content":string,
                "brand":string,
                "series":string,
                "spec":string,
                "spec_name":string,
                "domain":string
                }, ...]
        output: [{
                "id":int,
                "rewrite_content":string
                }, ...]
        '''
        if data is None or len(data) == 0:
            return []

        results = []

        data_ids = []
        data_domains = []
        data_contents = []
        data_keywords = []

        for idx, item in enumerate(data):
            id = item['id']
            comment = item['content'].strip()
            if not comment:
                continue

            tokens = jieba.lcut(comment)
            if len(tokens) < self.min_short_sen_length:
                continue

            domain = '#' + item['domain'].strip() + '#'
            if len(tokens) > self.max_sen_length:
                tokens = tokens[:self.max_sen_length]

            data_contents.append(comment)

            # data_keywords.append(self.process_line(tokens, self.high_freq_tokens, self.num_unit_split_pattern))
            data_keywords.append(self.extract_line_keywords(tokens, self.keywords, self.token2id, self.num_unit_split_pattern, self.idf_scores_dict))
            data_domains.append(domain)
            data_ids.append(id)

        all_multi_tags_results = self.multi_labels_cls_model.predict([{"content": content} for content in data_contents])
        all_multi_tags = [['__' + t_result['tag'] + '__' for t_result in pred_result['tags']] for pred_result in all_multi_tags_results]

        num_data = len(data_domains)

        data_tokens_li = []
        data_tokens_length = []
        for i in range(num_data):
            domain = data_domains[i]
            keywords = data_keywords[i]
            multi_tags = all_multi_tags[i]

            tokens_li = domain + ' <sep> ' + ' '.join(multi_tags) + ' <sep> ' + ' '.join(keywords)

            data_tokens_li.append(tokens_li)
            data_tokens_length.append(len(tokens_li.strip().split()))

        max_len = max(data_tokens_length)

        data_rewrite_results = self.get_tf_results(data_tokens_li, data_tokens_length, max_len)

        for i in range(num_data):
            id = data_ids[i]
            domain = data_domains[i].strip('#')
            content = data_contents[i]
            rewrite_str = data_rewrite_results[i]
            results.append({'id': id, 'domain': domain, 'content': content, 'rewrite_content': rewrite_str, 'keywords': data_tokens_li[i]})

            # results.append({'id': id, 'domain': domain, 'content': content, 'rewrite_content': rewrite_str, 'jieba_cut': ' '.join(jieba.lcut(content)), 'keywords': data_tokens_li[i]})

        return results


class CarRewriteBaseKeywords(SimplexBaseModel):
    def __init__(self, *args, **kwargs):
        super(CarRewriteBaseKeywords, self).__init__(*args, **kwargs)

        self.senti_label_cls_model = SimplexClient('BertCarSentiCls')  # 获取情感标签模型
        self.timeout = kwargs.get("timeout", 60)
        self.total_comments = [line.strip() for line in open(self.download(kwargs.get("total_comments_filepath"))).readlines()]
        self.vocab = set([line.strip() for line in open(self.download(kwargs.get("vocab_filepath"))).readlines()])
        self.keywords_table = set([line.strip() for line in open(self.download(kwargs.get("keywords_filepath"))).readlines()])
        self.car_info = set([line.strip() for line in open(self.download(kwargs.get("car_info_filepath"))).readlines()])
        self.stop_words = set([line.strip() for line in open(self.download(kwargs.get("stop_words_filepath"))).readlines()])
        self.vectorizer = TfidfVectorizer(stop_words=self.stop_words)
        self.idf_features = self.vectorizer.fit(self.total_comments)
        self.feature_words = numpy.array(self.vectorizer.get_feature_names())

    def tokenize(self, content):
        tokens = []
        for word in jieba.cut(content.strip()):
            word = word.strip()
            if word == '':
                continue
            (b, e) = digits.match(word).span()
            if e != 0 and word not in self.stop_words:
                for d in word[:e]:
                    tokens.append(d)
                tokens.append(word[e:])
            else:
                tokens.append(word)
        return " ".join(tokens)

    def get_tf_results(self, tokens_li, lengths, max_len):
        # tokens = tokens.split()
        # tf_data = {
        #     "signature_name": "serving_default",
        #     "instances": [{
        #         "tokens": tokens,
        #         "length": len(tokens)
        #     }]
        # }
        tf_data = {
            "signature_name": "serving_default",
            "instances": [{"tokens": tokens.split()+['eos']*(max_len-lengths[idx]), "length": lengths[idx]} for idx, tokens in enumerate(tokens_li)]
        }

        # url = 'https://car-rewrite-base-keywords-tf.aidigger.com/v1/models/car-rewrite-base-keywords:predict'

        ret = self._call_local_tf_service(tf_data, timeout=self.timeout).json()

        # try:
        #     ret = self._call_local_tf_service(tf_data, timeout=self.timeout).json()
        # except Exception as e:
        #     logger.warning("# Get tf results with error: {0}".format(e))
        #     ret = None

        # if ret is None:
        #     return ['']*len(lengths)

        rewrite_results = [''.join(result['tokens'][0][:result['length'][0]-1]) for result in ret['predictions']]

        # ret_tokens = ret['predictions'][0]['tokens'][0]
        # ret_tokens_len = ret['predictions'][0]['length'][0] - 1

        # return ret_tokens[:ret_tokens_len]
        return rewrite_results


    def get_comments_pieces(self, comment):
        if len(comment.strip()) == 0:
            return []

        if len(comment) < 55:
            return [comment]

        comment_pieces = []
        length = len(comment) // (len(comment) // 55 + 1)
        # pieces = re.split('[,，。?！!？]', comment)
        # pieces = re.findall(u'.*?[！|,|，|。|...|？|?|!|；|~|～|。|：：]+', comment)
        sens_li = re.findall(u'.*?[！|,|，|。|？|?|!|；|~|～|。|：：|#]+', comment + '#')
        pieces = sens_li[:-1] + [sens_li[-1].rstrip('#')]
        part = ''

        for piece in pieces:
            if len(part) >= length:
                # comment_pieces.append(part[:-1])
                comment_pieces.append(part)
                part = ''
            if piece.strip() == '': continue
            part += piece
            # part += '，'

        if len(part) > 1:
            # comment_pieces.append(part[:-1])
            comment_pieces.append(part)

        if not comment_pieces:
            return [comment]

        return comment_pieces


    def get_comment_keywords(self, comment):
        key_words = []
        length = len(comment.split())
        tt = self.idf_features.transform([comment]).toarray()
        cur_key = self.feature_words[numpy.argsort(tt).flatten()[::-1].tolist()][:length // 7 + 1]
        comment_split = re.split('[,，。?！!？]', comment)
        for sent in comment_split:
            flag = 0
            for word in sent.split():
                if word.strip() == '' or word not in self.vocab or word in self.stop_words:
                    continue
                if word in self.keywords_table or partten.match(word) or word in self.car_info or word in cur_key:
                    key_words.append(word)
                    flag = 1
            if flag == 0:
                for word in sent.split():
                    if word in self.vocab:
                        key_words.append(word)
        return key_words


    def predict(self, data, **kwargs):
        '''
        data: [{"id":int,
                "content":string,
                "brand":string,
                "series":string,
                "spec":string,
                "spec_name":string,
                "domain":string
                }, ...]
        output: [{
                "id":int,
                "rewrite_content":string
                }, ...]
        '''
        if data is None or len(data) == 0:
            return []

        results = []
        ids = []
        data_pieces_lengths = []
        data_pieces = []
        data_keywords_li = []
        data_domains = []
        data_contents = []
        data_tokens_li = []
        batch_size = len(data)

        for idx, item in enumerate(data):
            id = item['id']
            comment = item['content'].strip()
            if not comment:
                continue

            domain = item['domain'].strip()
            comments_pieces = self.get_comments_pieces(comment)
            if not comments_pieces:
                continue

            ids.append(id)
            data_domains.append(domain)
            data_contents.append(comment)
            data_pieces_lengths.append(len(comments_pieces))
            data_pieces.extend(comments_pieces)

            keywords_li = [self.get_comment_keywords(self.tokenize(piece)) for piece in comments_pieces]
            data_keywords_li.append(keywords_li)

        ## get senti labels
        all_senti_labels = self.senti_label_cls_model.predict([{'content':piece} for piece in data_pieces])
        prev_sum = 0
        data_tokens_length = []
        for i in range(batch_size):
            data_len = data_pieces_lengths[i]
            data_senti_labels = [senti_label['label'] for senti_label in all_senti_labels[prev_sum:prev_sum+data_len]]
            prev_sum += data_len
            keywords_li = data_keywords_li[i]
            domain = data_domains[i]

            tokens_li = [senti_label + ' ' + ' '.join(keywords_li[idx]) + ' ' + '<' + domain + '>' for idx, senti_label in enumerate(data_senti_labels)]
            data_tokens_li.extend(tokens_li)
            lengths = [len(tokens.strip().split()) for tokens in tokens_li]
            data_tokens_length.extend(lengths)

        max_len = max(data_tokens_length)

        data_rewrite_results = self.get_tf_results(data_tokens_li, data_tokens_length, max_len)
        prev_sum = 0
        for i in range(batch_size):
            rewrite_str = ''
            id = ids[i]
            domain = data_domains[i]
            content = data_contents[i]
            data_len = data_pieces_lengths[i]
            rewrite_results = data_rewrite_results[prev_sum: prev_sum+data_len]
            comments_pieces = data_pieces[prev_sum: prev_sum+data_len]
            prev_sum += data_len
            rewrite_results = [tmp_rewrite_str if '<unk>' not in tmp_rewrite_str else comments_pieces[idx]  for idx, tmp_rewrite_str in enumerate(rewrite_results)]
            rewrite_str += ' '.join(rewrite_results)
            results.append({'id': id, 'domain': domain, 'content': content, 'rewrite_content': rewrite_str})

        # for idx, item in enumerate(data):
        #     id = item['id']
        #     comment = item['content'].strip()
        #     if not comment:
        #         continue

        #     domain = item['domain'].strip()
        #     comments_pieces = self.get_comments_pieces(comment)
        #     if not comments_pieces:
        #         continue

        #     rewrite_str = ''

        #     ret = self.senti_label_cls_model.predict([{'content':piece} for piece in comments_pieces if len(piece) > 0])
        #     # ret = ['<POS>' for piece in comments_pieces if len(piece) > 0]
        #     if not ret:
        #         continue
        #     keywords_li = [self.get_comment_keywords(self.tokenize(piece)) for piece in comments_pieces if len(piece) > 0]
        #     tokens_li = [senti_label['label'] + ' ' + ' '.join(keywords_li[idx]) + ' ' + '<' + domain + '>' for idx, senti_label in enumerate(ret)]

        #     lengths = [len(tokens.strip().split()) for tokens in tokens_li]
        #     max_len = max(lengths)

        #     rewrite_results = self.get_tf_results(tokens_li, lengths, max_len)
        #     # assert(len(rewrite_results) == len(comments_pieces))
        #     rewrite_results = [tmp_rewrite_str if '<unk>' not in tmp_rewrite_str else comments_pieces[idx]  for idx, tmp_rewrite_str in enumerate(rewrite_results)]
        #     rewrite_str += ' '.join(rewrite_results)
        #     results.append({'id': id, 'rewrite_content': rewrite_str})

        #     # for piece in comments_pieces:
        #     #     senti_label = self.senti_label_cls_model.predict([{'content': piece}])[0]['label']  # 获取短语的senti label
        #     #     key_words = self.get_comment_keywords(self.tokenize(piece))
        #     #     tokens =  senti_label + ' ' + ' '.join(key_words) + ' ' + '<' + domain + '>'
        #     #     rewrite_tokens = self.get_tf_results(tokens)
        #     #     rewrite_str += ''.join(rewrite_tokens)
        #     #     rewrite_str += '，'
        #     # results.append({'id': id, 'rewrite_content': rewrite_str[:-1]})

        return results