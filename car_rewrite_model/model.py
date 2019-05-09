import logging
import random

import re
import jieba
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer

from simplex_base_model import SimplexBaseModel
from simplex_sdk import SimplexClient

logger = logging.getLogger(__name__)

partten = re.compile(r'[.a-zA-Z|零|一|二|三|四|五|六|七|八|九|十|百|千|万|号|升|尺|英尺|英寸|寸|米|牛米|吨|千米|千克|克|分|转|马|码|公里|代|里|英里|厘米|年|月|天|匹|点|几|倍|挡\d]+')
digits = re.compile(r'[\d\.]+|[零一二两三四五六七八九十百千万]*')


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
        pieces = re.findall(u'.*?[！|,|，|。|...|？|?|!|；|~|～|。|：：]+', comment)
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