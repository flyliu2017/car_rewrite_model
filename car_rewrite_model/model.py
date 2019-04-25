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
        self.timeout = kwargs.get("timeout", 20)
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

        try:
            ret = self._call_local_tf_service(tf_data, timeout=self.timeout).json()
        except Exception as e:
            logger.warning("# Get tf results with error: {0}".format(e))
            ret = None

        if ret is None:
            return []
            
        rewrite_results = [''.join(result['tokens'][0][:result['length'][0]-1]) for result in ret['predictions']]

        # ret_tokens = ret['predictions'][0]['tokens'][0]
        # ret_tokens_len = ret['predictions'][0]['length'][0] - 1

        # return ret_tokens[:ret_tokens_len]
        return rewrite_results


    def get_comments_pieces(self, comment):
        if len(comment) < 55:
            return [comment]

        comment_pieces = []
        length = len(comment) // (len(comment) // 55 + 1)
        pieces = re.split('[,，。?！!？]', comment)
        part = ''

        for piece in pieces:
            if len(part) >= length:
                comment_pieces.append(part[:-1])
                part = ''
            if piece == '': continue
            part += piece
            part += '，'

        if len(part) > 1:
            comment_pieces.append(part[:-1])

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
        for idx, item in enumerate(data):
            id = item['id']
            comment = item['content'].strip()
            domain = item['domain'].strip()
            comments_pieces = self.get_comments_pieces(comment)
            rewrite_str = ''

            ret = self.senti_label_cls_model.predict([{'content':piece} for piece in comments_pieces])
            keywords_li = [self.get_comment_keywords(self.tokenize(piece)) for piece in comments_pieces]
            tokens_li = [senti_label['label'] + ' ' + ' '.join(keywords_li[idx]) + ' ' + '<' + domain + '>' for idx, senti_label in enumerate(ret)]
            lengths = [len(tokens.strip().split()) for tokens in tokens_li]
            max_len = max(lengths)
            
            rewrite_results = self.get_tf_results(tokens_li, lengths, max_len)
            rewrite_str += ' '.join(rewrite_results)
            results.append({'id': id, 'rewrite_content': rewrite_str})

            # for piece in comments_pieces:
            #     senti_label = self.senti_label_cls_model.predict([{'content': piece}])[0]['label']  # 获取短语的senti label
            #     key_words = self.get_comment_keywords(self.tokenize(piece))
            #     tokens =  senti_label + ' ' + ' '.join(key_words) + ' ' + '<' + domain + '>'
            #     rewrite_tokens = self.get_tf_results(tokens)
            #     rewrite_str += ''.join(rewrite_tokens)
            #     rewrite_str += '，'
            # results.append({'id': id, 'rewrite_content': rewrite_str[:-1]})

        return results