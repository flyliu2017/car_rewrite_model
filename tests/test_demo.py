# coding: utf-8

import unittest
from unittest.mock import MagicMock, patch
from car_rewrite_model.model import CarRewriteBaseKeywords, CarRewriteSynonymsReplace


class TestDemo(unittest.TestCase):

    def test_car_rewrite_base_keywords(self):
        CarRewriteBaseKeywords.get_tf_results = MagicMock(return_value=['改', '写', '结', '果'])
        total_comments_filepath = 'oss://modelzoo/dev/ludezheng/total_comments.txt'
        vocab_filepath = 'oss://modelzoo/dev/ludezheng/vocab_jieba.txt'
        keywords_filepath = 'oss://modelzoo/dev/ludezheng/keywords_table.txt'
        car_info_filepath = 'oss://modelzoo/dev/ludezheng/car_info.txt'
        stop_words_filepath = 'oss://modelzoo/dev/ludezheng/stop_words.txt'
        m = CarRewriteBaseKeywords(total_comments_filepath=total_comments_filepath, vocab_filepath=vocab_filepath,
                                   keywords_filepath=keywords_filepath, car_info_filepath=car_info_filepath,
                                   stop_words_filepath=stop_words_filepath)
        test_data = [
            {'id': 1, 'content': '驾驶性能超好，还真别说，会开车的，开着瑞虎7操控感觉肯定也不错，外观造型漂亮，设计不会老土，最后就是虽然加速动力欠缺，但是确实很省油', 'domain': '最满意'}]
        ret = m.predict(test_data)
        # self.assertEqual(ret, 'hello')

    def test_car_rewirte_synonyms_replace(self):
        pos_model = '/data/share/liuchang/car_articles/pos.model'
        # pos_model = 'oss://modelzoo/dev/pos_model_for_car_rewrite/pos.model'
        model = CarRewriteSynonymsReplace(pos_model_path=pos_model,all_mask=True)
        l = [{'maskword': '性能',
              'candidates': [{'word': '<unk>', 'confidence': 1},
                             {'word': '<unk>', 'confidence': 1},
                             {'word': '<unk>', 'confidence': 1},
                             {'word': '<unk>', 'confidence': 1},
                             {'word': '<unk>', 'confidence': 1}]},
             {'maskword': '外观',
              'candidates': [{'word': '<unk>', 'confidence': 1},
                             {'word': '<unk>', 'confidence': 1},
                             {'word': '<unk>', 'confidence': 1},
                             {'word': '<unk>', 'confidence': 1},
                             {'word': '<unk>', 'confidence': 1}]}]
        data_size = 3
        return_list = [l] * data_size
        model.synonyms_recom_model.predict = MagicMock(return_value=return_list)

        test_data = [
            {'id': 1,
             'content': '驾驶性能超好，还真别说，会开车的，开着瑞虎7操控感觉肯定也不错，'
                        '外观造型漂亮，设计不会老土，最后就是虽然加速动力欠缺，但是确实很省油',
             'domain': '最满意'}] * data_size
        # input = {'data': test_data, 'topk': 5}

        result = model.predict(test_data)
        assert len(result) == 3
        print(result)
