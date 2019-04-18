# coding: utf-8

import unittest
from unittest.mock import MagicMock, patch
from car_rewrite_model.model import CarRewriteBaseKeywords

class TestDemo(unittest.TestCase):

    def test_model(self):
        CarRewriteBaseKeywords.get_tf_results = MagicMock(return_value = ['改', '写', '结', '果'])
        total_comments_filepath = 'oss://modelzoo/dev/ludezheng/total_comments.txt'
        vocab_filepath = 'oss://modelzoo/dev/ludezheng/vocab_jieba.txt'
        keywords_filepath = 'oss://modelzoo/dev/ludezheng/keywords_table.txt'
        car_info_filepath = 'oss://modelzoo/dev/ludezheng/car_info.txt'
        stop_words_filepath = 'oss://modelzoo/dev/ludezheng/sotp_words.txt'
        m = CarRewriteBaseKeywords(total_comments_filepath=total_comments_filepath, vocab_filepath=vocab_filepath, keywords_filepath=keywords_filepath, car_info_filepath=car_info_filepath, stop_words_filepath=stop_words_filepath)
        test_data = [{'id': 1, 'content': '驾驶性能超好，还真别说，会开车的，开着瑞虎7操控感觉肯定也不错，外观造型漂亮，设计不会老土，最后就是虽然加速动力欠缺，但是确实很省油', 'domain': '最满意'}]
        ret = m.predict(test_data)
        # self.assertEqual(ret, 'hello')
