# coding: utf-8

import unittest
from car_rewrite_model.model import CarRewrite_base_keywords

class TestDemo(unittest.TestCase):

    def test_model(self):
        m = CarRewrite_base_keywords(msg='hello')
        test_data = [{'id': 1, 'content': '驾驶性能超好，还真别说，会开车的，开着瑞虎7操控感觉肯定也不错，外观造型漂亮，设计不会老土，最后就是虽然加速动力欠缺，但是确实很省油', 'domain': '最满意'}]
        ret = m.predict(test_data)
        # self.assertEqual(ret, 'hello')
