# -*- coding=utf-8 -*-

class Seq2SeqModel(object):
    def __init__(self, params):
        self.params = params
        self.embedding = None
        self.params_check()
        self.init()

    def params_check(self):
        raise NotImplementedError()

    def init(self):
        raise NotImplementedError()

    def build_model(self, features, labels, mode):
        raise NotImplementedError()

    def _encode(self, **kwargs):
        raise NotImplementedError()

    def _decode(self, **kwargs):
        raise NotImplementedError()

    def compute_loss(self, **kwargs):
        raise NotImplementedError()

