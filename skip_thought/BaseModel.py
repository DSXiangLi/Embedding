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

    def _encode(self, *wargs):
        raise NotImplementedError()

    def _decode(self, *wargs):
        raise NotImplementedError()

    def compute_loss(self, *wargs):
        raise NotImplementedError()

    def predict(self, *wargs):
        raise NotImplementedError()

