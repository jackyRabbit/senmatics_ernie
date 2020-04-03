#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import os
import numpy as np
import types
import gzip
import logging
import re
import six
import collections
import tokenization
import paddle
import paddle.fluid as fluid
from tokenization import FullTokenizer
from batching import prepare_batch_data,pad_batch_data


class ErnieDataReader(object):
    def __init__(self,
                 vocab_path,
                 data,
                 batch_size=4096,
                 max_seq_len=512,
                 random_seed=1):

        self.vocab = self.load_vocab(vocab_path)
        #  vocab = load_vocab(vocab_path)
        self.vocab_path = vocab_path
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.max_seq_len = max_seq_len
        self.pad_id = self.vocab["[PAD]"]
        self.mask_id = self.vocab["[MASK]"]
        self.pos = 0
        self.data = data


    def parse_line(self, line, max_seq_len=512):
        """ parse one line to token_ids, sentence_ids, pos_ids, label
        """

        line = line.strip().split("，")
        assert len(line) == 3, \
            "One sample must have %d fields!" % 3

        text_left, text_right, masklabel = line
        tokenizer = FullTokenizer(self.vocab_path)
        # tokenizer = FullTokenizer(vocab_path)
        text_left = tokenizer.tokenize(text_left)
        masklabel = tokenizer.tokenize(masklabel)
        masklabel_ = len(masklabel)*["[MASK]"]
        text_right = tokenizer.tokenize(text_right)
        all_tokens = text_left + masklabel_ + text_right
        token_ids = tokenizer.convert_tokens_to_ids(all_tokens)
        sent_ids = [0]*len(all_tokens)
        pos_ids = [i for i in range(len(all_tokens))]
        input_mask = [1.0]*len(all_tokens)
        # 这儿还差一个mask_pos
        mask_pos = []
        for idx,mask in enumerate(token_ids):
            if mask ==self.mask_id:
                mask_pos.append(idx)
        # 添加一个mask_label
        mask_label = list(tokenizer.convert_tokens_to_ids(masklabel))
        assert len(token_ids) == len(sent_ids) == len(pos_ids)==len(input_mask) , "[Must be true]len(token_ids) == len(sent_ids) == len(pos_ids) == len(seg_labels)"
        if len(token_ids) > max_seq_len:
            return None
        return [token_ids, sent_ids, pos_ids, input_mask,mask_pos,mask_label]

    def parse_batch(self,batch):
        token_ids_batch, sent_ids_batch, pos_ids_batch, input_mask_batch,mask_pos_batch,mask_label_batch= [],[],[],[],[],[]
        for line in batch:
            token_ids, sent_ids, pos_ids, input_mask,mask_pos,mask_label = self.parse_line(line)
            token_ids_batch.append(token_ids)
            sent_ids_batch.append(sent_ids)
            pos_ids_batch.append(pos_ids)
            input_mask_batch.append(input_mask)
            mask_pos_batch.append(mask_pos)
            mask_label_batch.append(mask_label)
        return token_ids_batch, sent_ids_batch, pos_ids_batch, input_mask_batch,mask_pos_batch,mask_label_batch
    def next_predict_batch(self,batch_size):
        if self.pos>=len(self.data):
            self.pos=0
            return None
        else:
            batch = self.data[self.pos: self.pos + batch_size]
            self.pos += batch_size
            token_ids_batch, sent_ids_batch, pos_ids_batch, input_mask_batch,mask_pos_batch,mask_label_batch = self.parse_batch(batch)
            # 用padding处理一下
            token_ids_batch ,input_mask_batch= pad_batch_data(token_ids_batch,pad_idx=self.pad_id,return_input_mask=True)
            sent_ids_batch = pad_batch_data(sent_ids_batch,pad_idx=self.pad_id)
            pos_ids_batch = pad_batch_data(pos_ids_batch, pad_idx=self.pad_id)
            # input_mask_batch = pad_batch_data(input_mask_batch, pad_idx=self.pad_id)
            mask_pos_batch = np.array(mask_pos_batch).astype("int64").reshape([-1, 1])
            mask_label_batch = np.array(mask_label_batch).astype("int64").reshape([-1, 1])
            # mask_pos_batch并没有进行padding,而在ernie中似乎进行了padding
            return token_ids_batch, sent_ids_batch, pos_ids_batch, input_mask_batch,mask_pos_batch,mask_label_batch
    def convert_to_unicode(self, text):
        """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
        if six.PY3:
            if isinstance(text, str):
                return text
            elif isinstance(text, bytes):
                return text.decode("utf-8", "ignore")
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        elif six.PY2:
            if isinstance(text, str):
                return text.decode("utf-8", "ignore")
            elif isinstance(text, unicode):
                return text
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        else:
            raise ValueError("Not running on Python2 or Python 3?")
    def load_vocab(self, vocab_file):
        """Loads a vocabulary file into a dictionary."""
        vocab = collections.OrderedDict()
        fin = open(vocab_file,encoding='utf-8')
        for num, line in enumerate(fin):
            items = self.convert_to_unicode(line.strip()).split("\t")
            if len(items) > 2:
                break
            token = items[0]
            index = items[1] if len(items) == 2 else num
            token = token.strip()
            vocab[token] = int(index)
        return vocab


class process(object):

    def convert_to_unicode(self, text):
        """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
        if six.PY3:
            if isinstance(text, str):
                return text
            elif isinstance(text, bytes):
                return text.decode("utf-8", "ignore")
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        elif six.PY2:
            if isinstance(text, str):
                return text.decode("utf-8", "ignore")
            elif isinstance(text, unicode):
                return text
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        else:
            raise ValueError("Not running on Python2 or Python 3?")

    def read_file(self,file):
        assert file.endswith('.txt'), "[ERROR] %s is not a txt file" % file
        sentensens = []
        with open(file, encoding='utf-8') as f:
            for line in f:
                sentensens.append(line)
        return sentensens


if __name__ == "__main__":
    pass



# 最后产生的结果是return_list = [
#         src_id, pos_id, sent_id, self_input_mask, mask_label, mask_pos, labels
#     ]
