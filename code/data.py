import os
from collections import defaultdict
import numpy as np
import copy
import random
import logging


class Data_loader():
    def __init__(self, args):
        self.args = args
        self.include_reverse = True  # S_P_O to O_P-_S

        self.train_data = None
        self.test_data = None
        self.valid_data = None

        self.entity2num = None
        self.num2entity = None

        self.relation2num = None
        self.num2relation = None
        # self.relation2inv = None

        self.num_relation = 0
        self.num_entity = 0
        # self.num_operator = 0

        data_path = os.path.join(self.args.data_dir, self.args.dataset)
        self.load_data_all(data_path)

    def load_data_all(self, path):
        train_data_path = os.path.join(path, "train.txt")
        test_data_path = os.path.join(path, "test.txt")
        valid_data_path = os.path.join(path, "valid.txt")
        entity_path = os.path.join(path, "entities.txt")
        relations_path = os.path.join(path, "relations.txt")

        self.entity2num, self.num2entity = self._load_dict(entity_path)
        self.relation2num, self.num2relation = self._load_dict(relations_path)
        self._augment_reverse_relation()
        self._add_item(self.relation2num, self.num2relation, "Equal")  #  使实体有自回路
        self._add_item(self.relation2num, self.num2relation, "Pad")   #  Padding
        self._add_item(self.relation2num, self.num2relation, "Start")  #  Initialize the relationship at the beginning of training
        self._add_item(self.entity2num, self.num2entity, "Pad")

        self.num_relation = len(self.relation2num)
        logging.info('Num_relation : %s' % (self.num_relation))
        self.num_entity = len(self.entity2num)
        logging.info('Num_entity : %s' % (self.num_entity))

        self.train_data = self._load_data(train_data_path)
        self.valid_data = self._load_data(valid_data_path)
        self.test_data = self._load_data(test_data_path)

    def _load_dict(self, path):
        '''
        Convert the corresponding entity/relation to id, and id to entity/relation.
        :param path: entitys or relations file
        :return : two dicts
        '''
        obj2id = defaultdict(int)
        id2obj = defaultdict(str)
        data = [l.strip() for l in open(path, "r").readlines()]
        for id, obj in enumerate(data):
            obj2id[obj] = id
            id2obj[id] = obj
        return obj2id, id2obj

    def _augment_reverse_relation(self):
        '''
        To amplify (En_head, Rel, En_tail) to (En_tail, inv_Rel, En_head), expand relation to inv_relation.
        '''
        num_relation = len(self.num2relation)
        temp = list(self.num2relation.items())
        # self.relation2inv = defaultdict(int)
        for n, r in temp:
            rel = "inv_" + r
            num = num_relation + n
            self.relation2num[rel] = num
            self.num2relation[num] = rel
            # self.relation2inv[n] = num
            # self.relation2inv[num] = n

    def _add_item(self, obj2id, id2obj, item):
        '''
        Add a item manually for obj2id and id2obj.
        :param obj2id: entity2num or relation2num
        :param id2obj: num2entity or num2relation
        :param item: "Start", "Equal" or "Pad"
        '''
        count = len(obj2id)
        obj2id[item] = count
        id2obj[count] = item

    def _load_data(self, path):
        '''
        Load triples (with inverse) in train, test and valid respectively.
        :param path: train/test/valid file path
        :return: triples
        '''
        data = [l.strip().split("\t") for l in open(path, "r").readlines()]
        triples = list()
        for item in data:
            head = self.entity2num[item[0]]
            tail = self.entity2num[item[2]]
            relation = self.relation2num[item[1]]
            triples.append([head, relation, tail])
            if self.include_reverse:
                inv_relation = self.relation2num["inv_" + item[1]]
                triples.append([tail, inv_relation, head])
        random.shuffle(triples)
        return triples


    def get_train_graph_data(self):
        logging.info("Train graph contains " + str(len(self.train_data)) + " triples")
        return np.array(self.train_data, dtype=np.int64)

    def get_train_data(self):
        logging.info("Train data contains " + str(len(self.train_data)) + " triples")
        return np.array(self.train_data, dtype=np.int64)

    def get_test_graph_data(self):
        logging.info("Test graph contains " + str(len(self.train_data)) + " triples")
        return np.array(self.train_data, dtype=np.int64)

    def get_test_data(self):
        logging.info("Test data contains " + str(len(self.test_data)) + " triples")
        return np.array(self.test_data, dtype=np.int64)