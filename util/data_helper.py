#!/usr/local/python
# _*_ coding: utf-8 _*_
import re
import os
import cPickle
import logging
import multiprocessing
import numpy as np
import time

import sys
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_label(positive_data_file,negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    print "positive_size:" + str(len(positive_labels))
    print "negative_size:" + str(len(negative_labels))
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def batch_iter(data,batch_size,num_epochs,shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        print "num_epoch_"+str(epoch)
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def load_data(fpath,path):
    print "loading glove..."
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = cPickle.load(f)
        return data
    else:
        data = open(fpath,'rb').read()
        # w1 = {}
        vec = open(fpath, 'rb')
        # for line in vec.readlines():
        #     line = line.split(' ')
        #     w1[line[0]] = np.asarray([float(x) for x in line[1:]]).astype('float32')
        # vec.close()
        with open(path, 'wb') as pf:
            cPickle.dump(vec.readline(), pf,-1)
        pf.close()
        return data

def save(path,data):
    with open(path, 'wb') as pf:
        cPickle.dump(data, pf, -1)
    pf.close()

def load(fpath):
    with open(fpath, 'rb') as f:
        data = cPickle.load(f)
    return data

def output_vocab(vocab):
    for k,v in vocab.item():
        print k

def embedding_sentences(sentences,embedding_size = 128,window = 5,min_count = 5,file_to_load = None,file_to_save = None):
    if file_to_load is not None:
        w2vModel = Word2Vec.load(file_to_load)
    else:
        w2vModel = Word2Vec(sentences,size=embedding_size,window = window, min_count = min_count, workers = multiprocessing.cpu_count())
        if file_to_save is not None:
            w2vModel.save(file_to_save)
    all_vectors = []
    embeddingDim = w2vModel.vector_size
    embeddingUnknown = [0 for i in range(embeddingDim)]
    for sentence in sentences:
        this_vector = []
        for word in sentence:
            if word in w2vModel.wv.vocab:
                this_vector.append(w2vModel[word])
            else:
                this_vector.append(embeddingUnknown)
        all_vectors.append(this_vector)
    return all_vectors

def generate_word2vec_files(input_file, output_model_file, output_vector_file, size = 128, window = 5, min_count = 5):
    start_time = time.time()
    # model.init_sims(replace=True)
    model = Word2Vec(LineSentence(input_file),size=size,window=window, min_count = min_count, workers = multiprocessing.cpu_count())
    model.save(output_model_file)
    model.wv.save_word2vec_format(output_vector_file,binary=False)

    end_time = time.time()
    print "used time : %d s" % (end_time - start_time)

def run_main():
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
    logging.info("running %s" % ' '.join(sys.argv))

    if len(sys.argv)<4:
        print globals()['__doc__'] % locals()
        sys.exit(1)
    input_file,output_model_file,output_vector_file = sys.argv[1:4]

    generate_word2vec_files(input_file,output_model_file,output_vector_file)

def test():
    vectors = embedding_sentences([['first', 'sentence'], ['second', 'sentence']], embedding_size = 4, min_count = 1)
    print vectors




