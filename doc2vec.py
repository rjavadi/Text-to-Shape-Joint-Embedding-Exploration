from gensim.test.utils import common_texts, get_tmpfile
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim import similarities, models
from gensim.utils import simple_preprocess
import configparser
import pandas as pd


def get_model_path():
    config = configparser.ConfigParser()
    config.read('config.txt')
    return config['DEFAULT']['CaptionsDoc2VecModel']


def get_captions_path():
    config = configparser.ConfigParser()
    config.read('config.txt')
    return config['DEFAULT']['CaptionsFilePath']


def read_corpus(tokens_only=False):
    captions_df = pd.read_csv(get_captions_path())
    captions_limit = 75358
    for i, line in enumerate(captions_df[:captions_limit]['description'].values.tolist()):
        if len(line) < 3:
            print(line)
        if tokens_only:
            yield simple_preprocess(line)
        else:
            id = captions_df['id'].iat[i]
            yield TaggedDocument(simple_preprocess(line), ['SENT_%d' %id])


def train_and_save():
    documents = list(read_corpus())
    model = Doc2Vec(documents, vector_size=64, window=1, min_count=1,   workers=8, epochs=50, train_lbls=False)
    model.save(get_model_path())


def find_similars(query):
    # // TODO: if model file not exists, train and save it.
    model = Doc2Vec.load(get_model_path())
    vec_emb = model.infer_vector(query.lower().split())
    return model.docvecs.most_similar([vec_emb])


train_and_save()
