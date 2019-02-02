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


def get_embedding_path():
    config = configparser.ConfigParser()
    config.read('config.txt')
    return config['DEFAULT']['EmbeddingsCsv']


def get_labels_path():
    config = configparser.ConfigParser()
    config.read('config.txt')
    return config['DEFAULT']['LabelsCsv']


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


def create_embedding_datafram():
    model = Doc2Vec.load(get_model_path())
    docs = model.docvecs.vectors_docs

    captions_df = pd.read_csv(get_captions_path(), usecols=['category'])

    top_1000keys = list(model.docvecs.doctags.keys())[:1000]
    labels_df = pd.DataFrame(captions_df['category'].iloc[:1000])

    docs_df = pd.DataFrame(docs[:1000])
    docs_df.to_csv(get_embedding_path(), index=False)

    labels_df.to_csv(get_labels_path(), index=False)
    print('labels and embeddings csv created.')


# train_and_save()
create_embedding_datafram()

