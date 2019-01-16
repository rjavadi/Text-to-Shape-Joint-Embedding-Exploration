import errno

from flask import Flask
import argparse
import os.path

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
import doc2vec

import pandas as pd
import os
from difflib import SequenceMatcher
import configparser

app = Flask(__name__)

print("name: ", __name__)


def get_file_path():
    CONFIG_FILE_PATH = './config.txt'

    if not os.path.isfile(CONFIG_FILE_PATH):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), CONFIG_FILE_PATH)

    config = configparser.ConfigParser()
    config.read('config.txt')
    file_path = config['DEFAULT']['CaptionsFilePath']
    if not os.path.isfile(CONFIG_FILE_PATH):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), file_path)
    return file_path


if __name__ == '__main__':
    print("hello")
    app.run(port=8080)


@app.route('/')
def hello_world():
    return render_template('drpdn-search.html')


@app.route('/search')
def search():
    return render_template('query-page.html')


@app.route('/result', methods=('GET', 'POST'))
def show_result():
    if request.method == 'POST':
        caption_id = request.form['query']
        if not caption_id:
            error = 'Please enter a query'
            render_template('query-page.html', error=error)
        else:
            caption_dict = find_caption(caption_id)
            return render_template('result.html', captions=caption_dict)

    return render_template('query-page.html')


@app.route('/search-drop', methods=('GET', 'POST'))
def search_dropdown():
    print("in search by dropdown")
    if request.method == 'POST':
        color = request.form['color']
        shape = request.form['shape']
        material = request.form['material']
        object_type = request.form['objectType']
        style = request.form['style']
        query = shape + ' ' + color + ' ' + object_type
        if material != '':
            query += ' made of ' + material
        if style != '':
            query += ' it is ' + style
        if not query:
            error = 'Please select some features!'
            render_template('query-page.html', error=error)
        else:
            caption_dict = find_caption(query)
            # caption_dict = dummy_cap()
            return render_template('result.html', captions=caption_dict)

    return render_template('drpdn-search.html')


def find_caption(query):
    file_path = get_file_path()
    # captions_df = pd.read_csv(os.path.join("F:/", "DL-code", "text2shape-data", "captions.csv"),
    #                            index_col=['id'])
    captions_df = pd.read_csv(file_path,
                              index_col=['id'])
    result_dict = {}
    similars = doc2vec.find_similars(query)
    for id, score in similars:
        model_id = captions_df.loc[int(id[5:]), 'modelId']
        description = captions_df.loc[int(id[5:]), 'description']
        result_dict.update({model_id: description})

    print(similars)
    return result_dict