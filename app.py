import errno

from flask import Flask
import argparse
import os.path

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)

import pandas as pd
import os
from difflib import SequenceMatcher

app = Flask(__name__)

print("name: ", __name__)


def get_file_path():
    CONFIG_FILE_PATH = './config.txt'

    if not os.path.isfile(CONFIG_FILE_PATH):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), CONFIG_FILE_PATH)

    conf = open(CONFIG_FILE_PATH, mode='r')
    file_path = conf.readline()
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

    print("TODO: file found")
    # return captions_df.loc[int(id)]
    captions_df['ratio'] = captions_df.apply(lambda row: calculate_ratio(row, query), axis=1)
    sorted_df = captions_df.sort_values('ratio', ascending=False).head(10)
    cap_dict = dict(zip(sorted_df['modelId'], sorted_df['description']))
    return cap_dict


def calculate_ratio(row, query):
    if type(row['description']) is str:
        return SequenceMatcher(None, query, row['description']).ratio()


def dummy_cap():
    return {'8806336': 'Brown wood picnic table.',
            '6784346': 'can be used to keep things.', '7658344': 'This looks like a brown picnic table',
            '1122345': 'narrow, wooden high table',
            '4379243': 'An old fashion big brown color chair with thick arms and very thin back.'}
