import configparser
import errno
import os
import os.path

import dash
import pandas as pd
from flask import Flask
from flask import (
    render_template, request
)
from local import local_callbacks, local_layout

import doc2vec

server = Flask(__name__)
app_dash = dash.Dash(__name__, server=server)

print("name: ", __name__)

external_css = [
    "https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css",
    "//fonts.googleapis.com/css?family=Raleway:400,300,600",
    "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css"
]

for css in external_css:
    app_dash.css.append_css({"external_url": css})


app_dash.layout = local_layout
local_callbacks(app_dash)

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
    server.run(port=8080, debug=True)


@server.route('/')
def hello_world():
    return render_template('drpdn-search.html')


@server.route('/search')
def search():
    return render_template('query-page.html')


@server.route('/result', methods=('GET', 'POST'))
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


@server.route('/search-drop', methods=('GET', 'POST'))
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