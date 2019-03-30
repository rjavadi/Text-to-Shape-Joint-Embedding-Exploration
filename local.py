# -*- coding: utf-8 -*-

"""
The Local version of the app.
"""

import base64
import io
import json

import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from textwrap import dedent as d
import configparser

def get_captions_path():
    config = configparser.ConfigParser()
    config.read('config.txt')
    return config['DEFAULT']['CaptionsFilePath']


def input_field(title, state_id, state_value, state_max, state_min):
    """Takes as parameter the title, state, default value and range of an input field, and output a Div object with
    the given specifications."""
    return html.Div([
        html.P(title,
               style={
                   'display': 'inline-block',
                   'verticalAlign': 'mid',
                   'marginRight': '5px',
                   'margin-bottom': '0px',
                   'margin-top': '0px'
               }),

        html.Div([
            dcc.Input(
                id=state_id,
                type='number',
                value=state_value,
                max=state_max,
                min=state_min,
                size=7,
                disabled=True
            )
        ],
            style={
                'display': 'inline-block',
                'margin-top': '0px',
                'margin-bottom': '0px'
            }
        )
    ]
    )


def merge(a, b):
    return dict(a, **b)


def omit(omitted_keys, d):
    return {k: v for k, v in d.items() if k not in omitted_keys}


def Card(children, **kwargs):
    return html.Section(
        children,
        style=merge({
            'padding': 20,
            'margin': 5,
            'borderRadius': 5,
            'border': 'thin lightgrey solid',

            # Remove possibility to select the text for better UX
            'user-select': 'none',
            '-moz-user-select': 'none',
            '-webkit-user-select': 'none',
            '-ms-user-select': 'none'
        }, kwargs.get('style', {})),
        **omit(['style'], kwargs)
    )


# Generate the default scatter plot
# tsne_df = pd.read_csv("data/tsne_3d.csv", index_col=0)
data_df = pd.read_csv("doc2vec_emb.csv", index_col=0)
label_df = pd.read_csv("new_labels.csv")
pca = PCA(n_components=24)
data_pca = pca.fit_transform(data_df)
tsne = TSNE(n_components=3,
            perplexity=25,
            learning_rate=300,
            n_iter=500)
data_tsne = tsne.fit_transform(data_pca)
tsne_data_df = pd.DataFrame(data_tsne, columns=['x', 'y', 'z'])

# label_df.columns = ['category']

# combined_df = tsne_data_df.join(label_df)
tsne_data_df['category'] = label_df['category']
tsne_data_df['modelId'] = label_df['modelId']
tsne_data_df['captionId'] = label_df['id']

data = []

for idx, val in tsne_data_df.groupby('category'):
    # idx = int(idx)

    scatter = go.Scatter3d(
        name=idx,
        x=val['x'],
        y=val['y'],
        z=val['z'],
        customdata=val['modelId'],
        text=val['captionId'],
        mode='markers',
        marker=dict(
            size=2.5,
            symbol='circle'
        )
    )
    data.append(scatter)

# Layout for the t-SNE graph
tsne_layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)

local_layout = html.Div([
    # In-browser storage of global variables
    html.Div(
        id="data-df-and-message",
        style={'display': 'none'}
    ),

    html.Div(
        id="label-df-and-message",
        style={'display': 'none'}
    ),

    # Main app
    html.Div([
        html.H2(
            't-SNE Explorer',
            id='title',
            style={
                'float': 'left',
                'margin-top': '20px',
                'margin-bottom': '0',
                'margin-left': '7px'
            }
        ),
        html.Img(
            src="https://s3-us-west-1.amazonaws.com/plotly-tutorials/logo/new-branding/dash-logo-by-plotly-stripe.png",
            style={
                'height': '100px',
                'float': 'right'
            }
        )
    ],
        className="row"
    ),

    html.Div([
        html.Div([
            # Data about the graph
            html.Div(
                id="kl-divergence",
                style={'display': 'none'}
            ),

            html.Div(
                id="end-time",
                style={'display': 'none'}
            ),

            html.Div(
                id="error-message",
                style={'display': 'none'}
            ),

            # The graph
            dcc.Graph(
                id='tsne-3d-plot',
                figure={
                    'data': data,
                    'layout': tsne_layout
                },
                style={
                    'height': '80vh',
                },
            )
        ],
            id="plot-div",
            className="eight columns"
        ),
        html.Div(className="four columns", children=[
            Card([

                html.H4(
                    't-SNE Parameters',
                    id='tsne_h4'
                ),

                input_field("Perplexity:", "perplexity-state", tsne.perplexity, 50, 5),

                input_field("Number of Iterations:", "n-iter-state", tsne.n_iter, 1000, 250),

                input_field("Learning Rate:", "lr-state", tsne.learning_rate, 1000, 10),

                input_field("Initial PCA dimensions:", "pca-state", pca.n_components, 10000, 3),

                html.Div([
                    html.P(id='upload-data-message',
                           style={
                               'margin-bottom': '0px'
                           }),

                    html.P(id='upload-label-message',
                           style={
                               'margin-bottom': '0px'
                           }),

                    html.Div(id='training-status-message',
                             style={
                                 'margin-bottom': '0px',
                                 'margin-top': '0px'
                             }),

                    html.P(id='error-status-message')
                ],
                    id='output-messages',
                    style={
                        'margin-bottom': '2px',
                        'margin-top': '2px'
                    }
                )

            ]),
            Card(style={'padding': '5px'}, children=[
                html.Div(id='div-plot-click-message',
                         style={'text-align': 'center',
                                'margin-bottom': '7px',
                                'font-weight': 'bold'}
                         ),

                html.Div([
                    dcc.Markdown(d("""
                **Click Data**

                Click on points in the graph.
            """)),

                ], id="click-data", style={'text-align': 'center',
                                           'margin-bottom': '7px'}),

                html.Div(id='div-plot-click-wordemb')
            ])
        ])
    ],
        className="row"
    ),

],
    className="container",
    style={
        'width': '90%',
        'max-width': 'none',
        'font-size': '1.5rem'
    }
)


def local_callbacks(app):
    def parse_content(contents, filename):
        """This function parses the raw content and the file names, and returns the dataframe containing the data, as well
        as the message displaying whether it was successfully parsed or not."""

        if contents is None:
            return None, ""

        content_type, content_string = contents.split(',')

        decoded = base64.b64decode(content_string)

        try:
            if 'csv' in filename:
                # Assume that the user uploaded a CSV file
                df = pd.read_csv(
                    io.StringIO(decoded.decode('utf-8')))
            elif 'xls' in filename:
                # Assume that the user uploaded an excel file
                df = pd.read_excel(io.BytesIO(decoded))

            else:
                return None, 'The file uploaded is invalid.'
        except Exception as e:
            print(e)
            return None, 'There was an error processing this file.'

        return df, f'{filename} successfully processed.'

    # # Hidden Data Div --> Display upload status message (Data)
    # @app.callback(Output('upload-data-message', 'children'),
    #               [Input('data-df-and-message', 'children')])
    # def output_upload_status_data(data):
    #     return data[1]
    #
    # # Hidden Label Div --> Display upload status message (Labels)
    # @app.callback(Output('upload-label-message', 'children'),
    #               [Input('label-df-and-message', 'children')])
    # def output_upload_status_label(data):
    #     return data[1]

    @app.callback(
        Output('click-data', 'children'),
        [Input('tsne-3d-plot', 'clickData')])
    def display_click_data(clickData):
        print("%%%%%%%%%%%%%%%%%", clickData)
        modelId = clickData['points'][0]['customdata']
        # return json.dumps(clickData, indent=2)
        return html.Img(
            src="http://dovahkiin.stanford.edu/scene-toolkit/assets/download/3dw/image/" + modelId + "/rotatingImage",
            width=100, height=100)

    # Updated graph --> Training status message
    @app.callback(Output('training-status-message', 'children'),
                  [Input('end-time', 'children'),
                   Input('kl-divergence', 'children')])
    def update_training_info(end_time, kl_divergence):
        # If an error message was output during the training.

        if end_time is None or kl_divergence is None or end_time[0] is None or kl_divergence[0] is None:
            return None
        else:
            end_time = end_time[0]
            kl_divergence = kl_divergence[0]

            return [
                html.P(f"t-SNE trained in {end_time:.2f} seconds.",
                       style={'margin-bottom': '0px'}),
                html.P(f"Final KL-Divergence: {kl_divergence:.2f}",
                       style={'margin-bottom': '0px'})
            ]

    @app.callback(Output('div-plot-click-wordemb', 'children'),
                  [Input('tsne-3d-plot', 'clickData')])
    def display_click_word_neighbors(clickData):
        if clickData:
            caption_id = clickData['points'][0]['text']
            captions_df = pd.read_csv(get_captions_path(), usecols=['id', 'description', 'modelId'], index_col='id')
            text = captions_df.loc[caption_id]['description']

            # Get the nearest neighbors indices using Euclidean distance
            # vector = data_dict[dataset].set_index('0')
            # selected_vec = vector.loc[selected_word]

            # def compare_pd(vector):
                # return spatial_distance.euclidean(vector, selected_vec)

            # distance_map = vector.apply(compare_pd, axis=1)
            # nearest_neighbors = distance_map.sort_values()[1:6]

            # trace = go.Bar(
            #     x=nearest_neighbors.values,
            #     y=nearest_neighbors.index,
            #     width=0.5,
            #     orientation='h'
            # )

            # layout = go.Layout(
            #     title=f'5 nearest neighbors of "{selected_word}"',
            #     xaxis=dict(title='Euclidean Distance'),
            #     margin=go.Margin(l=60, r=60, t=35, b=35)
            # )
            #
            # fig = go.Figure(data=[trace], layout=layout)
            #
            return dcc.Graph(
                id='graph-bar-nearest-neighbors-word',
                figure=fig,
                style={'height': '25vh'},
                config={'displayModeBar': False}
            )

        else:
            return None

    @app.callback(Output('error-status-message', 'children'),
                  [Input('error-message', 'children')])
    def show_error_message(error_message):
        if error_message is not None:
            return [
                html.P(error_message[0])
            ]

        else:
            return []
