import configparser
import dash
from local import local_callbacks, local_layout
import errno
import os
import os.path


app_dash = dash.Dash("Dash app")

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
    app_dash.run_server(debug=True)
    # server.run(port=8080, debug=True)
