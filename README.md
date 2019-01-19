# Text-to-Shape-Exploration
A UI for exploring 3D objects and search them by query

## Usage
1. Install the required libraries listed in requirements.txt:

   `pip install -r /path/to/requirements.txt`

2. Initilaize `CaptionsFilePath` in config.txt to the path to captions.csv file (included in Text2Shape additional files)
3. Create a file named <b>captions_emb.doc2vec</b> in project root.
4. Before running the web application, run `doc2vec.py` to create the pretrained embeddings. It might take several minutes and the generated file is about 40MB. Remember to <b>uncomment this line</b> after this step (last line in doc2vec.py):

   `train_and_save()`
   
5. Run it as a Flask application using this command in project root: `flask run`. The application will run on http://localhost:5000/.
   
    
    
