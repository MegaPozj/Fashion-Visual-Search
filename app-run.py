import os
import skimage.draw
import PIL
from PIL import Image
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.backend import set_session

from detection.lib.config import Config
from detection.lib.model import MaskRCNN

from flask import Flask, render_template, url_for, request, redirect, send_from_directory
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from demo.visual_search import _init_models, visual_search
from flask_dropzone import Dropzone

import jinja2
env = jinja2.Environment()

app = Flask(__name__)

app.jinja_env.globals.update(zip=zip)

basedir = os.path.abspath(os.path.dirname(__file__))

app.config.update(
    UPLOADED_PATH=os.path.join(basedir, 'static/images/uploads'),
    DROPZONE_MAX_FILE_SIZE=3,
    DROPZONE_MAX_FILES=1,
    DROPZONE_REDIRECT_VIEW='completed',
    DROPZONE_ALLOWED_FILE_TYPE = 'image',
    DROPZONE_DEFAULT_MESSAGE = 'Drop an image here or click to upload'
)

dropzone = Dropzone(app)



class DetectionConfig(Config):
    """
    Configuration for performing detection.
    Derives from the base Config class.
    """
    NAME = "Fashion Item Detection"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 13 
    USE_MINI_MASK = True

global models
def init_models():
    # Build model for detection
    global session
    session = tf.Session(graph=tf.Graph())
    with session.graph.as_default():
        keras.backend.set_session(session)
        global model_dt
        det_config = DetectionConfig()
        model_dt = MaskRCNN(mode="inference", config=det_config, model_dir="None")
        model_dt.load_weights("detection/logs/mask_rcnn_deepfashion2_0005.h5", by_name=True)
        model_dt.keras_model._make_predict_function()

    # Load models and database for retrieval
    model_rt, model_lm, gallery_embeds, retriever = _init_models()
    models = {
        'model_rt': model_rt,
        'model_lm': model_lm,
        'gallery_embeds': gallery_embeds,
        'retriever': retriever
    }
    print("All models and database are loaded.")
    return models


@app.route("/", methods=["POST", "GET"])
def upload():
    global results
    global categories
    global filename
    global dt_results
    if request.method == 'POST':
        f = request.files.get('file')
        filename = time.strftime("%Y%m%d-%H%M%S") + '.jpg'
        query_path = os.path.join(app.config['UPLOADED_PATH'], filename)
        f.save(query_path)

        # Convert image to RGB
        img = Image.open(query_path)
        img = img.convert('RGB')
        img.save(query_path)

        # Perform detection
        img = skimage.io.imread(query_path)
        with session.graph.as_default():
            keras.backend.set_session(session)       
            dt_results = model_dt.detect([img], verbose=1)[0]
        
        # Perform visual search based on detection results
        results, categories = visual_search(img, models, dt_results, 'det_' + filename)
    else:
        results = []
        categories = []
        return render_template('index.html', results=results, categories=categories)

@app.route('/completed')
def completed():
    if len(results) != 0:
        return render_template('index.html', results=results, categories=categories, filename=filename, len_dt_results=len(dt_results['rois']))
    else:
        return redirect('/')


if __name__ == '__main__':
    models = init_models()
    app.run(debug=True, port=8000)
