from flask import Flask, flash, request, redirect, url_for
from flask_restful import Resource, Api
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
import json
import os

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'jpg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app, resources={r"/*": {"origins": "*"}})
api = Api(app)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/dream", methods=["POST"])
def dream():

    if 'file' not in request.files:
        return json.dumps({'status' : "file not in files"}), 200, {'ContentType':'application/json'}
    file = request.files['file']

    if file.filename == "":
        return json.dumps({'status' : "no file"}), 200, {'ContentType':'application/json'}

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        name, ext = os.path.splitext(filename)
        file.save(os.path.join(app.config["UPLOAD_FOLDER"], "{}_{}{}".format(name, dream.image_id, ext)))
        dream.image_id += 1
        return json.dumps({'status' : "success"}), 200, {'ContentType':'application/json'}

    return json.dumps({'status' : "file type not allowed"}), 200, {'ContentType':'application/json'}
dream.image_id = 0

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=2555)