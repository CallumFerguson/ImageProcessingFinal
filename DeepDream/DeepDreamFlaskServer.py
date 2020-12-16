from flask import Flask, flash, request, redirect, url_for, send_file
from flask_restful import Resource, Api
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
import json
import os
import DeepDream
import math
import time
import base64

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'jpg', 'png'}

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
        new_name = "{}_{}_{}{}".format(name, dream.image_id, math.floor(time.time()), ext)
        file.save(os.path.join(app.config["UPLOAD_FOLDER"], new_name))
        dream.image_id += 1
        arguments = request.form.get("arguments")
        arguments = json.loads(arguments)
        DeepDream.run(new_name, arguments)
        name, _ = os.path.splitext(new_name)
        output_path = os.path.join("outputs", "{}.png".format(name))
        with open(output_path, "rb") as image_file:
            image_encoded_string = base64.b64encode(image_file.read())
            image_string = image_encoded_string.decode("utf-8")
        return json.dumps({'status' : "success", "image" : image_string}), 200, {'ContentType':'application/json'}

    return json.dumps({'status' : ".jpg and .png file types only."}), 200, {'ContentType':'application/json'}
dream.image_id = 0

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=2555, threaded=False)