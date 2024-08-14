from flask import Flask,jsonify
from data.data_extracting import data_extractor
from aws.aws_handling import aws_img_saver
from model.model_img import Kmeans_model

app = Flask(__name__)

#-------------------------------------------------------#

@app.route('/model', methods=['GET'])
def model_route():
    df = data_extractor
    img_buffer = Kmeans_model(df)
    # aws_img_saver(img_buffer)
    return {"msg":"Model image saved to AWS!"}

#-------------------------------------------------------#

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)