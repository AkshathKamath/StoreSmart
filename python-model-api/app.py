from fastapi import FastAPI
from data.data_extracting import data_extractor
from aws.aws_handling import aws_img_saver
from model.model_img import Kmeans_model

app = FastAPI()

#-------------------------------------------------------#

@app.get('/model')
def model_route():
    df = data_extractor
    img_buffer = Kmeans_model(df)
    # aws_img_saver(img_buffer)
    return {"msg":"Model image saved to AWS!"}

#-------------------------------------------------------#

if __name__ == '__main__':
	uvicorn.run(app, host='0.0.0.0', port=5000)