# Importing the libraries
import uvicorn
from fastapi import FastAPI,Form
from ChurnPredictors import ChurnPredictor
from prediction import model_prediction

app = FastAPI()


@app.get('/')
def welcome():
    return {'hello':'Hello World'}


@app.post('/predict')
async def get_prediction(data:ChurnPredictor):

    data = data.dict()

    data = [v for v in data.values()]
    
    y_pred = model_prediction(data)

    if y_pred==[0]:
        predict = 'No, the Customer will not Churn'

    else:
        predict = 'Please take relevant action as the customer is going to leave the company'

    return {
        'predict':predict
    }



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
