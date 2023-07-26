import numpy as np
import onnxruntime as rt
import pandas as pd
import os
import json
from core.data_processor import DataLoader

def onnx():
    # get hyperparameters from json
    configs = json.load(open('config_2.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    # get data from csv
    filename = os.path.join('data', configs['data']['filename'])

    cols = configs['data']['columns']

    dataframe = pd.read_csv(filename)
    data_full = dataframe.get(cols)
    row = data_full.iloc[1610].values

    print(row)

    # Load the ONNX model
    sess = rt.InferenceSession(
        r'C:\Users\flash\OneDrive\桌面\LSTM_predict_assistant_level\converted_models\cw2_al.onnx')

    # Assume input_data is your input in numpy array format
    # Make sure the input data type and shape match the model's expected input
    input_data = np.array(row, dtype=np.float32)
    input_data = input_data[np.newaxis, np.newaxis, :]  # adds two dimensions
    print(input_data)

    # Get the name of the input node
    input_name = sess.get_inputs()[0].name

    # Make prediction
    prediction = sess.run(None, {input_name: input_data})

    print(prediction[0] * 0.11242022 + 0.26366002)


if __name__ == '__main__':
    onnx()
