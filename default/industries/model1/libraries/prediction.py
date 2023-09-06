import os
import torch
import glob
import torch.nn as nn
import pathlib
import torch
import json


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, input_sequence):
        encoded_sequence, _ = self.lstm(input_sequence)
        return encoded_sequence
    
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(hidden_size, output_size)
        self.linear = nn.Linear(output_size, output_size)

    def forward(self, encoded_sequence):
        decoded_sequence, _ = self.lstm(encoded_sequence)
        decoded_sequence = self.linear(decoded_sequence)
        return decoded_sequence

class Seq2SeqModelClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2SeqModelClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(hidden_size, output_size)

    def forward(self, input_sequence):
        encoded_sequence = self.encoder(input_sequence)
        output_sequence = self.decoder(encoded_sequence)
        return output_sequence
    


class Prediction:

    def __init__(self, steam_data):

        self.steam_data = steam_data
        self.active_model_dir= "../models/active/seq2seq_lstm_classifier.pth"
        self.train_model_dir = "../models/train/seq2seq_lstm_classifier.pth"

        self.scaling_file = f"../models/scalingfile/scaling_json_data.json"

        self.script_dir = pathlib.Path(__file__).parent.absolute()

        self.active_model = os.path.join(self.script_dir, self.active_model_dir)
        self.train_model = os.path.join(self.script_dir, self.train_model_dir)

        self.scaling_json_file = os.path.join(self.script_dir, self.scaling_file)

        
    def scaling_result(self):
                #features as ['cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3', 'T2', 'T24', 'T30', 'T50', 'P2',
                #           'P15', 'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd',
                #           'PCNfR_dmd', 'W31', 'W32']
  
            filename =   self.scaling_json_file
            
            with open(filename, "r") as file:
                data = json.load(file)

            engine_id = self.steam_data[0]  # engine ID helps for scaling with groupby ID
            features_data = self.steam_data[1:]  # Exclude the 'engine_ID' from feature       

            scaling_info = None
            num_classes = None
            for scaling_data in data:
                num_classes  = scaling_data['num_classes']
                if scaling_data['groupby_id'] == engine_id:
                    scaling_info = scaling_data['scaling_data']
                    break

            scaled_data = []
            for i, feature_data in enumerate(scaling_info):
                # print(feature_data)
                if i >= len(features_data):
                    break

                min_value = feature_data['min']
                max_value = feature_data['max']

                feature_value = features_data[i]

                scaled_feature = (feature_value - min_value) / (max_value - min_value)
                scaled_data.append(scaled_feature)
            
            return scaled_data ,num_classes



    def make_prediction(self):

        model_path = self.train_model

        scaled_data = self.scaling_result()

        scaled_stream_data = scaled_data[0]
        # print(scaled_stream_data)

        number_of_features = len(scaled_stream_data)

        num_classes = scaled_data[1]
        input_size = number_of_features
        hidden_size = number_of_features
        output_size = num_classes
        

        loaded_model = Seq2SeqModelClassifier(input_size, hidden_size, output_size)
        loaded_model.load_state_dict(torch.load(model_path))

        new_data = torch.tensor([scaled_stream_data], dtype=torch.float32)

        loaded_model.eval()
        with torch.no_grad():
            output = loaded_model(new_data)
            probabilities = torch.softmax(output, dim=1)
            # print(probabilities)
            _, predicted_classes = torch.max(output, 1)

        # print(predicted_classes.item())
        predicted_classes = predicted_classes.item()

        if predicted_classes == 0:
            result = "Excellent Condition"
           
        elif predicted_classes == 1:
            result = "Moderate Condition"
        else :
            result = "Warning Condition"
        
        return result

# 1,6,34.9996,0.84,100.0,449.44,554.77,1352.87,1117.01,5.48,7.97,193.82,2222.77,8340.0,1.02,41.44,181.9,2387.87,8054.1,9.3346,0.02,330,2223,100.0,14.91,8.9057 #0
# 1,201,0.0024,0.0,100.0,518.67,641.85,1583.2,1397.76,14.62,21.58,554.19,2387.98,9049.68,1.3,47.0,521.59,2387.91,8139.36,8.3595,0.03,391,2388,100.0,38.97,23.3957 #1
# 1,309,20.0038,0.7015,100.0,491.19,607.38,1495.68,1259.69,9.35,13.61,340.88,2324.52,8765.05,1.09,44.85,321.49,2388.64,8094.77,8.9928,0.03,367,2324,100.0,24.91,14.9363 #2
# 6,24,42.0018,0.8418,100.0,445.0,549.52,1350.65,1107.78,3.91,5.69,137.92,2211.95,8321.3,1.02,41.72,130.16,2388.05,8076.51,9.3355,0.02,328,2212,100.0,10.74,6.4205 #0
# 6,179,0.0016,0.0,100.0,518.67,641.93,1581.83,1403.18,14.62,21.57,555.44,2388.02,9052.79,1.3,47.28,523.03,2388.03,8136.98,8.3538,0.03,391,2388,100.0,39.1,23.5131 #1
# 6,290,35.005,0.8416,100.0,449.44,555.54,1362.08,1133.11,5.48,7.97,196.75,2223.35,8379.37,1.03,42.01,185.72,2388.45,8094.65,9.1539,0.02,335,2223,100.0,14.98,9.0517 #2


prediction = Prediction(steam_data =[6,290,35.005,0.8416,100.0,449.44,555.54,1362.08,1133.11,5.48,7.97,196.75,2223.35,8379.37,1.03,42.01,185.72,2388.45,8094.65,9.1539,0.02,335,2223,100.0,14.98,9.0517

]

)

make_prediction = prediction.make_prediction()
print(make_prediction)
