import os
import torch
import glob
import torch.nn as nn
import pathlib
import torch
import json


# Define the Encoder class
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.LSTM(input_size, hidden_size)

    def forward(self, input_seq):
        _, (hidden_state, cell_state) = self.encoder(input_seq)
        return hidden_state, cell_state

# Define the Decoder class
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.decoder = nn.LSTM(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden_state, cell_state):
        output_seq, _ = self.decoder(input_seq[-1].unsqueeze(0), (hidden_state, cell_state))
        output_seq = self.fc(output_seq)
        return output_seq

# Define the Seq2Seq model
class Seq2SeqModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2SeqModel, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(hidden_size, output_size)

    def forward(self, input_seq):
        hidden_state, cell_state = self.encoder(input_seq)
        output_seq = self.decoder(input_seq, hidden_state, cell_state)
        return output_seq


class StreamPrediction:

    def __init__(self, steam_data):

        self.steam_data = steam_data

        self.active_model_dir= "../models/active/seq2seq_model.pth"
        self.train_model_dir = "../models/train/seq2seq_model.pth"

        self.scaling_file = f"../models/scalingfile/scaling_json_data.json"

        self.script_dir = pathlib.Path(__file__).parent.absolute()

        self.active_model = os.path.join(self.script_dir, self.active_model_dir)
        self.train_model = os.path.join(self.script_dir, self.train_model_dir)

        self.scaling_json_file = os.path.join(self.script_dir, self.scaling_file)


    def model_path_url(self):            

            if  self.active_model:
                active_model_path =  self.active_model

                return active_model_path 

            else:
                train_model_path =  self.train_model

                return train_model_path


        
    def min_max_scaling(self):
                #features as ['engine_ID','cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3', 'T2', 'T24', 'T30', 'T50', 'P2',
                #           'P15', 'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd',
                #           'PCNfR_dmd', 'W31', 'W32']
  
            filename =   self.scaling_json_file
            
            with open(filename, "r") as file:
                data = json.load(file)

            engine_id = self.steam_data[0]  # engine ID helps for scaling with groupby ID
            features_data = self.steam_data[1:]  # Exclude the 'engine_ID' from feature
            
            number_of_features = [len(self.steam_data[1:])]

            current_cycle = [self.steam_data[1]]
            

            scaling_info = None
            cycles_max = []
            rul_min = []
            rul_max = []
            for scaling_data in data:
                if scaling_data['groupby_id'] == engine_id:
                    scaling_info = scaling_data['scaling_data']

                    for feature_data in scaling_info:
                        if feature_data['feature_name'] == 'cycles':
                            cycles_max.append(feature_data['max'])

                        elif feature_data['feature_name'] == 'RUL':
                            rul_min.append(feature_data['min'])
                            rul_max.append(feature_data['max'])
                    break

            scaled_data = []
            for i, feature_data in enumerate(scaling_info):
                if i >= len(features_data):
                    break

                min_value = feature_data['min']
                max_value = feature_data['max']

                feature_value = features_data[i]

                scaled_feature = (feature_value - min_value) / (max_value - min_value)
                scaled_data.append(scaled_feature)

            return scaled_data, cycles_max, current_cycle, rul_max, rul_min, number_of_features



    def make_prediction(self):

        model_path = self.model_path_url()

        min_max_scaled_data = self.min_max_scaling()

        scaled_stream_data = min_max_scaled_data[0]

        max_life_cycle = min_max_scaled_data[1][0]

        current_cycle = min_max_scaled_data[2][0]

        max_rul_value = min_max_scaled_data[3][0]

        min_rul_value = min_max_scaled_data[4][0]

        number_of_features = min_max_scaled_data[5][0]



        input_size = number_of_features # Number of input features
        hidden_size = number_of_features # Number of hidden units in the LSTM layers
        output_size = 1  # Size of the output (hot index)

        model = Seq2SeqModel(input_size, hidden_size, output_size)

        model.load_state_dict(torch.load(model_path))
                                                                                
        actual_rul = max_life_cycle - current_cycle
        
        scaled_stream_data = torch.tensor(scaled_stream_data, dtype=torch.float32)


        scaled_prediction_result = model(scaled_stream_data.unsqueeze(dim=0)).item()  # Unsqueeze to add a batch dimension
        # print(scaled_prediction_result)


        # Reverse scaling formula
        predicted_original_value = scaled_prediction_result * (max_rul_value -min_rul_value) + min_rul_value


        # print(f"Actual Remaining useful life: {actual_rul}")

        # print(f"Predicted Remaining useful life: {predicted_original_value:.2f}")

        # Calculate "LR"
        engine_life_ratio = (predicted_original_value/max_life_cycle)*100

        # print(f"max_life_cycle:{max_life_cycle}")
        # print(f"engine_life_ratio: {engine_life_ratio}")


        if engine_life_ratio >= 60:
            # print("Excellent Condition")
            result={
                "Actual Remaining useful life":f"{actual_rul}",
                "Predicted Remaining useful life":f"{predicted_original_value:.2f}",
                "max_life_cycle":f"{max_life_cycle}",
                "engine_life_ratio": f"{engine_life_ratio}",
                "condition":"Excellent"
            }
        elif 20 < engine_life_ratio <= 60:
            result={
                "Actual Remaining useful life":f"{actual_rul}",
                "Predicted Remaining useful life":f"{predicted_original_value:.2f}",
                "max_life_cycle":f"{max_life_cycle}",
                "engine_life_ratio": f"{engine_life_ratio}",
                "condition":"Moderate"
            }
            # print("Moderate Condition")   
        else :
            result={
                "Actual Remaining useful life":f"{actual_rul}",
                "Predicted Remaining useful life":f"{predicted_original_value:.2f}",
                "max_life_cycle":f"{max_life_cycle}",
                "engine_life_ratio": f"{engine_life_ratio}",
                "condition":"Warning"
            }
            # print("Warning Condition")  
        return result


# prediction = StreamPrediction(steam_data =[2.0, 5.0, 24.9999, 0.62, 60.0, 462.54, 537.0, 1259.55, 1043.95, 7.05, 9.03, 175.64, 1915.26, 8012.87, 0.94, 36.34, 165.3, 2028.13, 7867.08, 10.8841, 0.02, 307.0, 1915.0, 84.93, 14.26, 8.5789
                                           
#                                            ]


# )


# make_prediction = prediction.make_prediction()

# data = {
#     "engine_ID": 2.0,
#     "cycles": 5.0,
#     "op_setting_1": 24.9999,
#     "op_setting_2": 0.62,
#     "op_setting_3": 60.0,
#     "T2": 462.54,
#     "T24": 537.0,
#     "T30": 1259.55,
#     "T50": 1043.95,
#     "P2": 7.05,
#     "P15": 9.03,
#     "P30": 175.64,
#     "Nf": 1915.26,
#     "Nc": 8012.87,
#     "epr": 0.94,
#     "Ps30": 36.34,
#     "phi": 165.3,
#     "NRf": 2028.13,
#     "NRc": 7867.08,
#     "BPR": 10.8841,
#     "farB": 0.02,
#     "htBleed": 307.0,
#     "Nf_dmd": 1915.0,
#     "PCNfR_dmd": 84.93,
#     "W31": 14.26,
#     "W32": 8.5789,
#     "X": 50
# }
