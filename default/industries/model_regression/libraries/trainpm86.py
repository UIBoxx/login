import os
import torch
import pathlib
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from features_scaling import FeaturesScaling, min_max_scaling_groupby


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


#Define Training model
class TrainPM86:

    def __init__(self, epoch, lr_rate, train_test_ratio):

        self.lr_rate = lr_rate
        self.epoch_number = epoch
        self.train_model_dir = "../models/train"
        self.train_test_ratio = train_test_ratio

        self.script_dir = pathlib.Path(__file__).parent.absolute()
        self.train_models_dir = os.path.join(self.script_dir, self.train_model_dir)

       
    def trainpm86(self):

        # scaling = FeaturesScaling()
        # min_max_scaling_groupby = min_max_scaling_groupby
        # print(min_max_scaling_groupby)

        num_features = len(min_max_scaling_groupby.axes[1]) -1  # 1 is output features from above
        # print(num_features)

        normalized_data = min_max_scaling_groupby[0:700]  # Take 700 data point for now
        # print(normalized_data)
        data_array = normalized_data.to_numpy()

        X = data_array[:, :-1]  # Features
        y = data_array[:, -1]   # target value
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.train_test_ratio, random_state=42)

        # Convert the numpy arrays to torch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)


        input_size =  num_features # Number of input features
        hidden_size =  num_features # Number of hidden units in the LSTM layers
        output_size = 1  # Size of the output (hot index)
        lr_rate = self.lr_rate

        model = Seq2SeqModel(input_size, hidden_size, output_size)


        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr =lr_rate)

        # Training loop
        num_epochs = self.epoch_number 
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()

            output = model(X_train.unsqueeze(dim=0))

            mse_loss = criterion(output.squeeze(), y_train)

            rmse_loss = torch.sqrt(mse_loss)

            mse_loss.backward()
            optimizer.step()


            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Training MSE Loss: {mse_loss.item():.6f},Training RMSE Loss: {rmse_loss.item():.6f}")

        torch.save(model.state_dict(), f'{self.train_models_dir}/seq2seq_model.pth')


        model.eval()
        with torch.no_grad():
            # Forward pass
            output = model(X_test.unsqueeze(dim=0)) 
            y_pred = output.squeeze()
            mse_loss = criterion(y_test, y_pred)
            rmse_loss = torch.sqrt(mse_loss)


            y_mean = torch.mean(y_test)
            tss = torch.sum((y_test - y_mean) ** 2)
            rss = torch.sum((y_test - y_pred) ** 2)
            r2 = 1 - rss / tss

            n_samples = y_test.size(0)
            adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - num_features - 1)

            # Add a small epsilon to avoid division by zero
            epsilon = 0.00000001
            n = len(y_test)
            abs_percentage_error = torch.abs(y_test - y_pred)
            abs_percentage_error =  abs_percentage_error/ (y_test + epsilon)
            #MAPE means predictions made by the model differ from the true values by approximately by x%
            mape = torch.mean(abs_percentage_error)* 100.0

            print("Evaluation MSE Loss on test Dataset: {:.6f}".format(mse_loss.item()))
            print("Evaluation RMSE Loss on test Dataset: {:.6f}".format(rmse_loss.item()))
            print("Evaluation R2 score on test Dataset: {:.6f}".format(r2.item()))
            print("Evaluation Adj_R2 score on test Dataset: {:.6f}".format(adj_r2.item()))
            print("Evaluation MAPE score on test Dataset: {:.6f}".format(mape.item()))



train_result = TrainPM86(epoch=1000,lr_rate = 0.001,train_test_ratio= 0.2)

train_result = train_result.trainpm86()

