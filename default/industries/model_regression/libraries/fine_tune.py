import os
import torch
import pathlib
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from features_scaling import FeaturesScaling, min_max_scaling_groupby


#Define the Encoder class
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


class  FineTuneCustomData:
    "Prepare data  for training "

    def __init__(self, epoch, lr_rate):
       
        self.epoch_number = epoch
        self.lr_rate = lr_rate

        self.train_model_dir = "../models/train"
        self.active_model_dir= "../models/active/seq2seq_model.pth"
        self.script_dir = pathlib.Path(__file__).parent.absolute()
        self.active_models_dir = os.path.join(self.script_dir, self.active_model_dir)
        self.train_models_dir = os.path.join(self.script_dir, self.train_model_dir)


    def make_fine_tune(self):

        # scaling = FeaturesScaling()

        # min_max_scaling_groupby = min_max_scaling_groupby
        # print(min_max_scaling_groupby)

        num_features = len(min_max_scaling_groupby.axes[1]) -1  # 1 is output features from above
        # print(num_features)

        if min_max_scaling_groupby is not None:

            normalized_data =  min_max_scaling_groupby[0:100]
            data_array = normalized_data.to_numpy()

            X = data_array[:, :-1]  # Features
            y = data_array[:, -1]   # target value
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


            # Convert the numpy arrays to torch tensors
            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32)
            X_test = torch.tensor(X_test, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.float32)


            input_size = num_features # Number of input features
            hidden_size = num_features# Number of hidden units in the LSTM layers
            output_size = 1  # Size of the output (hot index)
            lr_rate = self.lr_rate

            # Create an instance of the Seq2SeqModel
           
            model = Seq2SeqModel(input_size, hidden_size, output_size)
            model.load_state_dict(torch.load(self.active_models_dir))

            # Define a loss function (MSE loss) and an optimizer (e.g., Adam)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr =self.lr_rate)

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
                print(f"Epoch [{epoch + 1}/{num_epochs}], Training MSE Loss: {mse_loss.item():.6f}, "
                      f"Training RMSE Loss: {rmse_loss.item():.6f}")

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
            epsilon = 1e-8
            abs_percentage_error = torch.abs((y_test - y_pred) / (y_test + epsilon))
            mape = torch.mean(abs_percentage_error) * 100.0

            print("Evaluation MSE Loss on test Dataset: {:.6f}".format(mse_loss.item()))
            print("Evaluation RMSE Loss on test Dataset: {:.6f}".format(rmse_loss.item()))
            print("Evaluation R2 score on test Dataset: {:.6f}".format(r2.item()))
            print("Evaluation Adj_R2 score on test Dataset: {:.6f}".format(adj_r2.item()))
            print("Evaluation MAPE score on test Dataset: {:.6f}".format(mape.item()))




fine_tune = FineTuneCustomData(epoch=10000,lr_rate = 0.001,)

rul = fine_tune.make_fine_tune()