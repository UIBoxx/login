import os
import torch
import pathlib
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from features_scaling import FeaturesScaling, min_max_scaling_groupby
from torcheval.metrics import MulticlassConfusionMatrix, MulticlassF1Score,  MulticlassPrecision, MulticlassRecall, MulticlassAccuracy 
from torcheval.metrics.functional import  binary_confusion_matrix, binary_accuracy, binary_f1_score, binary_precision, binary_recall


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

        input_features_numbers = len(min_max_scaling_groupby[0].axes[1])
        # print(input_features_numbers)
        input_features_normalized = min_max_scaling_groupby[0][0:6125]  # Take x data point for now
        # print(input_features_normalized)
        target_feature = min_max_scaling_groupby[1][0:6125]
        # print(target_feature)
        input_features_normalized_array = input_features_normalized.to_numpy()
        target_feature_array = target_feature.to_numpy()
        

        X = input_features_normalized_array
        y = target_feature_array
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.train_test_ratio, random_state=42)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)

        num_classes = len(np.unique(y))
        lr_rate = self.lr_rate
        input_size = X_train.size(1)
        hidden_size = X_train.size(1)
        output_size = num_classes
        batch_size = 32

        encoder = Encoder(input_size, hidden_size)
        decoder = Decoder(hidden_size, num_classes)
        model = Seq2SeqModelClassifier(input_size, hidden_size, output_size)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr_rate)
       
        # Training the model
        num_epochs = self.epoch_number
        for epoch in range(num_epochs):
            epoch_loss = 0.0  # Initialize loss for the epoch
            for i in range(0, len(X_train), batch_size):
                inputs = X_train[i:i + batch_size]
                labels = y_train[i:i + batch_size]
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()  # Accumulate loss for the epoch
            
            average_epoch_loss = epoch_loss / (len(X_train) // batch_size)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {average_epoch_loss:.6f}')
        
        torch.save(model.state_dict(), f'{self.train_models_dir}/seq2seq_lstm_classifier.pth')


        # Evaluate on the test dat
        model.eval()
        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                inputs = X_test[i:i + batch_size]
                labels = y_test[i:i + batch_size]
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                # print(f'Test Loss : {loss.item():.6f}')
            print(f'Test Loss: {loss.item():.6f}')


            output = model(X_test)
            probabilities = torch.softmax(output, dim=1)
            # print(probabilities)
            _, predicted_classes = torch.max(output, 1)
            # print(predicted_classes)

            if num_classes <= 2:
                #https://pytorch.org/torcheval/main/torcheval.metrics.html#
                confusion_matrix = binary_confusion_matrix(y_test, predicted_classes)
                print(f"confusion_matrix: {confusion_matrix}")
                accuracy_score = binary_accuracy(y_test, predicted_classes)
                print(f"accuracy_score: {accuracy_score}")
                precision_score = binary_precision(y_test, predicted_classes)
                print(f"precision_score: {precision_score}")
                recall_score = binary_recall(y_test, predicted_classes)
                print(f"recall_score: {recall_score}")
                f1_score = binary_f1_score(y_test, predicted_classes)
                print(f"f1_score: {f1_score}")

            else:
                #https://pytorch.org/torcheval/main/torcheval.metrics.html#

                metric = MulticlassConfusionMatrix(num_classes)
                metric.update(y_test, predicted_classes)
                confusion_matrix = metric.compute()
                confusion_matrix= confusion_matrix.int()
                print(f"confusion_matrix: {confusion_matrix}")

                metric = MulticlassAccuracy()
                metric.update(y_test, predicted_classes)
                accuracy_score = metric.compute()
                print(f"accuracy_score: {accuracy_score}")

                metric = MulticlassPrecision(num_classes=len(np.unique(y_test)), average='macro')
                metric.update(y_test, predicted_classes)
                precision_score = metric.compute()
                print(f"precision_score: {precision_score}")

                metric = MulticlassRecall(num_classes=len(np.unique(y_test)), average='macro')
                metric.update(y_test, predicted_classes)
                recall_score = metric.compute()
                print(f"recall_score: {recall_score}")

                metric = MulticlassF1Score(num_classes=len(np.unique(y_test)),average='macro')
                metric.update(y_test, predicted_classes)
                f1_score = metric.compute()
                print(f"f1_score: {f1_score}")



train_result = TrainPM86(epoch=100,lr_rate = 0.001,train_test_ratio= 0.2)

train_result = train_result.trainpm86()

