import os
import glob
import json
import pathlib
import pandas as pd                                     
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

class FeaturesScaling:

    def __init__(self , csv_name,  groupby_column, selected_columns, target_column):

        self.csvfile = f"../../../input_datasets/{csv_name}"
        self.groupby_column =  groupby_column

        self.selected_columns = selected_columns

        self.target_column = target_column

        self.scaling_file = f"../models/scalingfile"

        self.script_dir = pathlib.Path(__file__).parent.absolute()

        self.input_csvfile = os.path.join(self.script_dir, self.csvfile)
        self.scaling_file_dir = os.path.join(self.script_dir, self.scaling_file)


    def min_max_scaling(self):

        """Normalization( min_max_scaling) is the process of rescaling data in new range of 0 and 1.
        Normalization is good to use when our data does not follow a normal distribution(Gaussian distribution).
        Based on scaled_x = (x - x_min) / (x_max - x_min)"""

        dataset = pd.read_csv(self.input_csvfile)
        selected_columns = self.selected_columns
        target_column = self.target_column
        target_column = dataset[target_column]
        num_classes = target_column.nunique()

        ###### Save scaling_json_data ######
        df = dataset.astype(float)
        feature_data_list = []
        for feature in selected_columns:
            feature_dict = {
                "feature_columns": feature,
                "max": float(df[feature].max()),
                "min": float(df[feature].min()),
                "num_classes" : num_classes}
            feature_data_list.append(feature_dict)
        with open(f"{self.scaling_file_dir}/scaling_json_data.json", "w") as f:
            json.dump(feature_data_list, f)
        ###########################################

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(dataset[selected_columns])

        scaled_df = pd.DataFrame(scaled_data, columns=selected_columns)

        # dataset[selected_columns] = scaled_df[selected_columns]

        # print(dataset)
        return scaled_df, target_column


    def min_max_scaling_groupby(self):

        dataset = pd.read_csv(self.input_csvfile)
        selected_columns = self.selected_columns
        groupby_column = self.groupby_column
        target_column = self.target_column
        target_column = dataset[target_column]

        ###### Save scaling_json_data ######
        df = dataset.astype(float)
        grouped_data = df.groupby(groupby_column)
        scaling_data_list = []

        for groupby_id, groupby_data in grouped_data:
            num_classes = target_column.nunique()
            scaling_dict = {
                "groupby_id": groupby_id,
                "num_classes" : num_classes,
                "scaling_data": []
            }
            for feature in selected_columns:
                scaling_data = {
                    "feature_name": feature,
                    "min": float(groupby_data[feature].min()),
                    "max": float(groupby_data[feature].max())
                }
                scaling_dict["scaling_data"].append(scaling_data)
            scaling_data_list.append(scaling_dict)

        with open(f"{self.scaling_file_dir}/scaling_json_data.json", "w") as f:
            json.dump(scaling_data_list, f)

        ####################################

        scaler = MinMaxScaler()
        grouped = dataset.groupby(groupby_column)

        scaled_data = []

        for id, group in grouped:
            group[selected_columns] = scaler.fit_transform(group[selected_columns])
            scaled_data.append(group)

        scaled_data = pd.concat(scaled_data, ignore_index=True)
        scaled_df = pd.DataFrame(scaled_data, columns=selected_columns)

        # print(dataset)
        return scaled_df , target_column


    def standard_scaling(self):

        """Standardization is good to use when our data follow the normal distribution(Gaussian distribution) and 
        If the original data follows a normal distribution then scaled values will fall within the range of -3 to 3.
        StandardScaler can often give misleading results when the data contain outliers. 
        Outliers can often influence the sample mean and variance and hence give misleading results so first we need to handel outliers.
        Based on Z score = (x -mean) / std. deviation"""

        dataset = pd.read_csv(self.input_csvfile)
        selected_columns = self.selected_columns
        target_column = self.target_column
        target_column = dataset[target_column]
        num_classes = target_column.nunique()


        ###### Save scaling_json_data ######
        df = dataset.astype(float)
        feature_data_list = []
        for feature in selected_columns:
            feature_dict = {
                "feature_column": feature,
                "mean": float(df[feature].mean()),
                "std_deviation": float(df[feature].std()),
                "num_classes" : num_classes}            
            feature_data_list.append(feature_dict)
        with open(f"{self.scaling_file_dir}/scaling_json_data.json", "w") as f:
            json.dump(feature_data_list, f)
        ###########################################

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(dataset[selected_columns])

        scaled_df = pd.DataFrame(scaled_data, columns=selected_columns)
        # dataset[selected_columns] = scaled_df[selected_columns]

        # print(scaled_df)
        return scaled_df, target_column
    

    def standard_scaling_groupby(self,groupby_column):

        dataset = pd.read_csv(self.input_csvfile)
        selected_columns = self.selected_columns
        groupby_column = self.groupby_column
        target_column = self.target_column
        target_column = dataset[target_column]

        ###### Save scaling_json_data ######
        df = dataset.astype(float)
        grouped_data = df.groupby(groupby_column)
        scaling_data_list = []
        for groupby_id, groupby_data in grouped_data:
            num_classes = target_column.nunique()
            scaling_dict = {
                "groupby_id": groupby_id,
                "num_classes" : num_classes,
                "scaling_data": []
            }
            for feature in selected_columns:
                scaling_data = {
                    "feature_name": feature,
                    "mean": float(groupby_data[feature].mean()),
                    "std_deviation": float(groupby_data[feature].std())
                }
                scaling_dict["scaling_data"].append(scaling_data)
            scaling_data_list.append(scaling_dict)

        with open(f"{self.scaling_file_dir}/scaling_json_data.json", "w") as f:
            json.dump(scaling_data_list, f)

        ####################################

        scaler = StandardScaler()
        grouped = dataset.groupby(groupby_column)

        scaled_data = []

        for id, group in grouped:
            group[selected_columns] = scaler.fit_transform(group[selected_columns])
            scaled_data.append(group)

        scaled_data = pd.concat(scaled_data, ignore_index=True)
        scaled_df = pd.DataFrame(scaled_data, columns=selected_columns)
        # print(scaled_df)
        return scaled_df,  target_column
    
    def robust_scaling(self):
        """StandardScaler can often give misleading results when the data contain outliers.
        Outliers can often influence the sample mean and variance and hence give misleading results.
         In such cases, it is better to use a scalar that is robust against outliers.
         Based on X_robust = (x - Median(x))/(75thPercentile(x)- 25thPercentile(x))"""

        dataset = pd.read_csv(self.input_csvfile)
        selected_columns = self.selected_columns
        target_column = self.target_column
        target_column = dataset[target_column]
        num_classes = target_column.nunique()

        ###### Save scaling_json_data ######
        df = dataset.astype(float)
        feature_data_list = []
        for feature in selected_columns:
            feature_dict = {
                "feature_column": feature,
                "median": float(df[feature].median()),
                "75th_percentile": float(df[feature].quantile(0.75)),
                "25th_percentile": float(df[feature].quantile(0.25)),
                "num_classes" : num_classes}
            feature_data_list.append(feature_dict)
        with open(f"{self.scaling_file_dir}/scaling_json_data.json", "w") as f:
            json.dump(feature_data_list, f)
        ############################################

        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(dataset[selected_columns])

        scaled_df = pd.DataFrame(scaled_data, columns=selected_columns)
        # dataset[selected_columns] = scaled_df[selected_columns]

        # print(scaled_df )
        return scaled_df,  target_column
    

    def robust_scaling_groupby(self, groupby_column):

        dataset = pd.read_csv(self.input_csvfile)
        selected_columns = self.selected_columns
        groupby_column = self.groupby_column
        target_column = self.target_column
        target_column = dataset[target_column]

        ###### Save scaling_json_data ######
        df = dataset.astype(float)
        grouped_data = df.groupby(groupby_column)
        scaling_data_list = []
        for groupby_id, groupby_data in grouped_data:
            num_classes = target_column.nunique()
            scaling_dict = {
                "groupby_id": groupby_id,
                "num_classes" : num_classes,
                "scaling_data": []
            }
            for feature in selected_columns:
                scaling_data = {
                    "feature_name": feature,
                    "median": float(groupby_data[feature].median()),
                    "75th_percentile": float(groupby_data[feature].quantile(0.75)),
                    "25th_percentile": float(groupby_data[feature].quantile(0.25))
                }
                scaling_dict["scaling_data"].append(scaling_data)
            scaling_data_list.append(scaling_dict)

        with open(f"{self.scaling_file_dir}/scaling_json_data.json", "w") as f:
            json.dump(scaling_data_list, f)
        ####################################################

        scaler = RobustScaler()
        grouped = dataset.groupby(groupby_column)

        scaled_data = []

        for id, group in grouped:
            group[selected_columns] = scaler.fit_transform(group[selected_columns])
            scaled_data.append(group)

        scaled_data = pd.concat(scaled_data, ignore_index=True)
        scaled_df = pd.DataFrame(scaled_data, columns=selected_columns)
        # print(scaled_df)
        return scaled_df,  target_column


scaling = FeaturesScaling(csv_name = "train_clf_FD004.csv",
                        groupby_column='engine_ID',
                        selected_columns = ['cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3', 'T2', 'T24', 'T30', 'T50', 'P2',
                            'P15', 'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd',
                            'PCNfR_dmd', 'W31', 'W32'],
                        target_column = "engines_condition")

# min_max= scaling.min_max_scaling()
# print(min_max)

min_max_scaling_groupby = scaling.min_max_scaling_groupby()
# print(min_max_scaling_groupby)
# standard = scaling.standard_scaling()

# standard_groupby = scaling.standard_scaling_groupby()

# robust = scaling.robust_scaling()

# robust_groupby = scaling.robust_scaling_groupby()

