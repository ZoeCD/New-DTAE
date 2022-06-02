import unittest
import pandas as pd
import numpy as np
from DTAE import DTAE, Feature, analyzeDecisionPath
from pandas.util.testing import assert_frame_equal
from sklearn.preprocessing import OneHotEncoder
import arff
import re



class TestDTAE(unittest.TestCase):

    def test_valid_dataset(self):
        valid_X = pd.DataFrame(
            {'col1': ['1', '2', '1', '1', '2', '2', '1'],
             'col2': ['2', '1', '2', '1', '1', '2', '1'],
             'col3': ['1', np.nan, '1', np.nan, '2', '2', np.nan]})
        valid_y = pd.DataFrame({
            'class': ['1', '1', '1', '1', '1', '1', '1']
        })

        invalid_X = valid_X.values
        invalid_y = pd.DataFrame({
            'class': ['1', '2', '1', '1', '2', '1', '2']
        })

        dtae = DTAE()

        # Testing if raises error when not given pandas dataframes
        with self.assertRaises(TypeError):
            dtae._DTAE__check_valid_dataset(invalid_X, invalid_y)
            dtae._DTAE__check_valid_dataset(invalid_X, valid_y)

        # Testing if raises error when y has more than 1 class
        with self.assertRaises(ValueError):
            dtae._DTAE__check_valid_dataset(valid_X, invalid_y)

    def test_create_model(self):

        # Testing model creation without missing values
        y = pd.DataFrame({
            'class': ['1', '1', '1', '1', '1', '1', '1']
        })
        complete_X = pd.DataFrame(
            {'col1': ['1', '2', '1', '1', '2', '2', '1'],
             'col2': ['2', '1', '2', '1', '1', '2', '1'],
             'col3': ['A', 'C', 'A', 'A', 'C', 'C', 'A']})

        answer = [
            Feature('col1', 0, ['1', '2']),
            Feature('col2', 1, ['2', '1']),
            Feature('col3', 2, ['A', 'C'])
        ]

        dtae = DTAE()
        dtae._DTAE__create_model(complete_X)
        dtae_answer = dtae.model_features
        for i in range(len(dtae_answer)):
            self.assertEqual(answer[i].name, dtae_answer[i].name)
            for j in range(len(dtae_answer[i].values)):
                self.assertEqual(answer[i].values[j], dtae_answer[i].values[j])

        # Testing model creation with missing values
        incomplete_X = pd.DataFrame(
            {'col1': ['1', '2', '1', np.nan, '2', '2', '1'],
             'col2': [np.nan, '2', np.nan, '1', '1', '2', '1'],
             'col3': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
             'col4': ['A', 'C', 'A', 'A', 'C', np.nan, 'A']})

        answer2 = [
            Feature('col1', 0, ['1', '2']),
            Feature('col2', 1, ['2', '1']),
            Feature('col3', 2, [np.nan]),
            Feature('col4', 3, ['A', 'C'])
        ]

        dtae = DTAE()
        dtae._DTAE__create_model(incomplete_X)
        dtae_answer = dtae.model_features
        for i in range(len(dtae_answer)):
            self.assertEqual(answer2[i].name, dtae_answer[i].name)
            for j in range(len(dtae_answer[i].values)):
                self.assertEqual(answer2[i].values[j], dtae_answer[i].values[j])

    def test_impute_data(self):
        y = pd.DataFrame({
            'class': ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']})  # 13
        X = pd.DataFrame(
            {'col1': ['1', '2', '1', '1', '2', '2', '1', '1', '1', None, None, np.nan, np.nan],
             'col2': ['2', '1', '2', None, '1', '2', np.nan, '2', '2', '2', None, '1', '1'],
             'col3': [None, None, None, None, None, None, None, None, None, None, None, None, None],
             'col4': ['A', 'C', 'A', 'A', 'C', 'C', 'A', 'A', 'A', 'C', 'C', 'A', 'C']})

        dtae = DTAE()
        dtae._DTAE__create_model(X)
        dtae_answer = dtae._DTAE__handle_missing_data(X)
        dtae_answer = dtae._DTAE__clean_invalid_features(dtae_answer)
        self.assertFalse(dtae_answer.isnull().values.any())

    def test_create_encoder(self):
        y = pd.DataFrame({
            'class': ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']})  # 13

        X = pd.DataFrame(
            {'col1': ['1', '2', '1', '1', '2', '2', '1', '1', '1', '1', '1', '1', '1'],
             'col2': ['2', '1', '2', '2', '1', '2', '2', '2', '2', '2', '2', '1', '1'],
             'col3': ['A', 'C', 'A', 'A', 'C', 'C', 'A', 'A', 'A', 'C', 'C', 'A', 'C']})

        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False).fit(X)
        X_encoded = encoder.transform(X)
        dtae = DTAE()
        dtae._DTAE__create_model(X)
        dtae_X = dtae._DTAE__handle_missing_data(X)
        dtae._DTAE__set_encoder(dtae_X)
        dtae_encoder = dtae.get_encoder()
        dtae_X_encoded = dtae_encoder.transform(dtae_X)

        np.testing.assert_array_equal(X_encoded, dtae_X_encoded)

    def test_valid_features(self):

        y = pd.DataFrame({
            'class': ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']})  # 13
        X = pd.DataFrame(
            {'validFeature': ['1', '2', '1', '1', '2', '2', '1', '1', '1', '1', '2', np.nan, np.nan],
             'unvalidNumerical': [2, 1, 2, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1],
             'unvalidOneClass': ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A'],
             'unvalidFewInstances': ['A', 'C', 'A', 'B', 'A', 'A', 'B', 'D', 'A', 'A', 'A', 'A', 'A']})

        features = [
            Feature('validFeature', 0, ['1', '2']),
            Feature('unvalidNumerical', 1, [2, 1]),
            Feature('unvalidOneClass', 2, ['A']),
            Feature('unvalidFewInstances', 3, ['A', 'C', 'B', 'D'])
        ]
        dtae = DTAE()

        self.assertEqual(dtae._DTAE__check_valid_feature(X,features[0]), True)

        with self.assertRaises(TypeError):
            dtae._DTAE__check_valid_feature(X,features[1])

        self.assertEqual(dtae._DTAE__check_valid_feature(X, features[2]), False)

        self.assertEqual(dtae._DTAE__check_valid_feature(X, features[3]), False)

    def test_create_current_X(self):

        y = pd.DataFrame({
            'class': ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']})  # 13
        X = pd.DataFrame(
            {'col1': ['1', '2', '1', '1', '2', '2', '1', '1', '1', None, None, np.nan, np.nan],
             'col2': ['2', '1', '2', None, '1', '2', np.nan, '2', '2', '2', None, '1', '1'],
             'col3': ['A', 'C', 'A', 'A', 'C', 'C', 'A', 'A', 'A', 'C', 'C', 'A', 'C']})

        answer = [
            Feature('col1', 0, ['1', '2']),
            Feature('col2', 1, ['2', '1']),
            Feature('col3', 2, ['A', 'C'])
        ]

        X_complete = pd.DataFrame(
            {'col1': ['1', '2', '1', '1', '2', '2', '1', '1', '1', '2', '2', '1', '2'],
             'col2': ['2', '1', '2', '2', '1', '2', '2', '2', '2', '2', '1', '1', '1'],
             'col3': ['A', 'C', 'A', 'A', 'C', 'C', 'A', 'A', 'A', 'C', 'C', 'A', 'C']})

        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False).fit(X_complete)
        X_encoded = encoder.transform(X_complete)

        col1_ans = X_encoded[:9,2:]
        col2_ans = np.delete(X_encoded, [2, 3], axis=1)
        col2_ans = np.delete(col2_ans, [3, 6, 10], axis=0)
        col3_ans = X_encoded[:, :4]

        dtae = DTAE()
        dtae._DTAE__create_model(X)
        dtae_X = dtae._DTAE__handle_missing_data(X)
        dtae_X = dtae._DTAE__clean_invalid_features(dtae_X)
        dtae._DTAE__set_encoder(dtae_X)
        dtae_X_encoded = dtae._DTAE__encode_data(dtae_X)

        features = dtae.model_features
        dtae_col1_ans = dtae._DTAE__create_current_X_attributes(dtae_X_encoded, features[0])
        dtae_col2_ans = dtae._DTAE__create_current_X_attributes(dtae_X_encoded, features[1])
        dtae_col3_ans = dtae._DTAE__create_current_X_attributes(dtae_X_encoded, features[2])

        np.testing.assert_array_equal(dtae_col1_ans, col1_ans)
        np.testing.assert_array_equal(dtae_col2_ans, col2_ans)
        np.testing.assert_array_equal(dtae_col3_ans, col3_ans)

    def test_create_current_y(self):
        y = pd.DataFrame({
            'class': ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']})  # 13
        X = pd.DataFrame(
            {'col1': ['1', '2', '1', '1', '2', '2', '1', '1', '1', None, None, np.nan, np.nan],
             'col2': ['2', '1', '2', None, '1', '2', np.nan, '2', '2', '2', None, '1', '1'],
             'col3': ['A', 'C', 'A', 'A', 'C', 'C', 'A', 'A', 'A', 'C', 'C', 'A', 'C']})

        col1_ans = np.array(['1', '2', '1', '1', '2', '2', '1', '1', '1'])
        col2_ans = np.array(['2', '1', '2', '1', '2', '2', '2', '2', '1', '1'])
        col3_ans = np.array(['A', 'C', 'A', 'A', 'C', 'C', 'A', 'A', 'A', 'C', 'C', 'A', 'C'])


        dtae = DTAE()
        dtae._DTAE__create_model(X)
        dtae_X = dtae._DTAE__handle_missing_data(X)
        dtae_X = dtae._DTAE__clean_invalid_features(dtae_X)
        dtae._DTAE__set_encoder(dtae_X)

        dtae_X_encoded = dtae._DTAE__encode_data(dtae_X)
        features = dtae.model_features

        dtae_col1_ans = dtae._DTAE__create_current_y_class(dtae_X, features[0])
        dtae_col2_ans = dtae._DTAE__create_current_y_class(dtae_X, features[1])
        dtae_col3_ans = dtae._DTAE__create_current_y_class(dtae_X, features[2])

        np.testing.assert_array_equal(dtae_col1_ans, col1_ans)
        np.testing.assert_array_equal(dtae_col2_ans, col2_ans)
        np.testing.assert_array_equal(dtae_col3_ans, col3_ans)

    def test_check_if_trained(self):
        test = pd.DataFrame(
            {'col1': ['1', '2', '1', '1', '2', '2', '1', '1', '1', None, None, np.nan, np.nan],
             'col2': ['2', '1', '2', None, '1', '2', np.nan, '2', '2', '2', None, '1', '1'],
             'col3': ['A', 'C', 'A', 'A', 'C', 'C', 'A', 'A', 'A', 'C', 'C', 'A', 'C']})
        dtae = DTAE()
        with self.assertRaises(Exception):
            dtae.classify(test)

    def test_check_if_value_missing(self):
        X = pd.DataFrame(
            {'col1': ['1', '2', '1', '1', '2', '2', '1', '1', '1', None, None, np.nan, np.nan],
             'col2': ['2', '1', '2', None, '1', '2', np.nan, '2', '2', '2', None, '1', '1'],
             'col3': ['A', 'C', 'A', 'A', 'C', 'C', 'A', 'A', 'A', 'C', 'C', 'A', 'C']})

        dtae = DTAE()
        dtae._DTAE__create_model(X)
        dtae_X = dtae._DTAE__handle_missing_data(X)
        features = dtae.model_features
        self.assertEqual(dtae._DTAE__check_if_value_missing(10, features[0]), True)
        self.assertEqual(dtae._DTAE__check_if_value_missing(3, features[1]), True)
        self.assertEqual(dtae._DTAE__check_if_value_missing(0, features[2]), False)

    def test_create_current_instance(self):

        X = pd.DataFrame(
            {'col1': ['1', '2', '1', '1', '2', '2', '1', '1', '1', None, None, np.nan, np.nan],
             'col2': ['2', '1', '2', None, '1', '2', np.nan, '2', '2', '2', None, '1', '1'],
             'col3': ['A', 'C', 'A', 'A', 'C', 'C', 'A', 'A', 'A', 'C', 'C', 'A', 'C']})

        dtae = DTAE()
        dtae._DTAE__create_model(X)
        dtae_X = dtae._DTAE__handle_missing_data(X)
        dtae_X = dtae._DTAE__clean_invalid_features(dtae_X)
        dtae._DTAE__set_encoder(dtae_X)
        features = dtae.model_features

        X_test = pd.DataFrame(
            {'col1': [np.nan, '2', '2', '1', None, '2', '1', '2', '1', '1', '1',  '2', np.nan],
             'col2': ['2', None, '2', '2', '1', '2', '1', '2', '1', '2', '1', '2', '1'],
             'col3': ['A', 'C', 'C', 'A', 'A', 'C', 'A', 'C', 'A', 'C', 'A', 'A', 'C']})
        dtae_X_train = dtae._DTAE__handle_missing_data(X_test)
        dtae_X_train = dtae._DTAE__delete_invalid_features(dtae_X_train)
        dtae_X_encoded = [[1., 0., 0., 1., 1., 0.],
                           [0., 1., 0., 1., 0., 1.],
                           [0., 1., 0., 1., 0., 1.],
                           [1., 0., 0., 1., 1., 0.],
                           [1., 0., 1., 0., 1., 0.],
                           [0., 1., 0., 1., 0., 1.],
                           [1., 0., 1., 0., 1., 0.],
                           [0., 1., 0., 1., 0., 1.],
                           [1., 0., 1., 0., 1., 0.],
                           [1., 0., 0., 1., 0., 1.],
                           [1., 0., 1., 0., 1., 0.],
                           [0., 1., 0., 1., 1., 0.],
                           [1., 0., 1., 0., 0., 1.]]

        ans_instance_1 = [0., 1., 0., 1.]
        ans_instance_3 = [1., 0., 1., 0.]
        ans_instance_9 = [1., 0., 0., 1.]

        dtae_ans_1 = dtae._DTAE__create_current_instance(dtae_X_encoded[1], features[0])
        dtae_ans_3 = dtae._DTAE__create_current_instance(dtae_X_encoded[3], features[1])
        dtae_ans_9 = dtae._DTAE__create_current_instance(dtae_X_encoded[9], features[2])

        np.testing.assert_array_equal(dtae_ans_1, ans_instance_1)
        np.testing.assert_array_equal(dtae_ans_3, ans_instance_3)
        np.testing.assert_array_equal(dtae_ans_9, ans_instance_9)

    def test_classify_with_different_values_test_train(self):
        y = pd.DataFrame({
            'class': ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']})
        X = pd.DataFrame(
            {'col1': ['1', '2', '1', '1', '2', '2', '1', '1', '1', None, None, np.nan, np.nan],
             'col2': ['2', '1', '2', None, '1', '2', np.nan, '2', '2', '2', None, '1', '1'],
             'col3': ['A', 'C', 'A', 'A', 'C', 'C', 'A', 'A', 'A', 'C', 'C', 'A', 'C']})

        dtae = DTAE()
        dtae.train(X, y)

        y_test = pd.DataFrame({
            'class': ['1', '1', '0', '1', '1', '0', '1', '0', '1', '0', '1', '0', '1']})
        X_test = pd.DataFrame(
            {'col1': [np.nan, '2', '3', '1', None, '2', '1', '2', '3', '1', '1', '2', np.nan],
             'col2': ['2', None, '2', '2', '1', '2', '1', '2', '1', '2', '1', '2', '1'],
             'col3': ['A', 'C', 'C', 'A', 'A', 'C', 'A', 'C', 'A', 'C', 'A', 'A', 'C']})
        dtae.classify(X_test)

    def test_save_feature_attribute_names(self):

        y = pd.DataFrame({
            'class': ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']})  # 13
        X = pd.DataFrame(
            {'col1': ['1', '2', '1', '1', '2', '2', '1', '1', '1', None, None, np.nan, np.nan],
             'col2': ['2', '1', '2', None, '1', '2', np.nan, '2', '2', '2', None, '1', '1'],
             'col3': ['A', 'C', 'A', 'A', 'C', 'C', 'A', 'A', 'A', 'C', 'C', 'A', 'C']})

        dtae = DTAE()
        dtae.train(X, y)

        features = dtae.model_features

        np.testing.assert_array_equal(features[0].encoded_value_names, ['col2_1', 'col2_2', 'col3_A', 'col3_C'])
        np.testing.assert_array_equal(features[1].encoded_value_names, ['col1_1', 'col1_2', 'col3_A', 'col3_C'])
        np.testing.assert_array_equal(features[2].encoded_value_names, ['col1_1', 'col1_2', 'col2_1', 'col2_2'])


    def test_get_classifier_result(self):
        y = pd.DataFrame({
            'class': ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']})  # 13
        X = pd.DataFrame(
            {'col1': ['1', '2', '1', '1', '2', '2', '1', '1', '1', None, None, np.nan, np.nan],
             'col2': ['2', '1', '2', None, '1', '2', np.nan, '2', '2', '2', None, '1', '1'],
             'col3': [None, None, None, None, None, None, None, None, None, None, None, None, None],
             'col4': ['A', 'C', 'A', 'A', 'C', 'C', 'A', 'A', 'A', 'C', 'C', 'A', 'C']})

        dtae = DTAE()
        dtae.train(X, y)

        features = dtae._DTAE__valid_features

        y_test = pd.DataFrame({
            'class': ['1', '1', '0', '1', '1', '0', '1', '0', '1', '0', '1', '0', '1']})
        X_test = pd.DataFrame(
            {'col1': [np.nan, '2', '3', '1', None, '2', '1', '2', '3', '1', '1', '2', np.nan],
             'col2': ['2', None, '2', '2', '1', '2', '1', '2', '1', '2', '1', '2', '1'],
             'col3': [None, None, None, None, None, None, None, None, None, None, None, None, None],
             'col4': ['A', 'C', 'C', 'A', 'A', 'C', 'A', 'C', 'A', 'C', 'A', 'A', 'C']})

        instances = dtae._DTAE__handle_missing_data(X_test)
        instances = dtae._DTAE__delete_invalid_features(instances)
        encoded_instances = dtae._DTAE__encode_data(instances)

        current_instance = dtae._DTAE__create_current_instance(encoded_instances[0], features[0])
        answer = ['1', '2']
        dtae_answer = list(dtae._DTAE__get_classifier_result(0, current_instance).keys())
        np.testing.assert_array_equal(dtae_answer, answer)

        current_instance = dtae._DTAE__create_current_instance(encoded_instances[0], features[1])
        answer = ['1', '2']
        dtae_answer = list(dtae._DTAE__get_classifier_result(1, current_instance).keys())
        np.testing.assert_array_equal(dtae_answer, answer)

        current_instance = dtae._DTAE__create_current_instance(encoded_instances[0], features[2])
        answer = ['A', 'C']
        dtae_answer = list(dtae._DTAE__get_classifier_result(2, current_instance).keys())
        np.testing.assert_array_equal(dtae_answer, answer)

    def test_classifier_depth_grater_than_cero(self):
        with open('TrainingDatasetNames.txt', 'r', encoding='UTF-8') as file:
            for line in file:
                line = line.split('\n')
                file = line[0].split(".")
                file_name = "/Users/zoe/PycharmProjects/New-DTAE/OCC Categorical Datasets/" \
                            "OCC Categorical Datasets/" + file[0] + "/" + line[0]
                train_file = open(file_name, "r")
                train_dataset = arff.load(train_file)
                train_file.close()
                feature_names = list()
                for attribute in train_dataset['attributes']:
                    feature_names.append(attribute[0])

                train_dataset = pd.DataFrame(train_dataset['data'], columns=feature_names)
                X_train = train_dataset.iloc[:, 0:train_dataset.shape[1] - 1]
                y_train = train_dataset.iloc[:, train_dataset.shape[1] - 1: train_dataset.shape[1]]
                dtae = DTAE()
                dtae.train(X_train, y_train)
                for classifier in dtae._DTAE__classifiers:
                    self.assertGreater(classifier.get_depth(), 0)

    def test_decision_path_reduction(self):

        # Simple case - no reduction
        path = "If (age == adult) then {'dip': '100.0%', 'stretch': '0.0%'}"
        features = [Feature("age", 0, ["adult", "child"])]
        answer = "If (age == adult) then {'dip': '100.0%', 'stretch': '0.0%'}"
        new_path = analyzeDecisionPath(path, features)
        self.assertEqual(new_path, answer)

         # Two features - no reduction
        path = "If (age == adult) AND (color == yellow) then {'dip': '100.0%', 'stretch': '0.0%'}"
        features = [Feature("age", 0, ["adult","child"]), Feature("color", 1, ["purple","yellow"])]
        answer = "If (age == adult) AND (color == yellow) then {'dip': '100.0%', 'stretch': '0.0%'}"
        new_path = analyzeDecisionPath(path, features)
        self.assertEqual(new_path, answer)

        # Simple case - reduction
        path = "If (color != yellow) AND (color != purple) then {'dip': '100.0%', 'stretch': '0.0%'}"
        features = [Feature("color", 0, ["purple", "yellow","pink"])]
        answer = "If (color != yellow, purple) then {'dip': '100.0%', 'stretch': '0.0%'}"
        new_path = analyzeDecisionPath(path, features)
        self.assertEqual(new_path, answer)

        # Multiple case - reduction
        path = "If (color != yellow)  AND (size != big) AND (size != medium) AND (color != purple) then {'dip': '100.0%', 'stretch': '0.0%'}"
        features = [Feature("color", 0, ["purple", "yellow", "pink"]), Feature("size", 1, ["big", "medium", "small"])]
        answer = "If (color != yellow, purple) AND (size != big, medium) then {'dip': '100.0%', 'stretch': '0.0%'}"
        new_path = analyzeDecisionPath(path, features)
        self.assertEqual(new_path, answer)

    def test_decision_path_reduction_less_features_in_path(self):

        # No reduction
        path = "If (age == adult) AND (color == yellow) then {'dip': '100.0%', 'stretch': '0.0%'}"
        features = [Feature("age", 0, ["adult", "child"]), Feature("color", 1, ["purple", "yellow"]), Feature("size", 2, ["big", "small"])]
        answer = "If (age == adult) AND (color == yellow) then {'dip': '100.0%', 'stretch': '0.0%'}"
        new_path = analyzeDecisionPath(path, features)
        self.assertEqual(new_path, answer)

        # With reduction
        path = "If (color != yellow) AND (color != purple) AND (age != adult) then {'dip': '100.0%', 'stretch': '0.0%'}"
        features = [Feature("color", 0, ["purple", "yellow", "pink"]), Feature("age", 1, ["adult", "child"]), Feature("size", 2, ["big", "small"])]
        answer = "If (color != yellow, purple) AND (age != adult) then {'dip': '100.0%', 'stretch': '0.0%'}"
        new_path = analyzeDecisionPath(path, features)
        self.assertEqual(new_path, answer)

    def test_decision_path_reduction_audiology_paths(self):
        train_file = open('OCC Categorical Datasets/audiology/audiology.training1.arff', "r")
        train_dataset = arff.load(train_file)
        train_file.close()
        feature_names = list()
        for attribute in train_dataset['attributes']:
            feature_names.append(attribute[0])

        train_dataset = pd.DataFrame(train_dataset['data'], columns=feature_names)

        X_train = train_dataset.iloc[:, 0:train_dataset.shape[1] - 1]
        y_train = train_dataset.iloc[:, train_dataset.shape[1] - 1:train_dataset.shape[1]]

        dtae = DTAE()
        dtae._DTAE__check_valid_dataset(X_train, y_train)
        dtae._DTAE__create_model(X_train)
        X = dtae._DTAE__handle_missing_data(X_train)
        X = dtae._DTAE__clean_invalid_features(X)

        features = dtae._DTAE__valid_features

        audiology_paths = ["If (speech != normal) AND (air != mild) then {'f': '78.38%', 't': '21.62%'} ",
                           "If (bone != mild) AND (age_gt_60 != f) AND (speech != normal) then {'mild': '74.07%', 'moderate': '11.11%', 'normal': '14.81%', 'profound': '0.0%', 'severe': '0.0%'}",
                           "If (tymp == a) then {'f': '99.28%', 't': '0.72%'}",
                           "If (o_ar_c != absent) AND (o_ar_u != absent) AND (o_ar_c != elevated) then {'absent': '0.0%', 'elevated': '10.11%', 'normal': '89.89%'}",
                           "If (air != normal) AND (o_ar_c != absent) AND (age_gt_60 != f) AND (history_dizziness != f) then {'mild': '66.67%', 'moderate': '33.33%', 'normal': '0.0%', 'unmeasured': '0.0%'}",
                           "If (notch_at_4k != t) AND (age_gt_60 != f) AND (notch_4k != t) then {'f': '59.38%', 't': '40.62%'}",
                           "If (ar_c != absent) AND (ar_c != elevated) then {'absent': '3.26%', 'elevated': '9.78%', 'normal': '86.96%'}",
                           "If (ar_c != absent) AND (o_ar_c == normal) then {'absent': '1.11%', 'elevated': '13.33%', 'normal': '85.56}%'"
                          ]

        answer_paths = ["If (air != mild) AND (speech != normal) then {'f': '78.38%', 't': '21.62%'}",
                        "If (age_gt_60 != f) AND (bone != mild) AND (speech != normal) then {'mild': '74.07%', 'moderate': '11.11%', 'normal': '14.81%', 'profound': '0.0%', 'severe': '0.0%'}",
                        "If (tymp == a) then {'f': '99.28%', 't': '0.72%'}",
                        "If (o_ar_c != absent, elevated) AND (o_ar_u != absent) then {'absent': '0.0%', 'elevated': '10.11%', 'normal': '89.89%'}",
                        "If (age_gt_60 != f) AND (air != normal) AND (history_dizziness != f) AND (o_ar_c != absent) then {'mild': '66.67%', 'moderate': '33.33%', 'normal': '0.0%', 'unmeasured': '0.0%'}",
                        "If (age_gt_60 != f) AND (notch_4k != t) AND (notch_at_4k != t) then {'f': '59.38%', 't': '40.62%'}",
                        "If (ar_c != absent, elevated) then {'absent': '3.26%', 'elevated': '9.78%', 'normal': '86.96%'}",
                        "If (ar_c != absent) AND (o_ar_c == normal) then {'absent': '1.11%', 'elevated': '13.33%', 'normal': '85.56}"
                        ]

        for i in range(len(audiology_paths)):
            audiology_path = audiology_paths[i]
            answer = answer_paths[i]
            new_path = analyzeDecisionPath(audiology_path, features)
            self.assertEqual(new_path, answer)


