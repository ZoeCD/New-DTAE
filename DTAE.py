import numpy as np
import pandas as pd
from functools import reduce
from sklearn import tree
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import _tree
from colorama import Fore, Back, Style
from sklearn.neighbors import KNeighborsClassifier
from PBC4cip.core.Dataset import PandasDataset, FileDataset
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier



class DTAE():
    def __init__(self):
        self.__is_trained = False
        self.__feature_class = None
        self.__valid_features = list()
        self.__classifiers = list()
        self.__features_weights = list()
        self.__feature_value_weights = list()
        self.__model = None
        self.__encoder = None
        self.__attribute_names_by_feature = list()
        self.__tree_depth= None
        self.FEATURE_NAME = 0
        self.FEATURE_VALUES = 1
        self.model_features = []
        self.test_model_features = []

    def get_model(self):
        return self.__model

    def __set_model(self, X, y):
        self.__model = PandasDataset(X, y)

    def get_encoder(self):
        return self.__encoder

    def __set_encoder(self, X):
        self.__encoder = OneHotEncoder(handle_unknown="ignore", sparse=False).fit(X)

    def get_classifiers(self):
        return self.__classifiers

    def get_feature_weights(self):
        return self.__features_weights

    def get_feature_value_weights(self):
        return self.__feature_value_weights

    def get_atribute_names_by_feature(self, feature):
        feature_index = self.__model.GetFeatureIdx(feature)
        return self.__attribute_names_by_feature[feature_index]

    def __encode_data(self, X):
        return self.__encoder.transform(X)

    def __check_feature_nominal(self, feature):
        if feature.type != 'Nominal':
            raise TypeError(f"Unable to train {feature}: All features must be of type Nominal")

    def __check_enough_instances(self, X, feature):
        return True if(np.sum(X[feature.name].value_counts() > 2) >= 2) else False

    def __check_valid_feature(self, X, feature):
        self.__check_feature_nominal(feature)
        return self.__check_enough_instances(X, feature)

    def __save_feature_attribute_names(self,feature, column_count, number_of_values):
        attribute_names = np.delete(self.__encoder.get_feature_names_out(), np.s_[column_count:column_count+number_of_values])
        self.__attribute_names_by_feature.append(attribute_names)
        feature.set_encoded_value_names(attribute_names)

    def __create_current_X_attributes(self, X, feature):
        column_start = 0
        for f in self.model_features[:feature.index]:
            column_start += len(f.values)

        feature_value_len = len(feature.values)
        X_train = np.delete(X, feature.missing_value_objects, axis=0)
        X_train = np.delete(X_train, np.s_[column_start:column_start+feature_value_len], axis=1)
        self.__save_feature_attribute_names(feature, column_start, feature_value_len)
        return X_train

    def __create_current_y_class(self, X, feature):
        current_X = X.drop(feature.missing_value_objects, axis = 0, inplace=False)
        return current_X[feature.name].to_numpy()

    def __create_classification_tree(self, X, y):
        classifier = tree.DecisionTreeClassifier(random_state=0,max_depth=self.__tree_depth)
        classifier = classifier.fit(X,y)
        return classifier

    def __make_cross_validation_matrix(self, X, y):
        sampler = StratifiedKFold(n_splits=5)
        feature_values = np.unique(y)
        result_confusion_matrix = np.zeros((len(feature_values), len(feature_values)))

        for train_index, test_index in sampler.split(X, y):
            x_train_fold, x_test_fold = X[train_index], X[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]

            classifier = self.__create_classification_tree(x_train_fold,y_train_fold)
            y_pred_scores = np.reshape(classifier.predict(x_test_fold), (-1, 1))
            confusion_fold = confusion_matrix(y_test_fold, y_pred_scores, labels=feature_values)
            result_confusion_matrix = np.add(result_confusion_matrix ,confusion_fold)

        return result_confusion_matrix

    def __calculate_feature_value_weights(self, confusion_matrix):
        row_count = len(confusion_matrix[0])
        weight_by_feature_value = list()

        for i in range(row_count):
            weight_by_feature_value.append(reduce(lambda x, y:x+y, confusion_matrix[i]))

        totalSum = reduce(lambda x, y:x+y, weight_by_feature_value)

        for i in range(len(weight_by_feature_value)):
            weight_by_feature_value[i] /= totalSum
        return weight_by_feature_value

    def __calculate_weights(self, feature, X, y):
        current_confusion_matrix = self.__make_cross_validation_matrix(X,y)
        auc = calculate_auc(current_confusion_matrix, values_count=len(feature.values))
        feature_values_weights = self.__calculate_feature_value_weights(current_confusion_matrix)
        return auc, feature_values_weights

    def __train_feature_tree(self, X_encoded, X, feature):
        X_current = self.__create_current_X_attributes(X_encoded, feature)
        y_current = self.__create_current_y_class(X, feature)

        feature_weight, feature_values_weights = self.__calculate_weights(feature, X_current, y_current)
        classifier = self.__create_classification_tree(X_current, y_current)

        self.__features_weights.append(feature_weight)
        self.__feature_value_weights.append(feature_values_weights)
        self.__classifiers.append(classifier)
        self.__valid_features.append(feature)

    def __train_trees(self, X, y):
        X_encoded = self.__encode_data(X)
        for feature in self.model_features:
            if self.__check_valid_feature(X, feature):
                self.__train_feature_tree(X_encoded, X, feature)

    def __check_valid_dataset(self, X, y):
        unique_values = np.unique(y)

        if not (isinstance(X, pd.DataFrame)) or not (isinstance(y, pd.DataFrame)):
            raise TypeError("Variables X and y should be a Pandas Dataframe")

        if len(unique_values) > 1:
            raise ValueError(
                f"Unable to train {unique_values}: The training dataset must contain objects of a single "
                f"class!")

    def __create_model(self, X):
        for i in range(len(X.columns)):
            feature = X.columns[i]
            name = feature
            values = X[feature].unique()

            if pd.isnull(values).any():
                values = values[
                    ~pd.isnull(values)]
            self.model_features.append(ModelFeature(name, i, values))

    def __handle_missing_data(self, X):
        for feature in self.model_features:
            if X[feature.name].isnull().any():
                X = self.impute_data(X,feature)
        data = calculate_imputation(self.model_features, X)
        return X

    def impute_data(self, X, feature):
        missing_value_objects_indexes = X[X[feature.name].isnull()].index.to_list()
        feature.set_missing_value_objects(missing_value_objects_indexes)
        return X

    def __check_variability(self):
        if len(self.__classifiers) == 0:
            raise Exception("Unable to train: Not enough variability of the features")

    def train(self, X, y):
        self.__check_valid_dataset(X, y)
        self.__create_model(X)
        X = self.__handle_missing_data(X)
        self.__set_encoder(X)
        self.__train_trees(X,y)
        self.__check_variability()
        self.__is_trained = True

    def __check_if_trained(self):
        if not self.__is_trained:
            raise Exception("Unable to classify: Untrained classifier!")

    def __check_if_value_missing(self, instance_index, feature):
        return True if instance_index in feature.missing_value_objects else False

    def __get_weights(self,feature_index, actual_value):
        feature_weight = self.__features_weights[feature_index]
        actual_feature_value_weight = self.__feature_value_weights[feature_index][actual_value]
        return feature_weight, actual_feature_value_weight

    def __get_classifier_classes(self, feature_index):
        return self.__classifiers[feature_index].classes_

    def __get_classifier_result(self, feature_index, instance):
        classifier = self.__classifiers[feature_index]
        classifier_results = classifier.predict_proba(np.reshape(instance,(1,-1)))[0]
        classifier_classes = classifier.classes_
        probabilities = {}
        for value, prob in zip(classifier_classes, classifier_results):
            probabilities[value] = prob
        return probabilities

    def __sum_outlier(self, classifier_result, actual_value, feature_index):
            current_sum_outlier = 0
            count_outlier_votes = 0
            score_outlier = 0

            for key, j in zip(classifier_result.keys(), range(len(classifier_result))):
                if key != actual_value and classifier_result[key] > 0:
                    current_sum_outlier += self.__feature_value_weights[feature_index][j] * classifier_result[key]
                    count_outlier_votes += 1

            if count_outlier_votes > 0:
                score_outlier = self.__features_weights[feature_index] * current_sum_outlier/count_outlier_votes

            return score_outlier

    def __create_current_instance(self, instance, feature):
        column_start = 0
        for f in self.model_features[:feature.index]:
            column_start += len(f.values)
        feature_value_len = len(feature.values)
        current_instance = np.delete(instance, np.s_[column_start:column_start+feature_value_len])
        return current_instance

    def __calculate_classification_scores(self, feature_index, instance, encoded_instance):
        feature = self.__valid_features[feature_index]
        current_instance = self.__create_current_instance(encoded_instance, feature)
        classifier_results = self.__get_classifier_result(feature_index, current_instance)
        actual_feature_value = np.where(feature.values == instance[feature.index])[0]

        if not actual_feature_value.size == 0:
            actual_feature_value = actual_feature_value[0]
            feature_weight, actual_feature_value_weight = self.__get_weights(feature_index, actual_feature_value)
            score_normal = feature_weight * actual_feature_value_weight * classifier_results[instance[feature.index]]
            score_outlier = self.__sum_outlier(classifier_results, instance[feature.index], feature_index)
        else:
            score_normal, score_outlier = 0,0

        return score_normal, score_outlier

    def __classify_per_feature(self,  instance_index, instance, encoded_instance):
        score_normal, score_outlier = 0,0
        for feature_index in range(len(self.__valid_features)):
            if not self.__check_if_value_missing(instance_index, self.__valid_features[feature_index]):
                normal, outlier = self.__calculate_classification_scores(feature_index, instance, encoded_instance)
                score_outlier += outlier
                score_normal += normal
        return score_normal, score_outlier

    def __classify_instance(self, instance_index, instance, encoded_instance):
        score_normal, score_outlier = self.__classify_per_feature(instance_index, instance, encoded_instance)
        total_score = score_normal + score_outlier
        if total_score > 0:
            return [score_normal-score_outlier]
        else:
            return [0.0]

    def classify(self, instances):
        self.__check_if_trained()
        instances = self.__handle_missing_data(instances)
        encoded_instances = self.__encode_data(instances)
        results = list()
        for i in range(len(instances.values)):
            results.append(self.__classify_instance(i, instances.values[i], encoded_instances[i]))

        return results

    def __impute_data_instance(self, instance):
        for feature in self.model_features:
            if instance.iloc[:,feature.index].isnull().any():
                instance.iloc[:,feature.index] = feature.get_imputation_value()
        return instance

    def classify_and_interpret(self, instance):
        print(f"Input: {instance}")
        print(f"Output: ")
        instance = pd.DataFrame(np.reshape(instance,(1,-1)))
        instance_imputed = self.__impute_data_instance(instance.copy())
        encoded_instance = self.__encoder.transform(instance_imputed)

        score_normal, score_outlier = 0,0
        for feature_index in range(len(self.__valid_features)):
            feature = self.__valid_features[feature_index]
            if not pd.isnull(instance[feature.index][0]):
                print(Style.BRIGHT + f"Feature: {feature.name}")
                print(Style.RESET_ALL)

                normal, outlier = self.__calculate_classification_scores(feature_index, instance_imputed.values[0], encoded_instance)
                score_outlier += outlier
                score_normal += normal

                current_instance = self.__create_current_instance(encoded_instance, feature)
                classifier = self.__classifiers[feature_index]
                classifier_result = classifier.predict(np.reshape(current_instance,(1,-1)))[0]

                real_value = instance.iloc[:,feature.index][0]


                print(f"Real value: {real_value}")
                print(f"Prediction: {classifier_result}")
                print(f"Score normal: {normal}")
                print(f"Score outlier: {outlier}")

                self.__print_rules(classifier, 0, current_instance.reshape(1, -1), classifier_result, real_value, feature.encoded_value_names,feature.values)
        total_score = score_normal + score_outlier
        if total_score > 0:
            print(f"Classification score: {score_normal-score_outlier}")
        else:
            print(f"Classification score: {0.0}")

    def __print_rules(self, clf, sample_id, X_test, prediction, real, feature_names,class_names):
        '''By:  https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html'''

        tree_ = clf.tree_
        feature_name = [
                feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
                for i in tree_.feature
        ]

        feature = clf.tree_.feature
        threshold = clf.tree_.threshold

        node_indicator = clf.decision_path(X_test)
        leaf_id = clf.apply(X_test)

        # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
        node_index = node_indicator.indices[
            node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
        ]

        if real == prediction:
            color = Fore.GREEN
        else:
            color = Fore.RED

        values_leaf = tree_.value[leaf_id[sample_id]]
        dir = {}
        for v in range(len(values_leaf[0])):
            dir[class_names[v]] = str(np.round(100.0*values_leaf[0][v]/np.sum(values_leaf[0]),2)) + '%'

        print(color + "If ", end="")

        for node_id in node_index:
            # continue to the next node if it is a leaf node
            if leaf_id[sample_id] == node_id:
                continue

            # check if value of the split feature for sample 0 is below threshold
            if X_test[sample_id, feature[node_id]] == 1.0:
                threshold_value = feature_name[node_id].split('_')
                threshold_decision = "=="
                threshold_name = threshold_value[0]
                threshold_value = ' '.join([str(elem) for elem in threshold_value[1:]])
            else:
                threshold_value = feature_name[node_id].split('_')
                threshold_decision = "!="
                threshold_name = threshold_value[0]
                threshold_value = ' '.join([str(elem) for elem in threshold_value[1:]])

            if node_id == node_index[-2]:
                print(
                        color +
                        "({feature_name} "
                        "{decision} {threshold_value}) then {values})".format(
                            feature_name=threshold_name,
                            decision = threshold_decision,
                            threshold_value=threshold_value,
                            values = dir
                        ), end="\n"
                    )
            else:

                print(
                    color +
                    "({feature_name} "
                    "{decision} {threshold_value}) AND ".format(
                        feature_name=threshold_name,
                        decision = threshold_decision,
                        threshold_value=threshold_value,
                    ), end=""
                )

        print(Style.RESET_ALL)


def calculate_mode(column):
    return column.mode()[0]

def get_categorical_columns(features):
    categorical = []
    for feature in features:
        if feature.type == 'Nominal':
            categorical.append(feature.name)
    return categorical

def calculate_imputation(features, data):
    categorical = get_categorical_columns(features)
    encoders = []

    for col in categorical:
        series = data[col]

        encoder = LabelEncoder().fit(series[series.notnull()])

        data[col] = pd.Series(
            encoder.transform(series[series.notnull()]),
            index=series[series.notnull()].index)

        encoders.append(encoder)

    imp_cat = IterativeImputer(estimator=RandomForestClassifier(),
                               initial_strategy='most_frequent',
                               max_iter=10, random_state=0, skip_complete=True)

    data_transformed = imp_cat.fit_transform(data[categorical])
    all_missing = []
    if data_transformed.shape[1] != len(categorical):
        for feature in features:
            if len(feature.missing_value_objects) == data_transformed.shape[0]:
                all_missing.append(feature.name)
                data_transformed = np.insert(data_transformed, feature.index, data[feature.name], axis=1)
    data[categorical] = data_transformed

    for encoder, col in zip(encoders, categorical):
        if col not in all_missing:
            data[col] = data[col].astype(int)
            series = data[col]
            decoded_series = encoder.inverse_transform(series)
            data[col] = decoded_series

    return data

def obtainAUCBinary(tp, tn, fp, fn):
    nPos = tp +fn
    nNeg = tn + fp

    recall = tp/nPos if (nPos > 0) else 1.0
    sensibility = tn/nNeg if (nNeg > 0) else 1.0

    return (recall + sensibility) / 2


def obtainAUCMulticlass(confusion, num_classes):
    sumVal = 0
    for i in range(num_classes):
        tp = confusion[i][i]

        for j in range(i+1, num_classes):
            fp = confusion[j][i]
            fn = confusion[i][j]
            tn = confusion[j][j]
            sumVal = sumVal + obtainAUCBinary(tp, tn, fp, fn)

    avg = (sumVal * 2) / (num_classes * (num_classes-1))
    return avg


def calculate_auc(current_confusion_matrix, values_count):
    auc = obtainAUCMulticlass(current_confusion_matrix, values_count)
    if auc < 0.5:
        return 1 - auc
    else:
        return auc


class ModelFeature:
    def __init__(self, name, index, values):
        self.name = name
        self.index = index
        self.values = values
        self.missing_value_objects = []
        self.type = 'Numerical' if np.issubdtype(np.array(values).dtype, np.number) else 'Nominal'
        self.encoded_value_names = []
        self.imputation_value = None

    def get_name(self):
        return self.name

    def get_values(self):
        return self.values

    def get_missing_value_indexes(self):
        return self.missing_value_objects

    def set_missing_value_objects(self, indexes):
        self.missing_value_objects = indexes

    def set_encoded_value_names(self, names):
        self.encoded_value_names = names

    def set_imputation_value(self, value):
        self.imputation_value =  value

    def get_imputation_value(self):
        return self.imputation_value









