from math import log, e

#code before zero probability handling

# documentation

# Outlook    Temp    Humidity
#-------------------------------\
# Sunny      High      High     |    Yes
# Sunny      High      Normal   |    Yes
# Sunny      Low       Normal   |    No
# Sunny      Mild      High     |    Yes
# Rain       Mild      Normal   |    No
# Rain       High      High     |    No
# Rain       Low       Normal   |    No
# Cloudy     High      High     |    No
# Cloudy     High      Normal   |    Yes
# Cloudy     Mild      Normal   |    No

class DataSet:
    def __init__(self, features: list, data: list, values: list):
        self.features = features
        self.data = data
        self.values = values

        self.length = len(self.data)

    def where_query(self, feature: str, value: str):
        result_data = []
        result_values = []

        if feature == "val":
            for i in range(self.length):
                if self.values[i] == value:
                    result_data.append(self.data[i])
                    result_values.append(value)

        else:
            feature_index = self.features.index(feature)

            for i in range(self.length):
                if self.data[i][feature_index] == value:
                    result_data.append(self.data[i])
                    result_values.append(self.values[i])

        return DataSet(self.features, result_data, result_values)


def P(dataset: DataSet, feature: str, feature_value: str, value: str) -> float:
    value_query = dataset.where_query("val", value)

    return value_query.where_query(feature, feature_value).length / value_query.length


def P_by_value(dataset: DataSet, value: str) -> float:
    return dataset.where_query("val", value).length / dataset.length


def get_probability_for_value(dataset: DataSet, data_item: list, value: str) -> float:
    res = 1

    for i in range(len(dataset.features)):
        res += log(P(dataset, dataset.features[i], data_item[i], value), e) + log(P_by_value(dataset, value), e)

    return res


def possible_values(dataset: DataSet) -> list:
    return list(set(dataset.values))


def bayes_classifier(dataset: DataSet, data_item: list) -> str:
    res = ""
    highest_probability = 0

    for current_value in possible_values(dataset):
        current_probability = get_probability_for_value(dataset, data_item, current_value)

        if current_probability > highest_probability:
            highest_probability = current_probability
            res = current_value

    return res


data = DataSet(
    ["Outlook", "Temp", "Humidity"],
    [
        ["Sunny", "High", "High"],
        ["Sunny", "High", "Normal"],
        ["Sunny", "Low", "Normal"],
        ["Sunny", "Mild", "High"],
        ["Rain", "Mild", "Normal"],
        ["Rain", "High", "High"],
        ["Rain", "Low", "Normal"],
        ["Cloudy", "High", "High"],
        ["Cloudy", "High", "Normal"],
        ["Cloudy", "Mild", "Normal"]
    ],
    ["Yes", "Yes", "No", "Yes", "No", "No", "No", "No", "Yes", "No"]
)

print(bayes_classifier(data, ["Sunny", "Low", "Normal"]))
