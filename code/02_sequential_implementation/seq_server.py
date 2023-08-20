import os
import pandas as pd
from helpers.training import get_lenet5
from timeit import default_timer as timer
from tensorflow.python.keras import optimizers
from helpers.utils import get_dataset, encode_layer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Server:
    def __init__(self, clients=0, client_type=None, dataset=None):
        self.start_time = None
        self.end_time = None
        self.clients = clients
        self.average_weights = dict()
        _, _, self.X_test, self.y_test = get_dataset('server', dataset, 'iid', 'balanced')
        self.global_model = get_lenet5(dataset)
        self.round = 0
        self.assembly_count = 0
        self.client_type = client_type
        self.dataset = dataset

        self.record = list()
        self.current_accuracy = 0
        self.threshold = 0

    def start_round(self):
        print(f'Starting round ({self.round + 1})')

        self.start_time = timer()
        for layer in self.global_model.layers:
            if layer.trainable_weights:
                self.average_weights[layer.name] = [[], []]

        data = {
            "model_architecture": self.global_model.to_json(),
            "model_weights": encode_layer(self.global_model.get_weights()),
        }

        return data

    def fl_update(self, data):

        for layer in data.keys():
            temp_weight = data[layer]

            if len(self.average_weights[layer][0]) == 0 and len(self.average_weights[layer][1]) == 0:
                self.average_weights[layer][0] = temp_weight[0] / self.clients
                self.average_weights[layer][1] = temp_weight[1] / self.clients
            else:
                self.average_weights[layer][0] += temp_weight[0] / self.clients
                self.average_weights[layer][1] += temp_weight[1] / self.clients

        self.assembly_count = self.assembly_count + 1
        if self.clients == self.assembly_count:
            self.apply_updates()

    def apply_updates(self):
        for layer in self.global_model.layers:
            if layer.trainable_weights:
                layer.set_weights(self.average_weights[layer.name])

        self.global_model.compile(optimizer=optimizers.adam_v2.Adam(learning_rate=0.001),
                                  loss='categorical_crossentropy', metrics=['accuracy'])
        _, self.current_accuracy = self.global_model.evaluate(self.X_test, self.y_test, verbose=0)

        self.end_time = timer() - self.start_time
        self.round = self.round + 1

        print('Accuracy: ', self.current_accuracy)
        print(f'Round ({self.round}) Time: {self.end_time}')

        self.record.append({
            'round': self.round + 1,
            'accuracy': self.current_accuracy,
            'fl': self.end_time,
        })

        self.assembly_count = 0

        print("ROUND ENDED")

    def end_training(self):
        pd.DataFrame.from_dict(self.record).to_csv(f"resources/results/{self.client_type}/{self.dataset}/server.csv",index=False, header=True)
