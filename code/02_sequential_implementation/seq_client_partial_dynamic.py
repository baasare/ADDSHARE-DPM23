import os
import pandas as pd
import concurrent.futures
from seq_server import Server
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import model_from_json
from helpers.constants import NODES, ROUNDS, EPOCHS, DATASET
from helpers.utils import decode_layer, TimingCallback, get_dataset_x, fetch_dataset, fetch_index

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class VanillaPartialClientTwo:
    def __init__(self, client_id, epochs, dataset, index, x_train, y_train, x_test, y_test):

        self.client_id = client_id
        self.dataset = dataset
        self.server = None

        self.X_train, self.y_train, self.X_test, self.y_test = \
            get_dataset_x(index, dataset, x_train, y_train, x_test, y_test)
        self.model = None
        self.epochs = epochs

        self.record = list()

        self.round = 0
        self.current_accuracy = 0
        self.current_training_time = 0

    def start_training(self, global_model):
        self.round += 1
        self.model = model_from_json(global_model["model_architecture"])
        self.model.compile(optimizer=optimizers.adam_v2.Adam(learning_rate=0.001), loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.set_weights(decode_layer(global_model["model_weights"]))

        cb = TimingCallback()

        self.model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=10, callbacks=[cb], verbose=False,
                       use_multiprocessing=True)
        _, self.current_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        self.current_training_time = sum(cb.logs)

        model_weights = dict()
        counter = 1
        for layer in self.model.layers:
            if layer.trainable_weights:
                if self.round % 2:
                    if self.client_id % 2 == 0:
                        if counter % 2 != 0:
                            model_weights[layer.name] = layer.get_weights()
                    else:
                        if counter % 2 == 0:
                            model_weights[layer.name] = layer.get_weights()
                else:
                    if self.client_id % 2 == 0:
                        if counter % 2 == 0:
                            model_weights[layer.name] = layer.get_weights()
                    else:
                        if counter % 2 != 0:
                            model_weights[layer.name] = layer.get_weights()

                counter += 1

        self.record.append({
            'round': self.round,
            'accuracy': self.current_accuracy,
            'training': self.current_training_time,
        })

        return model_weights

    def end_training(self):
        pd.DataFrame.from_dict(self.record).to_csv(
            f"resources/results/partial_dynamic/{self.dataset}/client_{self.client_id}.csv",
            index=False, header=True)


def process_end_training(client):
    client.end_training()


if __name__ == "__main__":
    print(f"EXPERIMENT: PARTIAL DYNAMIC")
    for dataset in DATASET:
        print(f"DATASET: {dataset.upper()}")
        indexes = fetch_index(dataset)
        (x_train, y_train), (x_test, y_test) = fetch_dataset(dataset)

        fl_server = Server(NODES, 'partial_dynamic', dataset)
        clients = [VanillaPartialClientTwo(i, EPOCHS, dataset, indexes[i], x_train, y_train, x_test, y_test) for i in
                   range(NODES)]

        for _ in range(ROUNDS):
            client_weights = []
            data = fl_server.start_round()
            for client in clients:
                client_weights.append(client.start_training(data))

            for weight in client_weights:
                fl_server.fl_update(weight)

        fl_server.end_training()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_end_training, client) for client in clients]
            concurrent.futures.wait(futures)
