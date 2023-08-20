import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import json
import pandas as pd
from tensorflow.python.keras.models import model_from_json
from code.helpers.constants import MESSAGE_START_TRAINING, MESSAGE_FL_UPDATE, MESSAGE_END_SESSION, ADDRESS, SERVER_PORT
from code.helpers.libs.p2pnetwork.node import Node
from tensorflow.python.keras import optimizers
from code.helpers.utils import get_dataset, encode_layer, decode_layer, TimingCallback


class VanillaClient(Node):
    def __init__(self, host, port, client_id=None, epochs=3, max_connections=0, dataset=None, callback=None):
        super(VanillaClient, self).__init__(host, port, client_id, callback, max_connections)

        self.client = client_id
        self.dataset = dataset
        self.server = None

        self.X_train, self.y_train, self.X_test, self.y_test = get_dataset(self.client, dataset, 'iid', 'balanced')
        self.model = None
        self.epochs = epochs

        self.record = list()

        self.round = 0
        self.current_accuracy = 0
        self.current_training_time = 0

    def outbound_node_connected(self, node):
        if self.server is None:
            self.server = node

    def node_message(self, node, data):
        if data["message"] == MESSAGE_START_TRAINING:
            self.start_training(data)
        elif data["message"] == MESSAGE_END_SESSION:
            self.end_session(data)

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

        self.send_updates()

    def send_updates(self):
        model_weights = dict()
        counter = 1
        for layer in self.model.layers:
            if layer.trainable_weights:
                if self.client % 2 == 0:
                    if counter % 2 != 0:
                        model_weights[layer.name] = encode_layer(layer.get_weights())
                else:
                    if counter % 2 == 0:
                        model_weights[layer.name] = encode_layer(layer.get_weights())
                counter += 1

        self.record.append({
            'round': self.round,
            'accuracy': self.current_accuracy,
            'training': self.current_training_time,
        })

        data = {
            "message": MESSAGE_FL_UPDATE,
            "model_weights": model_weights,
        }

        self.send_to_node(self.server, json.dumps(data))

    def disconnect(self):
        pd.DataFrame.from_dict(self.record).to_csv(f"resources/results/partial_1/{self.dataset}/client_{self.client + 1}.csv",
                                                   index=False, header=True)
        # self.model.save(f"resources/models/partial_1/{self.dataset}/client_{self.client}_lenet_5.h5")
        self.stop()

    def end_session(self, data):
        model_weights = decode_layer(data['model_weights'])
        self.model.set_weights(model_weights)
        self.disconnect()


if __name__ == "__main__":
    DATASET = ['cifar-10','f-mnist', 'mnist', 'svhn']
    NODE_PORT = SERVER_PORT + int(sys.argv[1])
    client = VanillaClient(ADDRESS, NODE_PORT, client_id=int(sys.argv[1]) - 1, epochs=1, max_connections=6, dataset=DATASET[0])
    client.start()
    client.connect_with_node(ADDRESS, SERVER_PORT)
