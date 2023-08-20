import sys
import time
import json
import random
import numpy as np
import pandas as pd
from timeit import default_timer as timer
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import model_from_json

from code.helpers.libs.p2pnetwork.node import Node
from code.helpers.libs.tinysmpc.fixed_point import fixed_point, float_point
from code.helpers.utils import get_dataset, encode_layer, decode_layer, TimingCallback, generate_shamir_shares, \
    reconstruct_shamir_secret


class ShamirClient(Node):
    def __init__(self, host, port, client_id=None, epochs=3, max_connections=0, dataset='cifar-10', callback=None):
        super(ShamirClient, self).__init__(host, port, client_id, callback, max_connections)

        self.client = client_id
        self.dataset = dataset
        self.server = None
        self.port = port

        self.X_train, self.y_train, self.X_test, self.y_test = get_dataset(self.client, dataset, 'iid', 'balanced')
        self.model = None
        self.epochs = epochs

        # MPC
        self.fixedPoint = np.vectorize(fixed_point)
        self.floatPoint = np.vectorize(float_point)

        self.own_shares = dict()
        self.other_shares = dict()
        self.client_shares = list()

        self.fl_nodes = list()
        self.share_count = 0

        self.start_time = None
        self.end_time = None

        self.record = list()

        self.round = 0
        self.current_accuracy = 0
        self.current_training_time = 0

    def outbound_node_connected(self, node):
        if node.host == ADDRESS and node.port == SERVER_PORT:
            self.server = node

    def node_message(self, node, data):
        if data["message"] == MESSAGE_START_TRAINING:
            self.start_training(data)
        elif data["message"] == MESSAGE_END_SESSION:
            self.end_session(data)
        elif data["message"] == MESSAGE_ALL_NODES:
            self.start_secret_sharing(data)
        elif data["message"] == MESSAGE_MODEL_SHARE:
            self.accept_shares(data)
        elif data["message"] == MESSAGE_START_ASSEMBLY:
            self.reassemble_shares()

    def start_training(self, global_model):
        self.round += 1
        self.model = model_from_json(global_model["model_architecture"])

        self.model.compile(optimizer=optimizers.adam_v2.Adam(learning_rate=0.001), loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.set_weights(decode_layer(global_model["model_weights"]))

        cb = TimingCallback()

        self.model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=32, callbacks=[cb], verbose=False,
                       use_multiprocessing=True)
        _, self.current_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        self.current_training_time = sum(cb.logs)
        # print('Accuracy: ', self.current_accuracy)
        # print(f'Client ({self.client}) Training Time: {self.current_training_time}')

        data = {
            "message": MESSAGE_GET_ALL_NODES,
        }
        self.send_to_node(self.server, json.dumps(data))

    def start_secret_sharing(self, data):
        self.fl_nodes = data["nodes"]
        self.start_time = timer()

        clients = len(self.fl_nodes)
        threshold = int(THRESHOLD * clients)

        for layer in self.model.layers:
            if layer.trainable_weights:
                fixed_point_weight = self.fixedPoint(layer.weights[0])
                fixed_point_bias = self.fixedPoint(layer.weights[1])

                weight_shares = generate_shamir_shares(clients, threshold, fixed_point_weight)
                bias_shares = generate_shamir_shares(clients, threshold, fixed_point_bias)

                assert len(weight_shares) == clients
                assert len(bias_shares) == clients

                self.own_shares[layer.name] = [[], []]
                self.own_shares[layer.name][0].append(weight_shares.pop())
                self.own_shares[layer.name][1].append(bias_shares.pop())

                self.other_shares[layer.name] = [None, None]
                self.other_shares[layer.name][0] = weight_shares
                self.other_shares[layer.name][1] = bias_shares

        self.share_count += 1

        self.start_exchanging_shares()

    def start_exchanging_shares(self):
        ss_nodes = list(filter(lambda i: int(i['port']) != int(self.port), self.fl_nodes))

        for node in ss_nodes:
            layer_weights = dict()

            for layer in self.other_shares.keys():
                weight_bias = [None, None]
                weight_bias[0] = self.other_shares[layer][0].pop()
                weight_bias[1] = self.other_shares[layer][1].pop()

                layer_weights[layer] = encode_layer(weight_bias)

            data = {
                "message": MESSAGE_MODEL_SHARE,
                "model_share": layer_weights,
                "client": self.client
            }

            client_node = self.connect_get_node(node['address'], int(node['port']))
            self.send_to_node(client_node, json.dumps(data))
            time.sleep(5)

    def accept_shares(self, message):

        layer_weights = dict()
        layer_weights["client"] = message["client"]
        layer_weights["share"] = message["model_share"]
        self.client_shares.append(layer_weights)

        # data = message['model_share']
        # for layer in data.keys():
        # weight_bias = decode(data[layer])
        # self.own_shares[layer][0].append(weight_bias[0])
        # self.own_shares[layer][1].append(weight_bias[1])

        self.share_count += 1

        if self.share_count == len(self.fl_nodes):
            self.share_count = 0
            payload = {
                "message": MESSAGE_SHARING_COMPLETE,
            }
            self.send_to_node(self.server, json.dumps(payload))

    def reassemble_shares(self):
        # print("STARTING REASSEMBLY")
        layer_weights = dict()
        clients = len(self.fl_nodes)
        threshold = int(THRESHOLD * clients)

        for layer in self.own_shares.keys():
            weights_pool = random.sample(self.own_shares[layer][0], threshold)
            bias_pool = random.sample(self.own_shares[layer][1], threshold)

            weights_assembled = reconstruct_shamir_secret(weights_pool)
            bias_assembled = reconstruct_shamir_secret(bias_pool)

            temp_weight_bias = [None, None]
            temp_weight_bias[0] = self.floatPoint(weights_assembled)
            temp_weight_bias[1] = self.floatPoint(bias_assembled)

            layer_weights[layer] = encode_layer(temp_weight_bias)

        self.end_time = timer() - self.start_time
        # print(f'Client ({self.client}) Secret Sharing Time: {self.end_time}')

        self.record.append({
            'round': self.round,
            'accuracy': self.current_accuracy,
            'training': self.current_training_time,
            'secret_sharing': self.end_time
        })

        payload = {
            "message": MESSAGE_FL_UPDATE,
            "model_weights": layer_weights,
        }
        self.send_to_node(self.server, json.dumps(payload))

    def disconnect(self):
        pd.DataFrame.from_dict(self.record).to_csv(f"resources/results/shamir/{self.dataset}/client_{self.client}.csv",
                                                   index=False, header=True)
        # self.model.save(f"resources/models/shamir/{self.dataset}/client_{self.client}_lenet_5.h5")
        self.stop()

    def end_session(self, data):
        model_weights = decode_layer(data['model_weights'])
        self.model.set_weights(model_weights)
        self.disconnect()


if __name__ == "__main__":
    DATASET = ['cifar-10', 'mnist', 'f-mnist', 'svhn']
    NODE_PORT = SERVER_PORT + int(sys.argv[1])
    client = ShamirClient(ADDRESS, NODE_PORT, client_id=int(sys.argv[1]) - 1, epochs=3, max_connections=10,
                          dataset=DATASET[1])
    client.start()
    client.connect_with_node(ADDRESS, SERVER_PORT)
