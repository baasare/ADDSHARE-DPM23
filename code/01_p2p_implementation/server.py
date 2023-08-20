import os
import json
import pandas as pd
from code.helpers.training import get_lenet5
from timeit import default_timer as timer
from code.helpers.libs.p2pnetwork.node import Node
from tensorflow.python.keras import optimizers
from cryptography.hazmat.primitives import serialization
from code.helpers.utils import get_dataset, encode_layer, decode_layer, get_public_key

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Server(Node):
    def __init__(self, host, port, rounds, id=None, max_connections=0, client_type=None, dataset=None, callback=None):
        super(Server, self).__init__(host, port, id, callback, max_connections)

        self.start_time = None
        self.end_time = None
        self.pending_nodes = set()
        self.max_connections = max_connections
        self.average_weights = dict()
        _, _, self.X_test, self.y_test = get_dataset('server', dataset, 'iid', 'balanced')
        self.global_model = get_lenet5(dataset)
        self.max_rounds = rounds
        self.round = 0
        self.assembly_count = 0
        self.client_type = client_type
        self.dataset = dataset

        self.record = list()
        self.current_accuracy = 0
        self.threshold = 0

    def inbound_node_connected(self, node):
        if len(self.nodes_inbound) == self.max_connections:
            print(f"Connected to {self.max_connections} nodes starting training")
            self.start_round()
        else:
            print(
                f"Waiting to connect to {self.max_connections - len(self.nodes_inbound)} more nodes to start training"
            )

    def node_message(self, node, data):
        if data["message"] == MESSAGE_FL_UPDATE:
            self.fl_update(node, data["model_weights"])
        elif data["message"] == MESSAGE_GET_ALL_NODES:
            self.send_connected_nodes(node)
        elif data["message"] == MESSAGE_SHARING_COMPLETE:
            self.start_assembly()

    def start_round(self):
        print(f'Starting round ({self.round + 1})')

        self.start_time = timer()
        self.pending_nodes = self.nodes_inbound.copy()
        for layer in self.global_model.layers:
            if layer.trainable_weights:
                self.average_weights[layer.name] = [[], []]

        data = {
            "message": MESSAGE_START_TRAINING,
            "model_architecture": self.global_model.to_json(),
            "model_weights": encode_layer(self.global_model.get_weights()),
        }

        self.send_to_nodes(json.dumps(data))

    def fl_update(self, node, data):

        for layer in data.keys():
            temp_weight = decode_layer(data[layer])

            if len(self.average_weights[layer][0]) == 0 and len(self.average_weights[layer][1]) == 0:
                self.average_weights[layer][0] = temp_weight[0] / len(self.nodes_inbound)
                self.average_weights[layer][1] = temp_weight[1] / len(self.nodes_inbound)
            else:
                self.average_weights[layer][0] += temp_weight[0] / len(self.nodes_inbound)
                self.average_weights[layer][1] += temp_weight[1] / len(self.nodes_inbound)

        self.pending_nodes.remove(node)
        if not self.pending_nodes:
            self.apply_updates()

    def apply_updates(self):
        for layer in self.global_model.layers:
            if layer.trainable_weights:
                layer.set_weights(self.average_weights[layer.name])
        self.evaluate()

    def evaluate(self):

        self.global_model.compile(optimizer=optimizers.adam_v2.Adam(learning_rate=0.001),
                                  loss='categorical_crossentropy', metrics=['accuracy'])
        _, self.current_accuracy = self.global_model.evaluate(self.X_test, self.y_test, verbose=0)
        self.end_time = timer() - self.start_time
        print('Accuracy: ', self.current_accuracy)
        print(f'Round ({self.round + 1}) Time: {self.end_time}')

        self.record.append({
            'round': self.round + 1,
            'accuracy': self.current_accuracy,
            'fl': self.end_time,
        })
        self.end_round()

    def end_round(self):
        print("ROUND ENDED")
        self.round += 1
        if self.round < self.max_rounds:
            self.start_round()
        else:
            self.end_session()

    def end_session(self):
        print("SESSION ENDED")
        data = {
            "message": MESSAGE_END_SESSION,
            "model_weights": encode_layer(self.global_model.get_weights()),
        }
        pd.DataFrame.from_dict(self.record).to_csv(f"resources/results/{self.client_type}/{self.dataset}/server.csv",
                                                   index=False,
                                                   header=True)
        self.send_to_nodes(json.dumps(data))
        self.stop()

    def send_connected_nodes(self, node):
        connected_nodes = []
        for requested_node in self.nodes_inbound:

            if int(node.id) != int(requested_node.id):
                public_key = get_public_key(int(requested_node.id) + 1)

                public_key_bytes = public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                )

                connected_nodes.append(
                    {
                        "public_key": encode_layer(public_key_bytes),
                        "port": requested_node.port,
                        "address": requested_node.host,
                    }
                )

        data = {
            "message": MESSAGE_ALL_NODES,
            "nodes": connected_nodes,
        }

        self.send_to_node(node, json.dumps(data))

    def start_assembly(self):

        self.assembly_count += 1

        if self.assembly_count == len(self.nodes_inbound):
            self.assembly_count = 0
            data = {
                "message": MESSAGE_START_ASSEMBLY,
            }
            self.send_to_nodes(json.dumps(data))


if __name__ == "__main__":
    DATASET = ['cifar-10', 'mnist', 'f-mnist', 'svhn']
    fl_server = Server(ADDRESS, SERVER_PORT, ROUNDS, SERVER_ID, NODES, 'additive_rsa', DATASET[1])
    fl_server.start()
