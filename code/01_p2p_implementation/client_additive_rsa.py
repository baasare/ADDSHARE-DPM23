import os
import sys
import json
import numpy as np
import pandas as pd
from timeit import default_timer as timer
from code.helpers.libs.p2pnetwork.node import Node
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import model_from_json
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization, hashes
from code.helpers.utils import get_dataset, encode_layer, decode_layer, generate_additive_shares, TimingCallback, \
    NumpyEncoder, NumpyDecoder, get_private_key

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class AdditiveClient(Node):
    def __init__(self, host, port, client_id=None, epochs=3, max_connections=0, dataset='cifar-10', callback=None):
        super(AdditiveClient, self).__init__(host, port, client_id, callback, max_connections)

        self.client = client_id
        self.dataset = dataset
        self.server = None
        self.port = port

        self.X_train, self.y_train, self.X_test, self.y_test = get_dataset(self.client, dataset, 'iid', 'balanced')
        self.model = None
        self.epochs = epochs

        # MPC
        self.private_key = get_private_key(self.client + 1)

        self.own_shares = dict()
        self.other_shares = dict()

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
            self.start_secret_sharing(data["nodes"])
        elif data["message"] == MESSAGE_MODEL_SHARE:
            self.accept_shares(data['model_share'])
        elif data["message"] == MESSAGE_START_ASSEMBLY:
            self.reassemble_shares()

    def start_training(self, global_model):
        self.round += 1
        self.model = model_from_json(global_model["model_architecture"])

        self.model.compile(optimizer=optimizers.adam_v2.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.set_weights(decode_layer(global_model["model_weights"]))

        cb = TimingCallback()

        self.model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=32, callbacks=[cb], verbose=False,
                       use_multiprocessing=True)
        _, self.current_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        self.current_training_time = sum(cb.logs)

        data = {
            "message": MESSAGE_GET_ALL_NODES,
        }
        self.send_to_node(self.server, json.dumps(data))

    def start_secret_sharing(self, data):
        self.fl_nodes = data
        self.start_time = timer()

        shares = int(len(self.fl_nodes) + 1)

        for layer in self.model.layers:
            if layer.trainable_weights:
                weight_shares = list(generate_additive_shares(layer.weights[0], shares))
                bias_shares = list(generate_additive_shares(layer.weights[1], shares))

                self.own_shares[layer.name] = [[], []]
                self.own_shares[layer.name][0].append(weight_shares.pop())
                self.own_shares[layer.name][1].append(bias_shares.pop())

                self.other_shares[layer.name] = [None, None]
                self.other_shares[layer.name][0] = weight_shares
                self.other_shares[layer.name][1] = bias_shares

        self.share_count += 1

        self.start_exchanging_shares()

    def start_exchanging_shares(self):
        for node in self.fl_nodes:
            layer_weights = dict()

            for layer in self.other_shares.keys():
                weight_bias = [
                    self.other_shares[layer][0].pop(),
                    self.other_shares[layer][1].pop()
                ],

                layer_weights[layer] = weight_bias

            json_str = json.dumps(layer_weights, cls=NumpyEncoder)

            public_key = serialization.load_pem_public_key(
                decode_layer(node['public_key'])
            )

            value_bytes = json_str.encode('utf-8')
            num_chunks = (len(value_bytes) + CHUNK_SIZE - 1) // CHUNK_SIZE

            weight_chunks = []
            for i in range(num_chunks):
                start = i * CHUNK_SIZE
                end = start + CHUNK_SIZE
                chunk = value_bytes[start:end]
                weight_chunks.append(chunk)

            encrypted_messages = []
            for json_byte_chunk in weight_chunks:
                encrypted_messages.append(
                    public_key.encrypt(
                        json_byte_chunk,
                        padding.OAEP(
                            mgf=padding.MGF1(algorithm=hashes.SHA256()),
                            algorithm=hashes.SHA256(),
                            label=None
                        )
                    )
                )

            data = {
                "message": MESSAGE_MODEL_SHARE,
                "model_share": encode_layer(encrypted_messages),
            }

            client_node = self.connect_get_node(node['address'], int(node['port']))
            self.send_to_node(client_node, json.dumps(data))

    def accept_shares(self, encoded_data):
        decoded_messages = decode_layer(encoded_data)

        decrypted_messages = []
        for chunk in decoded_messages:
            decrypted_messages.append(
                self.private_key.decrypt(
                    chunk,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
            )

        separator = b''
        decoded_data = separator.join(decrypted_messages).decode('utf8')
        data = json.loads(decoded_data, cls=NumpyDecoder)

        for layer in data.keys():
            weight_bias = data[layer][0]
            self.own_shares[layer][0].append(weight_bias[0])
            self.own_shares[layer][1].append(weight_bias[1])

        self.share_count += 1

        if self.share_count == int(len(self.fl_nodes) + 1):
            self.share_count = 0
            payload = {
                "message": MESSAGE_SHARING_COMPLETE,
            }
            self.send_to_node(self.server, json.dumps(payload))

    def reassemble_shares(self):
        layer_weights = dict()

        for layer in self.own_shares.keys():
            temp_weight_bias = [None, None]
            temp_weight_bias[0] = np.sum((self.own_shares[layer][0]), axis=0)
            temp_weight_bias[1] = np.sum((self.own_shares[layer][1]), axis=0)
            layer_weights[layer] = encode_layer(temp_weight_bias)

        self.end_time = timer() - self.start_time

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
        pd.DataFrame.from_dict(self.record).to_csv(
            f"resources/results/additive_rsa/{self.dataset}/client_{self.client + 1}.csv",
            index=False,
            header=True)
        # self.model.save(f"resources/models/additive_rsa/{self.dataset}/client_{self.client}_lenet_5.h5")
        self.stop()

    def end_session(self, data):
        model_weights = decode_layer(data['model_weights'])
        self.model.set_weights(model_weights)
        self.disconnect()


if __name__ == "__main__":
    DATASET = ['cifar-10', 'mnist', 'f-mnist', 'svhn']
    NODE_PORT = SERVER_PORT + int(sys.argv[1])
    client = AdditiveClient(ADDRESS, NODE_PORT, client_id=int(sys.argv[1]) - 1, epochs=1, max_connections=6,
                            dataset=DATASET[1])
    client.start()
    client.connect_with_node(ADDRESS, SERVER_PORT)
