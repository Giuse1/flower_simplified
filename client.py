import ast
import copy
import optparse
import time
import torch
from utils import *
from model import *
import flwr as fl
import logging
import os
import re
import torch.nn.utils.prune as prune
from  simplify import simplify

list_dir = [x for x in os.listdir() if "reports" in x]
list_numbers_dirs = [int(re.findall(r'[0-9]+', x)[0]) for x in list_dir]
max_dirs = max(list_numbers_dirs)
folder = f"reports{int(max_dirs)}"

with open(f'settings.txt', 'r') as file_dict:
    settings = file_dict.read().replace('\n', '')
    settings = ast.literal_eval(settings)

batch_size = settings["batch_size"]
total_num_clients = settings["total_num_clients"]
client_per_round = settings["client_per_round"]
ADDRESS = settings["address"]

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False


class CifarClient(fl.client.NumPyClient):

    def __init__(self, id, trainloader, testloader, batch_size) -> None:
        super(CifarClient, self).__init__()

        self.id = id
        self.batch_size = batch_size
        if self.id < 2:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        print(self.device)
        self.net = cifarNet().to(self.device)
        self.starting_dict = copy.deepcopy(self.net.state_dict())
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = {"trainset": len(trainloader.dataset), "testset": len(testloader.dataset)}
        self.current_round = -1
        self.ordered_keys = (self.net.state_dict().keys())
        self.round_pruning = int(settings["round_pruning"])+1

        self.criterion_train = torch.nn.CrossEntropyLoss()
        self.criterion_test = torch.nn.CrossEntropyLoss(reduction="sum")

        self.logger = self.setup_logger('client_logger', f'{folder}/client_{self.id}.csv')

    def get_parameters(self):

        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):

        print("BEFORE SET PARAMETERS")
        print(self.net.conv1.weight[0][0])


        if self.current_round == self.round_pruning:
            # print("RECEIVED PARAMETERS")
            # print(state_dict)
            #
            # l = [v.shape for v in state_dict.values()]
            # l = [tuple(el[:2]) for el in l if len(el) > 1]
            # self.net = simplifiedCifarNet(l)

            # reset weights
            params_dict = zip(["conv1.weight","conv2.weight","conv3.weight","fc1.weight"], parameters) # todo

            self.net.load_state_dict(self.starting_dict, strict=True)

            received_state_dict = OrderedDict({k: v for k, v in params_dict})
            for k in self.ordered_keys:
                if "weight" in k and "fc2" not in k:
                    mask = torch.tensor(
                        np.unpackbits(received_state_dict[k], count=torch.numel(self.starting_dict[k])).reshape(
                            self.starting_dict[k].shape))
                    name, att = k.split('.')
                    prune.custom_from_mask(getattr(self.net, name), name=att, mask=mask)
                    prune.remove(getattr(self.net, name), 'weight')


            dummy_input = torch.zeros(1, 3, 32, 32)  # Tensor shape is that of a standard input for the given model
            self.net = simplify(self.net, dummy_input)  # todo pinned model list
            self.ordered_keys = (self.net.state_dict().keys())

        else:
            params_dict = zip(self.ordered_keys, parameters)
            received_state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.net.load_state_dict(received_state_dict, strict=True)

    def fit(self, parameters, config):

        self.current_round = config["rnd"]
        print(f"CLIENT {self.id} TRAIN - ROUND {self.current_round}")

        self.set_parameters(parameters)

        self.logger.info(','.join(
            map(str, [self.current_round, "", "training", "start", time.time_ns(), time.process_time_ns(), "", ""])))
        self.train(lr=config["lr"], local_epochs=config["local_epochs"])
        self.logger.info(','.join(
            map(str, [self.current_round, "", "training", "end", time.time_ns(), time.process_time_ns(), "", ""])))

        self.logger.info(','.join(
            map(str, [self.current_round, "", "evaluation", "start", time.time_ns(), time.process_time_ns(), "", ""])))
        train_loss, train_acc = self.test()
        self.logger.info(','.join(map(str, [self.current_round, "", "evaluation", "end", time.time_ns(),
                                            time.process_time_ns(), train_loss, train_acc])))

        print("AFTER TRAINING")
        print(self.net.conv1.weight[0][0])

        return self.get_parameters(), self.num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        raise Exception("ENTERED EVALUATE")

        self.current_round = config["rnd"]
        self.set_parameters(parameters)
        self.logger.info(','.join(
            map(str, [self.current_round, "evaluate", "start", time.time_ns(), time.process_time_ns(), "", ""])))
        loss, accuracy = self.test()
        self.logger.info(
            ','.join(map(str, [self.current_round, "evaluate", "end", time.time_ns(), time.process_time_ns(), "", ""])))

        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}

    def train(self, lr, local_epochs):

        self.net.train()

        optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)

        for _ in range(local_epochs):
            for batch, data in enumerate(self.trainloader, 0):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                self.logger.info(','.join(map(str, [self.current_round, batch, "forward", "start", time.time_ns(),
                                                    time.process_time_ns(), "", ""])))
                output = self.net(images)
                t1 = time.time_ns()
                cpu_t1 = time.process_time_ns()

                self.logger.info(','.join(map(str, [self.current_round, batch, "forward", "end", t1, cpu_t1, "", ""])))
                self.logger.info(','.join(map(str, [self.current_round, batch, "backward", "start", t1, cpu_t1, "", ""])))

                loss = self.criterion_train(output, labels)
                loss.backward()
                optimizer.step()
                self.logger.info(','.join(map(str, [self.current_round, batch, "backward", "end", time.time_ns(),
                                                    time.process_time_ns(), "", ""])))

    def test(self):
        self.net.eval()

        correct, loss = 0, 0.0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.net(images)
                loss += self.criterion_test(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        accuracy = correct / len(self.testloader.dataset)
        loss = loss / len(self.testloader.dataset)
        return loss, accuracy

    def setup_logger(self, name, log_file, level=logging.INFO):

        handler = logging.FileHandler(log_file, mode='w')
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)
        logger.info("round,batch,operation,phase,t,p,test_loss,test_acc")

        return logger


parser = optparse.OptionParser()
parser.add_option('-i', dest='id', type='int')
(options, args) = parser.parse_args()

trainloader, testloader = get_cifar_iid(batch_size, total_num_clients, options.id)
fl.client.start_numpy_client(f"{ADDRESS}:8080", client=CifarClient(options.id, trainloader, testloader, batch_size))
