from DataCenter import DataCenter
from AppUtils import logger
from NeuralNetworks import NeuralNetworks
import argparse
import os
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='CapitalOneAIEngine')
    parser.add_argument('--task', dest='task', type=str,
                        help='Predict or Train from the model')

    args = parser.parse_args()
    logger.info("Task is :{}".format(args.task))
    logger.info("Job started!")

    save_model_path = './Model/AlertTransactionModel'

    neural_network = NeuralNetworks(save_model_path)
    task = args.task
    if task == "predict":
        new_transaction, _ = DataCenter().run(task)
        predict = neural_network.sample(new_transaction)
        print("*Predict result:{}".format(predict))
        with open('prediction.txt', 'w') as outfile:
            json.dump({'predict':predict}, outfile)
    elif task == "train":
        inputs, targets = DataCenter().run(task)
        neural_network.train(inputs, targets)
    else:
        logger.fatal("!No task assigned, check the input arg '--task'")
    logger.info("Job finished!")

