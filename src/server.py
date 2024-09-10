import flwr as fl
from custom_transformer import CustomTransformer
import numpy as np
from collections import OrderedDict
import torch
import glob
import os 

with torch.no_grad():

    # Should match the client
    CONTEXT_LEN = 128
    BLOCK_COUNT = 2
    EMBED_DIM = 256
    NUM_HEADS = 16
    LEARNING_RATE = 1e-2
    BATCH_COUNT = 64
    ITERATIONS = 1
    VOCAB_SIZE = 2500

    def weighted_average(metrics):
        accuracies = [num_examples * m["loss"] for num_examples, m in metrics]
        losses = [num_examples * m["loss"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        return {"accuracy": sum(accuracies) / sum(examples), "loss": sum(losses) / sum(examples)}

    net = CustomTransformer(VOCAB_SIZE, CONTEXT_LEN, EMBED_DIM, NUM_HEADS, BLOCK_COUNT)

    if (not os.path.exists("out")):
        os.makedirs("out")

    class SaveModelStrategy(fl.server.strategy.FedAvg):
        def aggregate_fit(
            self,
            server_round: int,
            results,#: #list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures,#: list[Union[Tuple[ClientProxy, FitRes], BaseException]],
        ):# -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

            if aggregated_parameters is not None:
                print(f"Saving round {server_round} aggregated_parameters...")

                # Convert `Parameters` to `List[np.ndarray]`
                aggregated_ndarrays: list[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

                # Convert `List[np.ndarray]` to PyTorch`state_dict`
                params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                net.load_state_dict(state_dict, strict=True)

                # Save the model
                torch.save(net.state_dict(), f"out/model_round_{server_round}.pth")

            return aggregated_parameters, aggregated_metrics

    list_of_files = [fname for fname in glob.glob("./out/model_round_*")]
    if (len(list_of_files) > 0):
        latest_round_file = max(list_of_files, key=os.path.getctime)
        print("Loading pre-trained model from: ", latest_round_file)
        state_dict = torch.load(latest_round_file)
        net.load_state_dict(state_dict)
        state_dict_ndarrays = [v.cpu().numpy() for v in net.state_dict().values()]
        parameters = fl.common.ndarrays_to_parameters(state_dict_ndarrays)
    else:
        parameters = None

    # Create strategy and run server

    strategy = SaveModelStrategy(fraction_fit=0.49,fraction_evaluate=0.49,evaluate_metrics_aggregation_fn=weighted_average, min_fit_clients=1,min_evaluate_clients=1,min_available_clients=1, initial_parameters=parameters)
    #fl.server.start_server(strategy=strategy)

    # Start server 
    fl.server.start_server(server_address="0.0.0.0:8085",config=fl.server.ServerConfig(num_rounds=ITERATIONS),strategy=strategy)