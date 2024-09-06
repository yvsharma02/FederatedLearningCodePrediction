import flwr as fl
 
def weighted_average(metrics):
    accuracies = [num_examples * m["loss"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples), "loss": sum(losses) / sum(examples)}

strategy = fl.server.strategy.FedAvg(fraction_fit=0.49,fraction_evaluate=0.49,evaluate_metrics_aggregation_fn=weighted_average, min_fit_clients=1,min_evaluate_clients=1,min_available_clients=1)
# Start server 
fl.server.start_server(server_address="0.0.0.0:8081",config=fl.server.ServerConfig(num_rounds=3, round_timeout=1),strategy=strategy)