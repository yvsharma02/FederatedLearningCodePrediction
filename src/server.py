import flwr as fl

def weighted_average(metrics):
    accuracies = [num_examples * m["loss"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples), "loss": sum(losses) / sum(examples)}

# Define strategy
# strategy = fl.server.strategy.FedAvg(
#     fraction_fit=1.0,
#     fraction_evaluate=1.0,
#     evaluate_metrics_aggregation_fn=weighted_average,
# )

# Start server
fl.server.start_server(
    server_address="0.0.0.0:8081",
    config=fl.server.ServerConfig(num_rounds=3, round_timeout=10),
)