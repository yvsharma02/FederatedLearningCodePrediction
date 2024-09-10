# Code Completion Using Federated Learning LLMs and Transformers

This was my attempt at creating my own LLM (Small LM?). The basic idea was to develop a vscode extension, that would use Federated Learning to privately train models on proprietary code. <br/>
People would have an incentive to use this because of it's existing code completion feature capability. The model would then train on the users private code to improve itself. <br/>

This uses Flower Framework for Federated Learning, and uses FedAvg for aggrgation <br/>

The architecture is very simple, based on the paper Attention is All you Need. Though I had to downscale the model by a lot to train it on my local machine. The resulsts were obviously not specticular, given the models size. It was ofc not going to autocomplete all the code at 22M parameters. But it was successful in predicting out chunks that vaguely resembled code. With my limited time and resources, I consider that as a win!<br/>
This is largely inspired by Andrej Karpthy's nanoGPT sreies. <br/>
My model has been trained and tested on a very small part of CoDesc (Java Code Dataset) dataset.


The standalone model is of 22M parameter. Here is its' training graph: <br/>


The federated model is of 2.3M parameter. Here is it's training graph: <br/>

To run the extension, just open the codepredictpoc folder, and press start debugging. Now whenever you press space in a javafile, wait for 15-20s (this is a delay to load env, the llm is almost instant), and a suggestion should popup. <br/>

To train the model in a standalone way, use main.ibpy <br/>

To start a server for federated learning, start src/server.py <br/>
To train the model in a federated way, start the code extension, press ctrl+shift=P then run Federated Learning. <br/>
Or you can use src/trainer.py as well <br/>