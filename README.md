# Code Completion Using Federated Learning LLMs and Transformers

This was my attempt at creating my own LLM (Small LM?). The basic idea was to develop a vscode extension, that would use Federated Learning to privately train models on proprietary code. <br/>
People would have an incentive to use this because of it's existing code completion feature capability. The model would then train on the users private code to improve itself. <br/>

This uses Flower Framework for Federated Learning, and uses FedAvg for aggrgation <br/>

The architecture is very simple, based on the paper Attention is All you Need. Though I had to downscale the model by a lot to train it on my local machine. The resulsts were obviously not specticular, given the models size. It was ofc not going to autocomplete all the code at 22M parameters. But it was successful in predicting out chunks that vaguely resembled code. With my limited time and resources, I consider that as a win!<br/>
This is largely inspired by Andrej Karpthy's nanoGPT sreies. <br/>
My model has been trained and tested on a very small part of CoDesc (Java Code Dataset) dataset.

The standalone model is of 22M parameter. Here is its' training graph (Trained for over a day at 50K samples): <br/>

![output](https://github.com/user-attachments/assets/15b7623f-1d4b-44b0-9d5b-1e1b560f0e61)

<br/> The federated model is of 2.3M parameter. Here is it's training graph (Trained at 2K samples): <br/>

![federated_loss](https://github.com/user-attachments/assets/df2c3a1b-3c32-4ae8-acd5-ec245b04c719)


<br/> To run the extension, just open the codepredictpoc folder, and press start debugging. Now whenever you press space in a javafile, wait for 15-20s (this is a delay to load env, the llm is almost instant), and a suggestion should popup. <br/>

Example of a predicition by 22M model: <br/>

_________________________SAMPLED_________________________:
PluggableType pluggableType=BackendDescriptor.PluggableType.UNKNOWN;
      if (cn != null && cn.endsWith(DATABASE_JE_MONITORING_ENTRY_SUFFIX)) {
        pluggableType=BackendDescriptor.PluggableType.JE;
        monitorBackendID=cn.substring(0,cn.length() - DATABASE_JE_MONITORING_ENTRY_SUFFIX.length());
      }
      if (cn != null && cn.endsWith(DATABASE_PDB_MONITORING_ENTRY_SUFFIX)) {
        pluggableType=BackendDescriptor.PluggableType.PDB;
        monitorBackendID=cn.substring(0,cn.length() - DATABASE_PDB_MONITORING_ENTRY_SUFFIX
_________________________PREDICTED_______________________:
,0,ASE_CONFRELTERERENCEMCAPPROVID,fs.variableCompospVED_RESS);
        getSrc.start();
      }
       MessageType.set(asser_TYPE,ge);
    }
    return level;
    AIREWORY;
    else {
    }
    if (level != null) {
      cn=(cmpS + 1) >>> 1) {
           for (Cubst.nf : zplu > Colorphice	float b.nDB.Counts) {
        return NEqualledlyphaNeeds;
  }
 catch (per) {
    show.finishertra(bit.WAMPLEM_PROLIENTEREQUEST_ON_SYNER,b) - 1);
    } }

  }
  }
  entermin",getLabelPfinal Client.getLabel() > 2) + 1 1);
  readPrevFocusableByte((byte)data.getC


To train the model in a standalone way, use main.ibpy <br/>

To start a server for federated learning, start src/server.py <br/>
To train the model in a federated way, start the vscode extension, press ctrl+shift+P then run Federated Learning. <br/>
Or you can use src/trainer.py as well <br/>
