# Model training

- To train our CKRM  model for question answering (Q \rightarrow A) or answer justification (QA \rightarrow R), run:
```
sh run.sh
```
Each output folder contains the model checkpoint best.th and its predictions valpreds.npy on the validation set.

# Model evaluating

- To evaluate our CKRM models on validation set for question answering (Q \rightarrow A), answer justification (QA \rightarrow R) and combine their predictions ((Q \rightarrow AR), run:
```
sh evaluate.sh
```


