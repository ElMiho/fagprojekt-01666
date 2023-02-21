# Project in 01666

Just a genereic README :))

## To do
### Report
- [ ] Introduction ---> Jonas og ElMiho
### Data processing
- [ ] Make flow diagram of the process
- [ ] Generate dataset files
- [ ] Dataset reader output
- [ ] Dataset reader input --> Jonas
- [ ] Tokenize input data ---> Jonas
- [ ] Decide on data representations
- [X] Make rational to symbol converter
- [X] Convert infix <=> postfix ---> Hugo
- [ ] Convert infix <=> prefix ---> Jacob og Christian
- [ ] Convert prefix <=> postfix
- [ ] Convert from token list to string equation
- [ ] Evaluate token list i.e. method in the equation class for the conversion: `List[Token]` => `float`
- [ ] Make vocabulary class - wrapper class for token2idx and idx2token mappings
- [ ] Make vectorizer class - wrapper for `vectorize` function to turn equation into list of indices
- [ ] Make pytorch `dataset` and `dataloaders`

### Model
- [ ] Load or make model
- [ ] Initialize loss function and optimizer
- [ ] Make model checkpoint saver
- [ ] Train model

### Inference
- [ ] Create model inference function
- [ ] Create beam search inference function
- [ ] Make statistics of output results

### Extra
- [ ] Generate tester scripts (using `unittest` module)
- [ ] Map of the facebook github


# Generating sums locally vs. HPC
See `batch.sh` for running the script; on HPC we use `WolframKernel` instead of `wolframscript`

To run a file use

```
WolframKernel -script file.wls
```

Generating the polynomials requires a `data` folder locally. 


# Literature
1. Train a BERT model from hugging faces on custom dataset: https://medium.com/@utk.is.here/encoder-decoder-models-in-huggingface-from-almost-scratch-c318cce098ae




