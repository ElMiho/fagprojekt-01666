# Project in 01666

Just a genereic README :))

## To do
### Report

- [X] Introduction. Background ---> Jonas
- [ ] Introduction. Neural Network --> Mikael 
- [X] Introduction. Our idea ---> Jonas
- [ ] Document progress of data handling --> Jacob/Christian

### Data processing
- [ ] Make flow diagram of the process
- [ ] Generate dataset files
- [ ] Dataset reader output
- [X] Dataset reader input --> Jonas
- [X] Tokenize input data ---> Jonas
- [ ] From input token to rational numbers 
- [X] Decide on data representations
- [X] Make rational to symbol converter
- [X] Convert infix <=> postfix ---> Hugo
- [ ] Convert infix <=> prefix ---> Jacob og Christian
- [ ] Convert prefix <=> postfix
- [X] Infix to float value ---> Jacob
- [X] Convert from token list to string equation
- [ ] Evaluate token list i.e. method in the equation class for the conversion: `List[Token]` => `float`
- [X] Make vocabulary class - wrapper class for token2idx and idx2token mappings
- [X] Make vectorizer class - wrapper for `vectorize` function to turn equation into list of indices (is part of the vocabulary now)
- [ ] Make pytorch `dataset` and `dataloaders`
- [ ] Convert output to readable math ---> Jacob

### Model
- [X] Load or make model
- [X] Initialize loss function and optimizer
- [ ] Make model checkpoint saver
- [X] Train model

### Inference
- [ ] Create model inference function
- [ ] Create beam search inference function
- [ ] Make statistics of output results

### Extra Model validation tests
- [ ] different orders of the same roots in both denominator and numerator evaluates to the same sum. -> Jonas
- [ ] non reduced fractions must give the same sum as the reduced fraction. -> Jonas

### Extra
- [X] Write down all our statutes in readme --> Jonas
- [ ] Generate tester scripts (using `unittest` module)
- [ ] Map of the facebook github
- [ ] Generalize s.t. code can be used on any math problem
- [ ] Guide to use code for general math problem (mostly for own use)

## statutes
- all roots are from the closed bounded interval [-5, 5]
- we want about 40m sums that can be evaluated in mathematica 
- "#" repersent an poly of degree 0 in token space
- "/" represent an the devision between polynomials
- for the 1000-dataset, we have max_seq_length_target = 171
- We start our series from the index `n = 1`

## Generating sums locally vs. HPC
See `batch.sh` for running the script; on HPC we use `WolframKernel` instead of `wolframscript`

To run a file use

```
WolframKernel -script file.wls
```

Generating the polynomials requires a `data` folder locally. 


# Literature
1. Train a BERT model from hugging faces on custom dataset: https://medium.com/@utk.is.here/encoder-decoder-models-in-huggingface-from-almost-scratch-c318cce098ae




