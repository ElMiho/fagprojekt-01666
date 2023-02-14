# Project in 01666

Just a genereic README :))

## To do
### Report
-[ ] Introduction ---> Jonas og ElMiho
### Data processing
- [ ] Make flow diagram of the process
- [ ] Generate dataset files
- [ ] Dataset reader
- [ ] Decide on data representations
- [ ] Make rational to symbol converter
- [X] Convert infix <=> postfix ---> Hugo
- [ ] Convert infix <=> prefix ---> Jakob og Christian
- [ ] Convert prefix <=> postfix
- [ ] Convert from token list to string equation
- [ ] Make vocabulary class
- [ ] Make vectorizer class
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
