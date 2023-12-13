
# CCM: Context Chain Machine

CCM is inspired by GPT forward process. It is a experimental 
construct that seeks to understand the probabilistic model origin of transformer architecture


### Fit

```python
### the model is default to 2-layer
python3 train.py config/train_shakespeare.py --init_from=scratch --compile=True --n_layer=2 --model=CCM06
```

### Sample

```python
python3 sample.py --out_dir=out-shakespeare-word --model=CCM06
```

## acknowledgements

Training and sampling scripts are taken from the [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Kaparthy


## Changelog:
2023-12-14,Feng Geng: implemented CCM06 

