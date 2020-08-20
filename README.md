## Usage sample

```console
~$CUDA_VISIBLE_DIVICES=0 python main.py --model_id=01 --use_styleloss=0 --scale=0.5
```

## fixed parameters for now
img size = 512
network architecture = resnet51
# of firs filters = 64
use_patch = True

## Model Comparison
|model_id|scale(pivot at 2)|style_loss|epoch/epoch_step|
|:---:|:---:|:---:|:-----:|
|01||||
|||||
|10|1|x|10/2|
|11|1|o|10/2|
|||||
|20|0.5|x|40/10|
|21|0.5|x|40/10|
|||||

