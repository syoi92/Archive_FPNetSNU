## Usage sample

```console
~$CUDA_VISIBLE_DIVICES=0 python main.py --model_id=01 --use_styleloss=0 --scale=0.5
```

## Fixed parameters for now
img size = 512
network architecture = resnet51
number of first filters = 64
use_patch = True

## Models
| model_id | <p>scale<br>(pivot at 2)</p> | style_loss |<p>epoch<br>epoch_step</p> |
|:----:|:---:|:---:|:---:|
|**01**||||
|||||
|**10**|1|x|<p>10<br>2</p>|
|**11**|1|o|<p>10<br>2</p>|
|||||
|**20**|0.5|x|<p>40<br>10</p>|
|**21**|0.5|x|<p>40<br>10</p>|
|||||

