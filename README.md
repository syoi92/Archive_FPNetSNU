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
% |model_id|<p>scale<br>(pivot at 2)</p>|style_loss|<p>epoch<br>epoch_step</p>|
|model_id|description|
|:----:|---|
|**01**||
|..||
|**1 **|**scale=1/epoch=10/epoch_step=2**|
|10|default|
|11|style_loss=0|
|12|style_ratio=0.03|
|..||
|**2 **|**scale=0.5/epoch=40/epoch_step=10**|
|20|default|
|21|style_loss=0|
|22|style_ratio=0.03|
|..||
|**3 **|**scale=0.25/epoch=160/epoch_step=40**|
|30|default|
|31|style_loss=0|
|32|style_ratio=0.03|


