Delivery Result for GPU reproduction: [README_gpu](README_gpu.md)

checkpoint and pbtxt in onebox->HiSpark V2 -> /model_training_HQ/AVOD/
https://onebox.huawei.com/p/a5f11e8eac23b6a75f762ea6eb836e87

### Code changes after using conversion tool:  
| Issue | Code change|
|-------|------------|
|Pad + conv2d -> somehow pad operation seems to be fused into conv2d, causing shape issue when backpropgation  | put padding operation outside of model | 
|Boolean_mask causes issue in backprop calculation, seems related to dynamic shape | replace boolen_mask with fixed shape | 
|Tf.case tf.cond seems also not working well in backprob| move the condition outside of the model  |
|Dynamic shape input is not well supported| pad reasonable values to make static input|

Analysis: all the issues are somehow related to dynamic shape or if-condition, not likely to be resolved by the code conversion tool
