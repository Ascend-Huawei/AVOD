Delivery Result for GPU reproduction: [README_gpu](1_gpu_training/README_gpu.md) and support documents in folder [1_gpu_training](1_gpu_training)

Delivery Result for NPU training (currently, conversion, loss convergence, accuracy): as below, support documents in folder [2_npu_training](2_npu_training)


## Evaluation AP
### KITTI Object Detection Results (3D and BEV) Car
|              |             |   |           |        AP-3D |           |   |           |       AP-BEV |           |
|:------------:|:-----------:|---|:---------:|:------------:|:---------:|---|:---------:|:------------:|:---------:|
|            | **Runtime** |   |  **Easy** | **Moderate** |  **Hard** |   |  **Easy** | **Moderate** |  **Hard** |
|     Paper |      0.10   |   | **81.94** |    **71.88** | **66.38** |   |   88.53   |      83.79   | 77.90 |
|      GPU |1.83||80.97|67.27|65.61|   |**89.56**|**86.33**|**79.60**|
|      NPU |3.50||77.61|67.40|65.74|   |89.41|80.06|79.30|

### Code changes after using Conversion Tool:  
| Issue | Code change|
|-------|------------|
| path_drop_probabilities | initially set to 1.0;  |
|tf.contrib.memory_stats.MaxBytesInUse() not supported | remove |
|missing npu config|custom_op.name = "NpuOptimizer";rewrite_options.remapping; rewrite_options.memory_optimization; |
|Error Caused by: Pad BEV input from 700 to 704 to allow even divisions for max pooling; Pad + conv2d -> somehow pad operation seems to be fused into conv2d, causing shape issue when backpropgation| put padding operation outside of model |
|<s>Error Caused by: resize input image in model<s> | <s>move out to pre-processing & set input to static<s> |
| Dynamic shape caused by `bool_mask` - `mb_mask` | regularize the mask to static shape `[1024]`   |
| Dynamic shape in (`anchors_info`) and (`label_anchors`, `label_boxes_3d`, `label_classes`)| Padding anchor to a max static shape `30000`, `20`|
|Tf.case tf.cond seems also not working well in backprob|move the condition outside of the model|

Analysis: all the issues are somehow related to dynamic shape or if-condition, not likely to be resolved by the code conversion tool
