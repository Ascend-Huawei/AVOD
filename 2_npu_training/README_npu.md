# GPU doc

## Training Dataset 训练数据集
Images, point cloud 图像，点云等: http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
```
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip
```
Extra txt: https://drive.google.com/open?id=1yjCwlSOfAZoPNNqoMtWfEjPCfhRfJB-Z

The folder should look something like the following:
```
Kitti
    object
        testing
        training
            calib
            image_2
            label_2
            planes
            velodyne
        train.txt
        val.txt
```

## Repo https://github.com/kujason/avod
Clone
```
git clone git@github.com:kujason/avod.git --recurse-submodules
```
Pre-process data by following the [README](README.md)  根据它的[README](README.md)做数据预处理

## GPU training command GPU训练命令

```
python avod/experiments/run_training.py --pipeline_config=avod/configs/pyramid_cars_with_aug_example.config  --device='0' --data_split='train'
```
OR
```
bash run_1p.sh
```
## Dependencies

Training: [requirement.txt](requirement.txt) <br>
Evaluation: [requirement_eval.txt](requirement_eval.txt)

## Frozen Graph

Get the frozen graph by

```
python pb_model/freeze_model.py --checkpoint_name='pyramid_cars_with_aug_example' --data_split='val' --ckpt_index=18 --device='1'
```
The `ckpt_index` here indicates the index of the checkpoint to freeze. If the `checkpoint_interval` inside your config is `1000`, to freeze checkpoints `116000`, the indices should be `--ckpt_index=116`. The default value is `-1` : freeze the last checkpoint. 

The frozen graph `.pb` file is saved in `pb_model/pb_model`. To further get the `.pbtxt` file, run following
```
python pb_model/tensorflow_util.py pb_model/pb_model/frozen_model_ckpt_-1.pb -t
```

## Evaluation AP
### KITTI Object Detection Results (3D and BEV) Car
|              |             |   |           |        AP-3D |           |   |           |       AP-BEV |           |
|:------------:|:-----------:|---|:---------:|:------------:|:---------:|---|:---------:|:------------:|:---------:|
|            | **Runtime** |   |  **Easy** | **Moderate** |  **Hard** |   |  **Easy** | **Moderate** |  **Hard** |
|     Paper |      0.10   |   | **81.94** |    **71.88** | **66.38** |   |   88.53   |      83.79   | 77.90 |
|      GPU |1.83||80.97|67.27|65.61|   |**89.56**|**86.33**|**79.60**|
|      NPU |3.50||77.61|67.40|65.74|   |89.41|80.06|79.30|

### Code changes after using conversion tool:  
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