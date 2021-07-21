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
python pb_model/freeze_model.py --checkpoint_name='pyramid_cars_with_aug_example' --ckpt_index=18 --device='1'
```
The `ckpt_index` here indicates the index of the checkpoint to freeze. If the `checkpoint_interval` inside your config is `1000`, to freeze checkpoints `116000`, the indices should be `--ckpt_index=116`. The default value is `-1` : freeze the last checkpoint. 

The frozen graph `.pb` file is saved in `pb_model/pb_model`. To further get the `.pbtxt` file, run following
```
python pb_model/tensorflow_util.py pb_model/pb_model/frozen_model_ckpt_-1.pb -t
```

checkpoint and pbtxt in 
https://drive.google.com/drive/folders/16XYpDWfaou6KAEL-ZxZwU2Q-iy7zHNw9?usp=sharing
