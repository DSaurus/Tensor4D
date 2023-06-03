# Preprocessing Tensor4D Dataset

## Example: _yxd-6views_

Once you have extracted the data, the data directory structure should be as follows:

```
yxd-6views
├── 20114279.hevc
├── 20114281.hevc
├── 22053924.hevc
├── 22053927.hevc
├── 22138903.hevc
├── 22139915.hevc
└── calibration_full.json
```

There are three steps to process the data:
1. Transform the `hevc` format videos to `jpg` images and save extrinsics and intrinsics.
    
    `python preprocess.py --mode 'process' --data_dir /path/to/data/directory`
    
[//]: # (    After using the command line above, the directory structure should be as follows:)

[//]: # (   ```)

[//]: # (   .)

[//]: # (   ├── 20114279)

[//]: # (   ├── 20114279_extrinsic.npy)

[//]: # (   ├── 20114279_intrinsic.npy)

[//]: # (   ├── 20114281)

[//]: # (   ├── 20114281_extrinsic.npy)

[//]: # (   ├── 20114281_intrinsic.npy)

[//]: # (   ├── 22053924)

[//]: # (   ├── 22053924_extrinsic.npy)

[//]: # (   ├── 22053924_intrinsic.npy)

[//]: # (   ├── 22053927)

[//]: # (   ├── 22053927_extrinsic.npy)

[//]: # (   ├── 22053927_intrinsic.npy)

[//]: # (   ├── 22138903)

[//]: # (   ├── 22138903_extrinsic.npy)

[//]: # (   ├── 22138903_intrinsic.npy)

[//]: # (   ├── 22139915)

[//]: # (   ├── 22139915_extrinsic.npy)

[//]: # (   ├── 22139915_intrinsic.npy)

[//]: # (   └── calibration_full.json)

[//]: # ()
[//]: # (   ```)

2. Create the masks with any segmentation or matting methods. The required directory structure of one camera view should be as follows:
   ```
    20114279
    ├── color00000.jpg
    ├── color00001.jpg
    ├── ...
    ├── mask00001.jpg
    ├── mask00001.jpg
    ├── ...
    ``` 
3. Generate data with Tensor4D format.

   `python preprocess.py --mode 'generate' --start_frame ${start_frame_index} --n_frames ${number_frames} --data_dir /path/to/data/directory/of/step1 --output_dir /path/to/output/directory`

   `--start_frame`: Specifies the starting frames of the data used for converting to the Tensor4D format.
   
   `n_frames`: Specifies the number of frames required in each view. 
   
   The final output directory structure:
   ```
   output_dir
   ├── cameras_sphere.npz
   ├── image
   └── mask
   ```