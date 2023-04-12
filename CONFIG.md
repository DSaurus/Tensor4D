# Config Documentation

## Multi-view cases
```
general {
    # The base experiment directory where saves all results
    base_exp_dir = ./exp/CASE_NAME/t4d_thumbsup_img

    # Record the current models and settings for debug
    recording = [
        ./,
        ./models
    ]
}

dataset {
    # The directory of dataset
    data_dir = ../data/CASE_NAME/

    # The calibration file that contains camera parameters of each image
    render_cameras_name = cameras_sphere.npz 
    object_cameras_name = cameras_sphere.npz

    # The number of cameras in multi-view cases
    g_nums = 4
}

train {
    learning_rate = 5e-4
    
    # Decay parameter of learning rate
    learning_rate_alpha = 0.1

    # Total iterations
    end_iter = 100000

    # After 5000 iterations, Tensor4D start using fine level features.
    fine_level_iter = 5000

    # After 7500 iterations, Tensor4D start using features with the original resolution. Before that, it will downsample features (64x64) to accelerate convergence.
    downsample_iter = 7500

    # The batch_size of rays
    batch_size = 1024

    # Downsample 4x for fast validation 
    validate_resolution_level = 4

    # Use the first 2000 iterations and the first 50 images to warm up the training proces since we use gradient which is not stable at the initial state.
    warm_up_end = 2000
    warm_up_imgs = 50

    anneal_end = 0
    use_white_bkgd = False

    # Save checkpoints every 10000 iterations
    save_freq = 10000

    # Validate tensor4d every 500 iterations
    val_freq = 500

    # Print losses and learning rate every 100 iterations
    report_freq = 100

    # Surface constraint terms
    igr_weight = 0.2

    # Surface constraint terms in time, stablize the geometry at every frame
    tgr_weight = 0.2

    # Mask loss terms
    mask_weight = 0.3

    # Regularization terms of Tensor4D feature planes, add it will slightly reduce the training speed
    tv_weight = 0.00

    # Sampling more rays in areas with greater error
    weighted_sample = True

    # If true, compute color loss as Loss = (pred - gt) * mask
    mask_color_loss = False
}

model {
    flow = False

    # It is recommended to use "bounding" in training process which can mask samples out of bounding box. For rendering, you can use "Visualhull" to mask samples which can furhter accelerate rendering process.
    mask3d {
        mask_type = "bounding"
    }

    tensor4d {
        # Dimensions setting, the first parameter is the spatial dimension, the second parameter is the temporal dimension, the third parameter is the channel of features.
        lr_resolution = [128, 128, 32]
        hr_resolution = [512, 128, 16]
        feature_type = "4d"

        # Extract features every two images, the base channel of 2D CNN is 16
        image_guide = True
        image_guide_interval = 2
        image_guide_base = 16
    }

    sdf_network {
        d_out = 257
        d_in = 3
        d_hidden = 256
        n_layers = 3
        skip_in = [1]

        # Same with mip-nerf, we don't use mip-nerf by default
        min_emb = 0
        max_emb = 8      

        # The time embedding, it is recommended to use small number. Higher dimensions of time embedding will cause lattice noise.
        t_emb = 1 

        bias = 0.5

        # Initialize the geometry as a sphere
        geometric_init = True

        # Normalize layers of network
        weight_norm = True
    }

    variance_network {
        init_val = 0.3 # NeuS parameter
    }

    rendering_network {
        d_feature = 256
        mode = idr
        d_in = 9
        d_out = 3
        d_hidden = 256
        n_layers = 3
        weight_norm = True
        multires_view = 3
        squeeze_out = True
    }

    neus_renderer {
        n_samples = 64
        n_importance = 64
        n_outside = 0
        up_sample_steps = 1     # 1 for simple coarse-to-fine sampling
        perturb = 1.0
        mip_render = False
    }
}

```


## Monocular cases

We only pick the flow part of configs:

```

dataset {
    # monocular cases
    g_nums = 1 
}

train {
    # Since we use canonical space for geometry modeling which is independent of time, it is not necessary to constrain geometry in time.
    tgr_weight = 0.0
}

model {
    flow = True
    flow_tensor4d {
        # Flow tensor4d only have feature planes with low resolution. 
        lr_resolution = [128, 128, 32]
        hr_resolution = []
        feature_type = "4d"
        image_guide = False
    }
    flow_network {
        d_out = 3
        d_in = 3
        d_hidden = 256
        n_layers = 3
        skip_in = [1]
        min_emb = 0
        max_emb = 8
        t_emb = 6
        bias = 0.5
        geometric_init = False
        weight_norm = True
    }

    tensor4d {
        # Tensor4D in 3d formualtion, the first parameter is the spatial dimensions, the second parameter is the channels of features.
        lr_resolution = [128, 32]
        hr_resolution = [512, 16]
        feature_type = "3d"
        image_guide = False
    }
}
```