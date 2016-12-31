Improved Stereo Matching with Constant Highway Networks and Reflective Loss
===================================================================================

This implements the full pipeline of our paper [Improved Stereo Matching with Constant Highway Networks and Reflective Loss]() by Amit Shaked and Lior Wolf

The repository contains

- Training of the Constant Highway Network to compute the matching cost
- A few post processing steps taken from [MC-CNN](https://github.com/jzbontar/mc-cnn)
- Training of the Global Disparity Network with the Reflective Loss
- A confidence based outlier detection and interpolation


## Requirements
- Install [Torch](http://torch.ch/docs/getting-started.html) on a machine with CUDA GPU
- Install [cuDNN v4 or v5](https://developer.nvidia.com/cudnn) and the Torch [cuDNN bindings](https://github.com/soumith/cudnn.torch/tree/R4)
If you already have Torch installed, update `nn`, `cunn`, and `cudnn`.
- Install [OpenCV 2.4](http://opencv.org/) and [png++](http://www.nongnu.org/pngpp/)
- A NVIDIA GPU with at least 6 GB of memory is required to run on the KITTI
data set and 12 GB to run on the Middlebury data set.

The code is released under the BSD 2-Clause license.
Please cite our [paper]()
if you use code from this repository in your work.

	@article{shaked2016stereo,
	  title={Improved Stereo Matching with Constant Highway Networks and Reflective Loss},
	  author={{Shaked, Amit and Wolf, Lior},
	  journal={arXiv preprint },
	  year={2016}
	}

Setup
------------------------
Create directory for the data to be stored and link it under the name "storage" where the README file is
```bash
ln -s [your_dir] storage
```
Or simply create a storage directory
```bash
mkdir storage
```

Run mkdirs script:
```bash
scripts/mkdirs.sh
```

Compile the shared libraries:
```bash
make
```

The command should produce the files: `libadcensus.so`, `libcv.so` and `libcuresmatch.so` in the lib dir.


### KITTI


- Download the [KITTI 2012](http://www.cvlibs.net/download.php?file=data_stereo_flow.zip) data set and unzip it
into `storage/data.kitti/unzip` (you should end up with a file `storage/data.kitti/unzip/training/image_0/000000_10.png`) and 
- Download the [KITTI 2015](http://www.cvlibs.net/download.php?file=data_scene_flow.zip) data set and unzip it
into `storage/data.kitti2015/unzip` (you should end up with a file `storage/data.kitti2015/unzip/training/image_2/000000_10.png`).


Run the preprocessing script:
```bash
scripts/preprocess_kitti.lua -color rgb -storage storage
```

It should output:
```bash
dataset 2012
1
...
389
dataset 2015
1
...
400
```

### Middlebury
Run `download_middlebury.sh` to download the training data
(this can take a long time, depending on your internet connection).
```bash
scripts/download_middlebury.sh
```

The data set is downloaded into the `data.mb/unzip` directory.

Compile the [MiddEval3-SDK](http://vision.middlebury.edu/stereo/submit3/). You
should end up with the `computemask` binary in one of the directories listed in
your `PATH` enviromential variable.  

Install [ImageMagick](http://www.imagemagick.org/script/index.php); the
preprocessing steps requires the `convert` binary to resize the images.

Run the preprocessing script:
```bash
mkdir storage/data.mb.imperfect_gray
scripts/preprocess_mb.py imperfect gray
```

It should output:
```bash
Adirondack
Backpack
...
testH/Staircase
```

The preprocessing is slow (it takes around 30 minutes) the first time it is
run, because the images have to be resized.


Usage
---------------------
Enter the src directory.
The `main.lua` file contains different training and testing options:

- 'a' is the action, it can be can be 'train\_mcn' to train the matching cost network, 'train\_gdn' to train the global disparity network, 'test' to check the pipeline on the validation set and 'submit' to create the submission file for the online evaluation servers
- 'ds' is the dataset (kitti, kitti2015 or mb)
- 'mc' is the matching cost architecture to use
- 'm' is the mode ('fast', 'acrt' or 'hybrid' for the hybrid loss)
- 'gdn' is the global disparity network architecture. Use 'ref' for reflective.
   Don't use this option when training the matching cost network
- 'all' is to train on both training and validation data.
   When choosing this option the gdn will be automatically trained and the submission file would be created.

See `opts.lua` for other options.

### Training

Try training the hybrid Resmatch matching cost network:
```bash
th main.lua -ds kitti -a train_mcn -mc resmatch -m hybrid
```

And then training the gdn with the reflective loss, using this matching cost network:
```bash
th main.lua -ds kitti -a train_gdn -mc resmetch -m hybrid -mcnet ../storage/net/mc/kitti_resmatch_hybrid_LL_rgb.t7 -gdn ref
```

You can also try training the fast resmatch architecture, on 0.2 of the data, and test it every 3 epochs:

```bash
th main.lua -ds kitti -a train_mcn -mc resmatch -m fast -debug -times 3 -subset 0.2
```

