# RGBAvatar

TL;DR Our method reconstructs a high-fidelity head avatar from a monocular video stream on the fly.


## Installation

1. Clone this repository.

   ```
   git clone https://github.com/LinzhouLi/RGBAvatar.git
   cd RGBAvatar
   ```

2. Create conda environment.

   ```
   conda create -n rgbavatar python=3.10
   ```

3. Install [PyTorch](https://pytorch.org/get-started/locally/) and [nvdiffrast](https://nvlabs.github.io/nvdiffrast/).

5. Install other packages.

   ```
   pip install -r requirements.txt
   ```

6. Compile PyTorch CUDA extension.

   ```
   pip install submodules/diff-gaussian-rasterization
   ```

## Data preprocessing

For offline reconstruction, we use FLAME template model and follow INSTA to preprocess the video sequence.

1. You need to create an account on the [FLAME website](https://flame.is.tue.mpg.de/download.php) and download FLAME 2020 model. Please unzip FLAME2020.zip and put `generic_model.pkl` under `./data/FLAME2020`.

2. Please follow the instructions in [INSTA](https://github.com/Zielon/INSTA). You may first use [Metrical Photometric Tracker](https://github.com/Zielon/metrical-tracker) to track and run `generate.sh` provided by INSTA to mask the head.

3. Organize the INSTA's output in the following form, and modify the `data_dir` in config file to refer to the dataset path.

   ```
   <DATA_DIR>
       ├──<SUBJECT_NAME>
               ├── checkpoint # FLAME parameter for each frame, generated by the tracker 
               ├── images # generated by the script of INSTA
   ```

For online reconstruction, we use FaceWareHouse template model and DDE tracker to compute the expression coeafficients in real-time. Due to copy right, we can not release the code of this version.

## Running

### Offline Training

```
python train_offline.py --subject SUBJECT_NAME --work_name WORK_NAME --config CONFIG_FILE_PATH --preload
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for train_offline.py</span></summary>

  #### --subject
  Subject name for training (`bala` by default).
  #### --work_name
  A nick name for the experiment, training results will be saved under `output/WORK_NAME`.
  #### --config
  Config file path (`config/offline.yaml` by default).
  #### --split
  Use `train`/`test`/`all` split of the image sequence (`train` by default).
  #### --preload
  Whether to preload image data to CPU memory, which accelerate the training speed.
  #### --log
  Whether to output log information during training.

</details>

### Online Training

```
python train_online.py --subject SUBJECT_NAME --work_name WORK_NAME --config CONFIG_FILE_PATH --video_fps 25
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for train_online.py</span></summary>

  #### --subject
  Subject name for training (`bala` by default).
  #### --work_name
  A nick name for the experiment, training results will be saved under `output/WORK_NAME`.
  #### --config
  Config file path (`config/online.yaml` by default).
  #### --video_fps
  FPS of the input video stream (`25` by default ).
  #### --log
  Whether to output log information during training.

</details>

### Evaluation

```
python calculate_metrics.py --subject SUBJECT_NAME --work_name WORK_NAME --config CONFIG_FILE_PATH
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for calculate_metrics.py</span></summary>

  #### --subject
  Subject name for training (`bala`  by default).
  #### --output_dir
  Path of the expeirment output folder (`output` by default).
  #### --work_name
  Name of the experiment to be evaluated.
  #### --split
  Frame number where split the training and test set. (`-350` by default ).

</details>

### Rendering

```
python render.py --subject SUBJECT_NAME --work_name WORK_NAME
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for render.py</span></summary>

  #### --subject
  Subject name for training (`bala`  by default).
  #### --output_dir
  Path of the expeirment output folder (`output` by default).
  #### --work_name
  Name of the experiment to be rendered.
  #### --white_bg
  Whether to use white background, back by default.
  #### --alpha
  Whether to render the alpha channel.

</details>

## Citation

```
TBD
```

