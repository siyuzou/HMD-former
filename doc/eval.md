# Evaluation

## Preparation

Download [dataset, smpl parameters and our network checkpoint files](https://drive.google.com/drive/folders/1r1kxonqDhkl0SBZl1lVfrRuUKP6vLHlQ?usp=sharing), 
unzip them, and put them under `${ROOT}/data`.

Make sure the directory structure of `${ROOT}/data` is as follow:

``` 
${ROOT}
|-- data
    |-- dataset
        |-- Human3.6M_test
            |-- S9
            ...
    |-- model
        |-- smpl
            |-- SMPL_NEUTRAL.pkl
            ...
    |-- snap
        |-- h36m.pth
```

## Evaluation on Human3.6M

Run:
```bash
cd ${ROOT}
python3 test.py cfg="core/cfg/test/h36m.yaml" net.model_paths.hmd_former="data/snap/h36m.pth" exe.output_dir=${OUTPUT_DIR}
```