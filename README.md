__3D FMRI reconstructions with GAN-like deep neural networks__

![Gemini_Generated_Image_x0uupex0uupex0uu](https://github.com/user-attachments/assets/4ae9bcf1-e144-4b40-a2ca-134f8404a864)

## About the project
In this project i present networks for neuro data analization. The main perpose of this project was to get accurate EEg to 3d BOLD FMRI-like scence approch based on 3d CNN GAN-like models. Above you can see results of modeling (on the right side) and related data from true data distribution (on the left side).
![brain_samples](https://github.com/user-attachments/assets/3534d805-4533-48b8-9de7-7cb8387772db)
Also in this project you will find attempts to reconstruct images that was shown to the test subjects during the EEG-FMRI scanning. On image above you find images from cifar10 dataset that was recoverd from test subjects visual cortex recordings during the tests
![image](https://github.com/user-attachments/assets/67f0a919-81c3-41b9-9f46-bd6000ba5a26)



Data were taken from datasets towards the linkcs: 
BOLD-5000 dataset wich is consist from 3d BOILD FMRI data of visual cortex activities correlated with images wich was shown to the experimental test subjects
```
https://openneuro.org/datasets/ds001499/versions/1.3.0
```
EEG-FMRI dataset with the set of 3d FMRI data correlated with EEG samples for EEG2FMRI projections
```
https://openneuro.org/datasets/ds004196/versions/1.0.1/file-display/sub-01:ses-EEG:eeg:sub-01_ses-EEG_task-inner_eeg.bdf
```

## TODO
There is still some sutf that me done in this projection. So you can observ TODO list above
- [x] write EEg to 3d projection model.
- [x] images reconstractions from visual cortex of brain.
- [ ] automated model trainers.
- [ ] other utils.

## Project structure
```
.
├── README.md
├── Source.gv
├── ae_eeg.yaml
├── src
│   ├── models
│   │   ├── __pycache__
│   │   │   ├── ae_net.cpython-311.pyc
│   │   │   ├── eeg2volume.cpython-311.pyc
│   │   │   ├── img2img_net.cpython-311.pyc
│   │   │   ├── layers.cpython-311.pyc
│   │   │   ├── mels2img_net.cpython-311.pyc
│   │   │   └── vol_cnn.cpython-311.pyc
│   │   ├── cnn_ed_net.py
│   │   ├── eeg2volume.py
│   │   ├── layers.py
│   │   ├── mels2img_net.py
│   │   ├── model_builder.py
│   │   ├── noise2img_net.py
│   │   ├── reserch.ipynb
│   │   ├── test.gv
│   │   ├── test.py
│   │   ├── vol_cnn.py
│   │   └── volume_reserch.ipynb
│   ├── models_weights
│   │   └── ae_eeg_net_params.pt
│   ├── torchsets
│   │   ├── eeg2img_dataset.py
│   │   ├── eeg2volume_dataset.py
│   │   └── volume_dataset.py
│   ├── trainers
│   │   ├── ae_models_trainer.py
│   │   ├── eeg2volume_trainer.py
│   │   ├── gan_models_trainer.py
│   │   └── volume_ae_trainer.py
│   └── utils
│       ├── data_formater.py
│       ├── eeg2volume_formate.py
│       └── volume_formater.py
└── test.gv
```


