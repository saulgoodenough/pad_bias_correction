
Code for paper [Age-level bias correction in brain age prediction](https://www.sciencedirect.com/science/article/pii/S2213158223000086).
Detail introduction of datasets, methods and hyper-parameters can be found here:  [Supplementary File of the Paper](https://www.sciencedirect.com/science/article/pii/S2213158223000086#m0005)
### Run commands:
- For running of age prediction methods on UK Biobank, ABIDE or OASIS, run files in corresponding script folder such as:
```shell
python ./pad_bias_correction/code/scripts/range_resnet3d34_sgd_script.py
```
- For running of bias corrections, we give a whole python script:
```shell
python ./pad_bias_correction/code/bia_correction/run_bias_correction.py
```
The detail bias methods can be found in 
```
./pad_bias_correction/utils/bias_correction.py
```


Paper Link: https://doi.org/10.1016/j.nicl.2023.103319. 

Please cite the paper as 
```text
@article{zhang2023age,
  title={Age-level Bias Correction in Brain Age Prediction},
  author={Zhang, Biao and Zhang, Shuqin and Feng, Jianfeng and Zhang, Shihua},
  journal={NeuroImage: Clinical},
  pages={103319},
  year={2023},
  publisher={Elsevier}
}
```

Please contact us via emailing littlebiao@outlook.com or opening a GitHub issue for any questions. 