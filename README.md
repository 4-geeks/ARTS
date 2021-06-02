# ARTS
ARTS: Action Recognition using Terminal States. A complete action-recognition framework based-on Google MediaPipe. 🔥🔥
## Requirements
```pip install mediapipe==0.8.3```
## simple usage

![image](https://drive.google.com/uc?export=view&id=1519OdEV2oMkIW0wY2OF4zoF49axlZWbK)
![image](https://drive.google.com/uc?export=view&id=14jx-F_j7iGHyncxWc2Vg2a-3aCEDW_HN)

- Training

download dataset from below

[dataset](https://drive.google.com/file/d/1mTcz4jYpScHwOwGnvBerZDzZzJhRIXGS/view
)

Run commands below to reproduce results on  dataset

```python train.py --IMAGEIN pathofimage --CSVOUTPUT pathofcsvoutput --IMAGEOUT pathofimageoutput ```
- Inference

run inference to obtain result

```python inference.py --videoin pathofvideo --csvin pathofcsvs --dicaction dictionaryofaction ```

- Tutorial

To run tutorial GoogleColab notebook 

