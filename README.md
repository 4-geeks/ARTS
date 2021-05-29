# ARTS
ARTS: Action Recognition using Terminal States. A complete action-recognition framework based-on Google MediaPipe. 🔥🔥
# Requirements
```pip install mediapipe==0.8.3```
# simple usage
- Training

download dataset from below

https://drive.google.com/file/d/1mTcz4jYpScHwOwGnvBerZDzZzJhRIXGS/view

Run commands below to reproduce results on  dataset

```python train.py --IMAGEIN pathofimage --CSVOUTPUT pathofcsvoutput --IMAGEOUT pathofimageoutput ```
- Inference

run inference to obtain result

```python inference.py --videoin pathofvideo --csvin pathofcsvs --dicaction dictionaryofaction ```
