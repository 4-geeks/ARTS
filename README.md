# ARTS
ARTS: Action Recognition using Terminal States. A complete action-recognition framework based-on Google MediaPipe. 🔥🔥
## Requirements
```pip install mediapipe==0.8.3```
## simple usage

![image](https://drive.google.com/uc?export=view&id=1aisWCj2x9mqKEGYX28AUe2YwQ_-EOICW)
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

<a href="https://colab.research.google.com/github/4-geeks/ARTS/blob/main/ARTSTUTORIAL.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
