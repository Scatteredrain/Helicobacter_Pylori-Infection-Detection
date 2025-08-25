# H. pylori-Infection-Detection

This is a PyTorch implementation for **Artificial intelligence–assisted endoscopic diagnosis system for diagnosing Helicobacter pylori infection: a multicenter study**.

## ⚙️ Setup

Our experiments are conducted in a [conda](https://www.anaconda.com/download) environment (python 3.9 is recommended) and you can use the below commands to install necessary dependencies:
```shell
conda create --name new_env python=3.9
pip install -r requirements.txt
```


## 💾 Start Inference on Examples

We have given some cases in directory `./Code-HOPE-AI/cases` for a quick test, by running the follow command:
```bash
cd Code-HOPE-AI
python inference.py
```

The visualization results will be saved to directory `./Code-HOPE-AI/results`.
