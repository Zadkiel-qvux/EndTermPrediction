# Dự đoán điểm cuối kỳ — CS523.N11

> Predict final exam scores from midterm scores using Linear Regression
> implemented from scratch via Computational Graph.

## Live Demo

**[Open App](YOUR_STREAMLIT_CLOUD_LINK_HERE)**

## Mathematical Model

$$\hat{y} = w \cdot x + b$$

$$L = \frac{1}{2N} \sum_{i=1}^{N} (\hat{y}^{(i)} - y^{(i)})^2$$

## Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/cs523-final-prediction
cd cs523-final-prediction
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
model.py   — Linear Regression from scratch (NumPy)
graph.py   — Computational Graph visualizer (matplotlib)
app.py     — Streamlit web application
data/      — Dataset (TRAIN2.xlsx)
```

## Dataset

- 515 samples: midterm score → final score (scale 0–10)
- 80/20 train/test split

## References

- [GeeksforGeeks — Computational Graphs in Deep Learning](https://www.geeksforgeeks.org/computational-graphs-in-deep-learning/)
- CS523.N11 Group 13 slides
