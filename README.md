# NeuralRandUCB

RL project for the fall 2021 semester of the IFT-7201 (RL) course @ Université Laval taught by [Audrey Durand](https://audur2.ift.ulaval.ca).

This repo contains the code needed to carry out the experiments described in the project report. We have taken and adapted the following code: [NeuralTS](https://github.com/ZeroWeight/NeuralTS),  [RandUCB](https://github.com/vaswanis/randucb), [Personalized News Article Recommendation](https://github.com/antonismand/Personalized-News-Recommendation).

In addition, we used a sample of the R6A - Yahoo front page today module user click log dataset to perform the experiment using news article recommendation. This can be obtained from the following [link](https://drive.google.com/file/d/1jkKjUaL3uyOZb1mRYC0JVy50Peja8msm/view?usp=sharing).

This file must then be included in the following directories: articles_recommendation > dataset > R6

Make sure you install the required libraries:

```python
pip install -r requirements.txt
```

Please note that the experiments were carried out with an NVIDIA RTX 3080 graphics card, which requires the most up-to-date version of Pytorch (1.11). This version may not work on your machine, and you may need to install an earlier version.

## Experiment 1: NeuralRandUCB's performance with different configurations

To generate the results:

```python
python neuralrandUCB_vs_neuralTS.py
```

To generate the graphs:

```python
python neuralrandUCB_vs_neuralTS_plot.py
```

## Experiment 2: NeuralRandUCB's performance compared to other algorithms

To generate the results:

```python
python performance.py
python stats_analysis.py
```

To generate the graphs:
```
python performance_plot.py
python stats_analysis_plot.py
```

## Experiment 3: NeuralRandUCB's robustness to reward delay

To generate the results:

```python
python rewards_delay.py
```

To generate the graphs

```python
python rewards_delay_plot.py
```

## Experiment 4: News article recommendation with NeuralRandUCB

To generate the results:

```python
python articles_recommendation/main.py
```
