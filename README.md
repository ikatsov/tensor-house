<p align="center">
  <img src="https://github.com/ikatsov/algorithmic-marketing-examples/blob/master/resources/logo-2000x436px-gr.png" title="TensorHouse Logo">
</p>

### About
TensorHouse is a collection of reference machine learning and optimization models for enterprise operations: marketing, pricing, supply chain, and more. The goal of the project is to provide baseline implementations for industrial, research, and educational purposes.

The project focuses on models, techniques, and datasets that were originally developed either by industry practitioners or by academic researchers who worked in collaboration with leading companies in technology, retail, manufacturing, and other sectors. In other words, TensorHouse focuses mainly on industry-proven methods and models rather than on theoretical research.

TensorHouse contains the following resources:
* a well-documented repository of reference model implementations, 
* a manually curated list of [important papers](https://github.com/ikatsov/tensor-house/blob/master/resources/papers.md) in modern operations research,
* a manually curated list of [public datasets](https://github.com/ikatsov/tensor-house/blob/master/resources/datasets.md) related to enterprise use cases.

### Illustrative Examples
*Strategic price optimization using reinforcement learning: \
DQN learns a Hi-Lo pricing policy that switches between regular and discounted prices*
![Price Optimization Using RL Animation](https://github.com/ikatsov/tensor-house/blob/master/resources/hilo-pricing-dqn-training-animation.gif)

*Supply chain optimization using reinforcement learning: \
World Of Supply simulation environment*
![Price Optimization Using RL Animation](https://github.com/ikatsov/tensor-house/blob/master/resources/demo-animation-world-of-supply.gif)

*Demand decomposition using Bayesian Structural Time Series*
![Demand Decomposition Example](https://github.com/ikatsov/tensor-house/blob/master/resources/demand-decomposition-example.png)

### List of Models 

* Promotions and Advertisements
   * Media Mix Modeling: Basic Adstock Model for Campaign/Channel Attribution ([notebook](https://github.com/ikatsov/tensor-house/blob/master/promotions/mediamix-adstock.ipynb))
   * Media Mix Modeling: Bayesian Model with Carriover and Saturation Effects ([notebook](https://github.com/ikatsov/tensor-house/blob/master/promotions/mediamix-bayesian.ipynb))
   * Dynamic Content Personalization using Contextual Bandits (LinUCB) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/promotions/dynamic-content-personalization-rl.ipynb))
   * Customer Lifetime Value (LTV) Modeling using Markov Chain ([notebook](https://github.com/ikatsov/tensor-house/blob/master/promotions/markov-ltv.ipynb))
   * Next Best Action Model using Reinforcement Learning (Fitted Q Iteration) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/promotions/next-best-action-rl.ipynb))
   * Multitouch Channel Attribution Model using Deep Learning (LSTM with Attention) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/promotions/channel-attribution-lstm.ipynb))
   * Customer Churn Analysis and Prediction using Deep Learning (LSTM with Attention) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/promotions/churn-prediction-lstm.ipynb))
* Search
   * Latent Semantic Analysis (LSA) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/search/lsa.ipynb))
   * Image Search by Artistic Style (VGG16) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/search/image-artistic-style-similarity.ipynb))
* Recommendations
   * Nearest Neighbor User-based Collaborative Filtering ([notebook](https://github.com/ikatsov/tensor-house/blob/master/recommendations/user-based-cf.ipynb))
   * Nearest Neighbor Item-based Collaborative Filtering ([notebook](https://github.com/ikatsov/tensor-house/blob/master/recommendations/item-based-cf.ipynb))
   * Item2Vec Model using NLP Methods (word2vec) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/recommendations/item2vec.ipynb))
   * Customer2Vec Model using NLP Methods (doc2vec) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/recommendations/customer2vec.ipynb))
   * Deep Learning Recommender (notebooks
[1](https://github.com/ikatsov/tensor-house/blob/master/recommendations/deep-recommender.ipynb)
[2](https://github.com/ikatsov/tensor-house/blob/master/recommendations/factorization-sgd-neural.ipynb))
* Pricing and Assortment
   * Market Response Functions ([notebook](https://github.com/ikatsov/tensor-house/blob/master/pricing/market-response-functions.ipynb))
   * Price Elasticity Analysis ([notebook](https://github.com/ikatsov/tensor-house/blob/master/pricing/price-elasticity.ipynb))
   * Price Optimization for Multiple Products ([notebook](https://github.com/ikatsov/tensor-house/blob/master/pricing/price-optimization-multiple-products.ipynb))
   * Price Optimization for Multiple Time Intervals ([notebook](https://github.com/ikatsov/tensor-house/blob/master/pricing/price-optimization-multiple-time-intervals.ipynb))
   * Dynamic Pricing using Thompson Sampling ([notebook](https://github.com/ikatsov/tensor-house/blob/master/pricing/dynamic-pricing-thompson.ipynb))
   * Dynamic Pricing with Limited Price Experimentation ([notebook](https://github.com/ikatsov/tensor-house/blob/master/pricing/dynamic-pricing-limited-experimentation.ipynb))
   * Bayesian Demand Models ([notebook](https://github.com/ikatsov/tensor-house/blob/master/pricing/bayesian-demand-models.ipynb))
   * Demand Uncostraining ([notebook](https://github.com/ikatsov/tensor-house/blob/master/pricing/demand-unconstraining.ipynb))
   * Price Optimization using Reinforcement Learning (DQN) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/pricing/price-optimization-using-dqn-reinforcement-learning.ipynb))
* Supply Chain
   * Single-echelon Inventory Optimization using (s,Q) and (R,S) Policies ([notebook](https://github.com/ikatsov/tensor-house/blob/master/supply-chain/single-echelon-sQ-RS.ipynb))
   * Multi-echelon Inventory Optimization using Reinforcement Learning (DDPG, TD3) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/supply-chain/supply-chain-reinforcement-learning.ipynb))
   * Supply Chain Simulator for Reinforcement Learning Based Optimization (PPO) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/supply-chain/world-of-supply/world-of-supply.ipynb))
* Enterpirse Time Series Analysis
   * Demand Forecasting Using ARIMA and SARIMA (notebooks
[1](https://github.com/ikatsov/tensor-house/blob/master/basic-components/time-series/arima-part-1-algorithm.ipynb)
[2](https://github.com/ikatsov/tensor-house/blob/master/basic-components/time-series/arima-part-2-use-case.ipynb))
   * Demand Decomposition and Forecasting using Bayesian Structural Time Series (BSTS) (notebooks
[1](https://github.com/ikatsov/tensor-house/blob/master/basic-components/time-series/bsts-part-1-decomposition.ipynb)
[2](https://github.com/ikatsov/tensor-house/blob/master/basic-components/time-series/bsts-part-2-forecasting.ipynb)
[3](https://github.com/ikatsov/tensor-house/blob/master/basic-components/time-series/bsts-part-3-forecasting-prophet.ipynb)
[4](https://github.com/ikatsov/tensor-house/blob/master/basic-components/time-series/bsts-part-4-forecasting-pymc3.ipynb))
   * Forecasting and Decomposition using Gradient Boosted Decision Trees (GBDT) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/basic-components/time-series/gbdt-forecasting.ipynb))
   * Forecasting and Decomposition using LSTM with Attention ([notebook](https://github.com/ikatsov/tensor-house/blob/master/basic-components/time-series/lstm-forecasting.ipynb))
   * Forecasting and Decomposition using VAR/VEC models (notebooks
[1](https://github.com/ikatsov/tensor-house/blob/master/basic-components/time-series/var-part-1-forecasting-decomposition.ipynb)
[2](https://github.com/ikatsov/tensor-house/blob/master/basic-components/time-series/var-part-2-market-data.ipynb))

### Approach
* The most basic models come from Introduction to Algorithmic Marketing book. Book's website - https://algorithmicweb.wordpress.com/
* More advanced models use deep learning techniques to analyze event sequences (e.g. clickstream) and reinforcement learning for optimization (e.g. safety stock management policy)
* Most models are based on industrial reports and real-life case studies

### Community
Follow our twitter feed for notifications about meetups and new developments.

[![Twitter Follow](https://img.shields.io/twitter/follow/DataPointsSMT.svg?style=social)](https://twitter.com/DataPointsSMT) 
