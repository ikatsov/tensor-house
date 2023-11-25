<p align="center">
  <img src="https://github.com/ikatsov/algorithmic-marketing-examples/blob/master/_resources/logo-2000x436px-gr.png" title="TensorHouse Logo">
</p>

### About
TensorHouse is a collection of reference machine learning and optimization models for enterprise operations: marketing, pricing, supply chain, and more. The goal of the project is to provide baseline implementations for industrial, research, and educational purposes.

The project focuses on models, techniques, and datasets that were originally developed either by industry practitioners or by academic researchers who worked in collaboration with leading companies in technology, retail, manufacturing, and other sectors. In other words, TensorHouse focuses mainly on industry-proven methods and models rather than on theoretical research.

TensorHouse contains the following resources:
* a well-documented repository of reference model implementations, 
* a manually curated list of [important papers](https://github.com/ikatsov/tensor-house/blob/master/_resources/papers.md) in modern operations research,
* a manually curated list of [public datasets](https://github.com/ikatsov/tensor-house/blob/master/_resources/datasets.md) related to enterprise use cases.

### Illustrative Examples
*Strategic price optimization using reinforcement learning: \
DQN learns a Hi-Lo pricing policy that switches between regular and discounted prices*
![Price Optimization Using RL Animation](https://github.com/ikatsov/tensor-house/blob/master/_resources/hilo-pricing-dqn-training-animation.gif)

*Supply chain optimization using reinforcement learning: \
World Of Supply simulation environment*
![Price Optimization Using RL Animation](https://github.com/ikatsov/tensor-house/blob/master/_resources/demo-animation-world-of-supply.gif)

*Anomaly detection in images using autoencoders*
![Anomaly Detection in Images](https://github.com/ikatsov/tensor-house/blob/master/_resources/visual-anomaly-example.png)

### List of Models 

* Promotions and Advertisements
   * Media Mix Modeling: Basic Adstock Model for Campaign/Channel Attribution ([notebook](https://github.com/ikatsov/tensor-house/blob/master/promotions/mediamix-adstock.ipynb))
   * Media Mix Modeling: Bayesian Model with Carryover and Saturation Effects ([notebook](https://github.com/ikatsov/tensor-house/blob/master/promotions/mediamix-bayesian.ipynb))
   * Dynamic Content Personalization Using Contextual Bandits (LinUCB) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/promotions/dynamic-content-personalization-rl.ipynb))
   * Customer Lifetime Value (LTV) Modeling Using Markov Chain ([notebook](https://github.com/ikatsov/tensor-house/blob/master/promotions/markov-ltv.ipynb))
   * Next Best Action Model Using Reinforcement Learning (Fitted Q Iteration) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/promotions/next-best-action-rl.ipynb))
   * Multitouch Channel Attribution Model Using Deep Learning (LSTM with Attention) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/promotions/channel-attribution-lstm.ipynb))
   * Customer Propensity Scoring Using Deep Learning (LSTM with Attention) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/promotions/propensity-scoring-lstm.ipynb))

* Search
   * Latent Semantic Analysis (LSA) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/search/text-search-lsa.ipynb))
   * Visual Search by Artistic Style (VGG16) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/search/visual-search-artistic-style.ipynb))
   * Visual Search based on Product Type (EfficientNetB0) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/search/visual-search-similarity.ipynb))
   * Visual Search Using Variational Autoencoders ([notebook](https://github.com/ikatsov/tensor-house/blob/master/search/visual-search-vae.ipynb))
   * Image Search Using a Language-Image Model (CLIP) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/search/image-search-clip.ipynb))
   * Product Attribute Discovery, Extraction, and Harmonization Using LLMs ([notebook](https://github.com/ikatsov/tensor-house/blob/master/search/product-attribute-extraction-llm.ipynb))
   * Relational Data Querying Using LLMs ([notebook](https://github.com/ikatsov/tensor-house/blob/master/search/relational-data-querying-llm.ipynb))
   * Retrieval-augmented Generation (RAG) Using LLMs ([notebook](https://github.com/ikatsov/tensor-house/blob/master/search/retrieval-augmented-generation-llm.ipynb))
   * Retrieval-augmented Generation (RAG) Using LLMs Agents ([notebook](https://github.com/ikatsov/tensor-house/blob/master/search/retrieval-augmented-generation-llm-agents.ipynb))

* Recommendations
   * Nearest Neighbor User-based Collaborative Filtering ([notebook](https://github.com/ikatsov/tensor-house/blob/master/recommendations/user-based-cf.ipynb))
   * Nearest Neighbor Item-based Collaborative Filtering ([notebook](https://github.com/ikatsov/tensor-house/blob/master/recommendations/item-based-cf.ipynb))
   * Item2Vec Model Using NLP Methods (word2vec) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/recommendations/item2vec.ipynb))
   * Customer2Vec Model Using NLP Methods (doc2vec) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/recommendations/customer2vec.ipynb))
   * Neural Collaborative Filtering - Prototype ([notebook](https://github.com/ikatsov/tensor-house/blob/master/recommendations/deep-recommender-factorization.ipynb))
   * Neural Collaborative Filtering - Hybrid Recommender ([notebook](https://github.com/ikatsov/tensor-house/blob/master/recommendations/deep-recommender-ncf.ipynb))
   * Behavior Sequence Transformer ([notebook](https://github.com/ikatsov/tensor-house/blob/master/recommendations/deep-recommender-transformer.ipynb))
   * Graph Recommender Using Node2Vec ([notebook](https://github.com/ikatsov/tensor-house/blob/master/recommendations/deep-recommender-graph-node2vec.ipynb))
   
* Demand Forecasting
   * Demand Forecasting Using Exponential Smoothing (ETS) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/demand-forecasting/demand-exponential-smoothing.ipynb))
   * Demand Forecasting and Price Elasticity Analysis Using Time Series Regression ([notebook](https://github.com/ikatsov/tensor-house/blob/master/demand-forecasting/price-regression-elasticity.ipynb))
   * Demand Forecasting Using DeepAR ([notebook](https://github.com/ikatsov/tensor-house/blob/master/demand-forecasting/demand-deepar.ipynb))
   * Demand Forecasting Using NeuralProphet ([notebook](https://github.com/ikatsov/tensor-house/blob/master/demand-forecasting/demand-neural-prophet.ipynb))
   * Demand Unconstraining ([notebook](https://github.com/ikatsov/tensor-house/blob/master/demand-forecasting/demand-unconstraining.ipynb))

* Pricing and Assortment
   * Market Response Functions ([notebook](https://github.com/ikatsov/tensor-house/blob/master/pricing/market-response-functions.ipynb))
   * Price Optimization for Multiple Products ([notebook](https://github.com/ikatsov/tensor-house/blob/master/pricing/price-optimization-multiple-products.ipynb))
   * Price Optimization for Multiple Time Intervals ([notebook](https://github.com/ikatsov/tensor-house/blob/master/pricing/price-optimization-multiple-time-intervals.ipynb))
   * Dynamic Pricing Using Thompson Sampling ([notebook](https://github.com/ikatsov/tensor-house/blob/master/pricing/dynamic-pricing-thompson.ipynb))
   * Dynamic Pricing with Limited Price Experimentation ([notebook](https://github.com/ikatsov/tensor-house/blob/master/pricing/dynamic-pricing-limited-experimentation.ipynb))
   * Bayesian Demand Models ([notebook](https://github.com/ikatsov/tensor-house/blob/master/pricing/bayesian-demand-models.ipynb))
   * Price Optimization Using Reinforcement Learning (DQN) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/pricing/price-optimization-using-dqn-reinforcement-learning.ipynb))

* Supply Chain
   * Single-echelon Inventory Optimization Using (s,Q) and (R,S) Policies ([notebook](https://github.com/ikatsov/tensor-house/blob/master/supply-chain/single-echelon-sQ-RS.ipynb))
   * Multi-echelon Inventory Optimization Using Reinforcement Learning (DDPG, TD3) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/supply-chain/supply-chain-reinforcement-learning.ipynb))
   * Inventory Allocation Optimization ([notebook](https://github.com/ikatsov/tensor-house/blob/master/supply-chain/inventory-allocation.ipynb))
   * Supply Chain Simulator for Reinforcement Learning Based Optimization (PPO) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/supply-chain/world_of_supply/world-of-supply.ipynb))
   * Supply Chain Control Tower Using LLMs ([notebook](https://github.com/ikatsov/tensor-house/blob/master/supply-chain/control_center_llm/control-center-llm.ipynb))
   
* Anomaly Detection
    * Noise Reduction in Multivariate Timer Series Using Linear Autoencoder (PCA) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/anomaly-detection/noise-reduction-pca.ipynb))
    * Remaining Useful Life Prediction Using Convolution Networks ([notebook](https://github.com/ikatsov/tensor-house/blob/master/anomaly-detection/remaining-useful-life-prediction.ipynb))
    * Anomaly Detection in Time Series ([notebook](https://github.com/ikatsov/tensor-house/blob/master/anomaly-detection/anomaly-detection-time-series.ipynb))
    * Anomaly Detection in Images using Autoencoders ([notebook](https://github.com/ikatsov/tensor-house/blob/master/anomaly-detection/visual-quality-control.ipynb))

### Basic Components

* Generic Regression and Classification Models
    * Neural Network with Vector Inputs ([notebook](https://github.com/ikatsov/tensor-house/blob/master/_basic-components/regression/vector-models.ipynb))
    * Neural Network with Sequential Inputs (ConvNet, LSTM, Attention) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/_basic-components/regression/sequence-models.ipynb))

* Enterprise Time Series Analysis
   * Forecasting Using ARIMA and SARIMA (notebooks
[1](https://github.com/ikatsov/tensor-house/blob/master/_basic-components/time-series/arima-part-1-algorithm.ipynb)
[2](https://github.com/ikatsov/tensor-house/blob/master/_basic-components/time-series/arima-part-2-use-case.ipynb))
   * Decomposition and Forecasting using Bayesian Structural Time Series (BSTS) (notebooks
[1](https://github.com/ikatsov/tensor-house/blob/master/_basic-components/time-series/bsts-part-1-decomposition.ipynb)
[2](https://github.com/ikatsov/tensor-house/blob/master/_basic-components/time-series/bsts-part-2-forecasting.ipynb)
[3](https://github.com/ikatsov/tensor-house/blob/master/_basic-components/time-series/bsts-part-3-forecasting-prophet.ipynb)
[4](https://github.com/ikatsov/tensor-house/blob/master/_basic-components/time-series/bsts-part-4-forecasting-pymc3.ipynb))
   * Forecasting and Decomposition using Gradient Boosted Decision Trees (GBDT) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/_basic-components/time-series/gbdt-forecasting.ipynb))
   * Forecasting and Decomposition using LSTM with Attention ([notebook](https://github.com/ikatsov/tensor-house/blob/master/_basic-components/time-series/lstm-forecasting.ipynb))
   * Forecasting and Decomposition using VAR/VEC models (notebooks
[1](https://github.com/ikatsov/tensor-house/blob/master/_basic-components/time-series/var-part-1-forecasting-decomposition.ipynb)
[2](https://github.com/ikatsov/tensor-house/blob/master/_basic-components/time-series/var-part-2-market-data.ipynb))

### Approach
* The most basic models come from the *Introduction to Algorithmic Marketing* book. 
    * Book's website - https://www.algorithmicmarketingbook.com/
* More advanced models use deep learning and reinforcement learning techniques from *The Theory and Practice of Enterprise AI* book. 
    * Book's website - https://www.enterprise-ai-book.com/
* Most models are based on industrial reports and real-life case studies

### Community
Follow our Twitter feed for notifications about new developments.

[![Twitter Follow](https://img.shields.io/twitter/follow/ikatsov.svg?style=social)](https://twitter.com/ikatsov) 
