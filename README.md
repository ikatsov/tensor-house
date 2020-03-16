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
![Price Optimization Using RL Animation](https://github.com/ikatsov/algorithmic-marketing-examples/blob/master/resources/hilo-pricing-dqn-training-animation.gif)

*Semantic analysis of customer transactions using NLP methods: \
Word2vec learns products embeddings from sequences of orders*
![Learning Semantic Item Representation from Transactions](https://github.com/ikatsov/algorithmic-marketing-examples/blob/master/resources/item2vec-clustering.gif)

### List of Models 

* Promotions and Advertisements
   * Campaign/Channel Attribution using Adstock Model
   * Customer Lifetime Value (LTV) Modeling using Markov Chain
   * Next Best Action Model using Reinforcement Learning (Fitted Q Iteration)
   * Multi-touch Multi-channel Attribution Model using Deep Learning (LSTM with Attention)
* Search
   * Latent Semantic Analysis (LSA)
* Recommendations
   * Nearest Neighbor User-based Collaborative Filtering
   * Nearest Neighbor Item-based Collaborative Filtering
   * Item2Vec Model using NLP Methods (word2vec)
   * Customer2Vec Model using NLP Methods (doc2vec)
* Pricing and Assortment
  * Markdown Price Optimization
  * Dynamic Pricing using Thompson Sampling
  * Dynamic Pricing with Limited Price Experimentation
  * Price Optimization using Reinforcement Learning (DQN)
* Supply Chain
  * Multi-echelon Inventory Optimization using Reinforcement Learning (DDPG, TD3)

### Approach
* The most basic models come from Introduction to Algorithmic Marketing book. Book's website - https://algorithmicweb.wordpress.com/
* More advanced models use deep learning techniques to analyze event sequences (e.g. clickstream) and reinforcement learning for optimization (e.g. safety stock management policy)
* Almost all models are based on industrial reports and real-life case studies

### Community
Follow our twitter feed for notifications about meetups and new developments.

[![Twitter Follow](https://img.shields.io/twitter/follow/DataPointsSMT.svg?style=social)](https://twitter.com/DataPointsSMT) 
