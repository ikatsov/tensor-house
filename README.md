<p align="center">
  <img src="https://github.com/ikatsov/algorithmic-marketing-examples/blob/master/resources/logo-2000x436px-gr.png" title="TensorHouse Logo">
</p>

### About
TensorHouse is a collection of reference machine learning and optimization models for enterprise operations: marketing, pricing, supply chain, and more. The goal of the project is to provide baseline implementations for industrial, research, and educational purposes.

This project contains the follwoing resources:
* a well-documented repository of reference model implementations, 
* a manually curated list of [important papers](https://github.com/ikatsov/tensor-house/blob/master/resources/papers.md) in modern operations research,
* a manually curated list of [public datasets](https://github.com/ikatsov/tensor-house/blob/master/resources/datasets.md) related to entrerpirse use cases.

### Illustrative Example 
*Strategic price optimization using reinforcement learning*
![Price Optimization Using RL Animation](https://github.com/ikatsov/algorithmic-marketing-examples/blob/master/resources/hilo-pricing-dqn-training-animation.gif)

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

### Contributors
* Ilya Katsov
* Dmytro Zikrach 
