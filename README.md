<p align="center">
  <img src="https://github.com/ikatsov/algorithmic-marketing-examples/blob/master/_resources/logo-banner.png" title="TensorHouse Logo">
</p>

## What is TensorHouse? 
TensorHouse is a collection of reference Jupyter notebooks and demo AI/ML applications for enterprise use cases: marketing, pricing, supply chain, smart manufacturing, and more. The goal of the project is to provide a toolkit for rapid readiness assessment, exploratory data analysis, and prototyping of various modeling approaches for typical enterprise AI/ML/data science projects.

TensorHouse provides the following resources:
* A well-documented repository of reference notebooks and demo applications (prototypes).
* Readiness assessment and requirement gathering questionnaires for typical enterprise AI/ML projects.
* Datasets, data generators, and simulators for rapid prototyping and model evaluation.

TensorHouse focuses mainly on industry-proven solutions that leverage deep learning, reinforcement learning, and casual inference methods and models. Most of these solutions were originally developed either by industry practitioners or by academic researchers who worked in collaboration with leading companies in technology, retail, manufacturing, and other sectors.

## How Does TensorHouse Help?
TensorHouse helps to accelerate the following steps of the solution development:
1. Faster evaluate readiness for specific use cases from the data, integration, and process perspectives using questionnaires and casual inference templates. 
2. Choose candidate methods and models for solving your use cases, evaluate and tailor them using simulators and sample datasets. 
3. Evaluate candidate methods and models on your data, build prototypes, and present preliminary results to stakeholders. 

## What Libs Does TensorHouse Use?
All prototypes and template are implemented in Python using a limited set of standard libraries: 
* Deep learning: mostly `TensorFlow`, some prototypes use `PyTorch`
* Reinforcement learning: `RLlib`
* Causal inference: `DoWhy`, `EconML`
* Probabilistic programming / Bayesian inference: `PyMC`
* Generative AI: `LangChain`
* Traditional ML: `statsmodels`, `scikit-learn`, `LightGBM`
* Basic libs: `NumPy`, `pandas`, `matplotlib`, `seaborn`

## Illustrative Examples

#### Strategic price optimization using reinforcement learning
*DQN learns a Hi-Lo pricing policy that switches between regular and discounted prices:*
<p align="center">
  <img src="https://github.com/ikatsov/tensor-house/blob/master/_resources/hilo-pricing-dqn-training-animation.gif" title="Price Optimization Using RL Animation">
</p>

#### Supply chain optimization using reinforcement learning
*DQN learns how to control procurement and logistics in a simulated environment:*
<p align="center">
  <img src="https://github.com/ikatsov/tensor-house/blob/master/_resources/demo-animation-world-of-supply.gif" title="Price Optimization Using RL Animation">
</p>

#### Supply chain management using large language models
*LLM dynamically writes a python script that invokes multiple APIs to answer user's question:*
<p align="center">
  <img src="https://github.com/ikatsov/tensor-house/blob/master/_resources/demo-animation-sc-control-tower.gif" title="Dynamic Scripting Using LLMs" width="90%">
</p>

#### Anomaly detection in images using autoencoders
*Deep autoencoders produce image reconstructions that facilitate detection of defect locations:*
<p align="center">
  <img src="https://github.com/ikatsov/tensor-house/blob/master/_resources/visual-anomaly-example.png" title="Anomaly Detection in Images">
</p>

## List of Prototypes and Templates
The artifacts listed in this section can help to rapidly evaluate different solution approaches and build prototypes using your datasets. Artifacts are marked with the following qualifiers:
  * 🧪 - artifacts that are particularly suitable for exploratory data analysis, evaluating the strength of causal effects in your data, and determining whether these data is feasible for solving a certain use case or not
  * 🚀 - conceptual prototypes that use advanced methods and not necessarily suitable for productization
  * 📚 - notebooks that demonstrate basic algorithms and intended mainly for educational purposes

#### Promotions, Offers, and Advertisements
These notebooks can be used to analyze the behavior of *individual* customers, calculate customer propensity (affinity) scores, and personalize offers, content, or digital experience. 

* Customer Scoring and Lifetime Value
   * Customer Propensity Scoring Using Deep Learning (LSTM with Attention) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/promotions/propensity-scoring-lstm.ipynb))
   * Customer-level Uplift Modeling Based On Observational Data Using Causal Inference ([notebook](https://github.com/ikatsov/tensor-house/blob/master/promotions/uplift-modeling-observational.ipynb)) (🧪)
   * Customer Lifetime Value (LTV) Estimation Using Markov Chains ([notebook](https://github.com/ikatsov/tensor-house/blob/master/promotions/ltv-markov.ipynb))
   * Customer Lifetime Value (LTV) Estimation Using Bayesian Buy-Till-You-Die (BTYD) Model ([notebook](https://github.com/ikatsov/tensor-house/blob/master/promotions/ltv-btyd-bayesian.ipynb))
* Decision Automation
   * Dynamic Content Personalization Using Contextual Bandits (LinUCB) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/promotions/dynamic-content-personalization-rl.ipynb))
   * Next Best Action Model Using Reinforcement Learning (Fitted Q Iteration) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/promotions/next-best-action-rl.ipynb))

#### Marketing, Customer, and Content Analytics
The notebooks can be used to perform *aggregated* analysis of the customer population or segments, get insights from user-generated content, and optimize marketing budgets.

* Content Analytics
   * Sentiment Analysis Using Basic Transformers ([notebook](https://github.com/ikatsov/tensor-house/blob/master/marketing-analytics/sentiment-analysis.ipynb)) 
   * Virtual Focus Groups Using LLMs ([notebook](https://github.com/ikatsov/tensor-house/blob/master/marketing-analytics/virtual-focus-groups.ipynb)) 
* Customer Behavior Analytics and Embeddings
   * Recency, Frequency, and Monetary Value (RFM) Analysis of Customer Purchases ([notebook](https://github.com/ikatsov/tensor-house/blob/master/marketing-analytics/rfm-analysis.ipynb)) (🧪)
   * Analysis of Customer Behavior Patterns Using LSTM/Transformers ([notebook](https://github.com/ikatsov/tensor-house/blob/master/marketing-analytics/behavior-patterns-analytics-lstm.ipynb))
   * Item2Vec Using Word2vec ([notebook](https://github.com/ikatsov/tensor-house/blob/master/marketing-analytics/item2vec.ipynb))
   * Customer2Vec Using Doc2vec (notebooks: [simulator](https://github.com/ikatsov/tensor-house/blob/master/marketing-analytics/customer2vec-prototype.ipynb), [prototype](https://github.com/ikatsov/tensor-house/blob/master/marketing-analytics/customer2vec.ipynb))
* Media Mix, Attribution, and Budget Optimization
   * Campaign Effect Estimation In Observational Data Using Causal Inference ([notebook](https://github.com/ikatsov/tensor-house/blob/master/marketing-analytics/campaign-effect-observational.ipynb)) (🧪)
   * Media Mix Modeling: Adstock Model for Campaign/Channel Attribution ([notebook](https://github.com/ikatsov/tensor-house/blob/master/marketing-analytics/mediamix-adstock.ipynb))
   * Media Mix Modeling: Bayesian Model with Carryover and Saturation Effects ([notebook](https://github.com/ikatsov/tensor-house/blob/master/marketing-analytics/mediamix-bayesian.ipynb)) (🧪)
   * Multitouch Channel Attribution Model Using Deep Learning (LSTM with Attention) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/marketing-analytics/channel-attribution-lstm.ipynb))

#### Search
These notebooks can be used to create enterprise search, product catalog search, and visual search solutions.  

* Text Search
   * Latent Semantic Analysis (LSA) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/search/text-search-lsa.ipynb)) (📚)
   * Retrieval-augmented Generation (RAG) Using LLMs ([notebook](https://github.com/ikatsov/tensor-house/blob/master/search/retrieval-augmented-generation-llm.ipynb))
* Visual Search
   * Visual Search by Artistic Style (VGG16) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/search/visual-search-artistic-style.ipynb))
   * Visual Search Based on Product Type (EfficientNetB0) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/search/visual-search-similarity.ipynb))
   * Visual Search Using Variational Autoencoders ([notebook](https://github.com/ikatsov/tensor-house/blob/master/search/visual-search-vae.ipynb))
   * Image Search Using a Language-Image Model (CLIP) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/search/image-search-clip.ipynb))
* Structured Data Search
   * Relational Data Querying Using LLMs ([notebook](https://github.com/ikatsov/tensor-house/blob/master/search/relational-data-querying-llm.ipynb))
* Data Preprocessing
   * Product Attribute Discovery, Extraction, and Harmonization Using LLMs ([notebook](https://github.com/ikatsov/tensor-house/blob/master/search/product-attribute-extraction-llm.ipynb))

#### Recommendations
These notebooks can be used to prototype product recommendation solutions. 

* Basic Collaborative Filtering
   * Nearest Neighbor User-based Collaborative Filtering ([notebook](https://github.com/ikatsov/tensor-house/blob/master/recommendations/collaborative-filtering-user-based.ipynb)) (📚)
   * Nearest Neighbor Item-based Collaborative Filtering ([notebook](https://github.com/ikatsov/tensor-house/blob/master/recommendations/collaborative-filtering-item-based.ipynb)) (📚)
* Deep and Hybrid Recommenders
   * Neural Collaborative Filtering - Prototype ([notebook](https://github.com/ikatsov/tensor-house/blob/master/recommendations/deep-recommender-factorization.ipynb)) (📚)
   * Neural Collaborative Filtering - Hybrid Recommender ([notebook](https://github.com/ikatsov/tensor-house/blob/master/recommendations/deep-recommender-ncf.ipynb))
   * Behavior Sequence Transformer ([notebook](https://github.com/ikatsov/tensor-house/blob/master/recommendations/deep-recommender-transformer.ipynb))
   * Graph Recommender Using Node2Vec ([notebook](https://github.com/ikatsov/tensor-house/blob/master/recommendations/deep-recommender-graph-node2vec.ipynb))
  
#### Demand Forecasting
These notebooks can be used to create demand and sales forecasting pipelines. These pipelines can further be used to solve inventory planning, price management, workforce optimization, and financial planning use cases.

* Traditional Methods
   * Demand Forecasting for a Single Entity Using Exponential Smoothing (ETS) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/demand-forecasting/demand-univariate-exponential-smoothing.ipynb))
   * Demand Forecasting for a Single Entity Using Autoregression (ARIMA/SARIMAX) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/demand-forecasting/demand-univariate-arima.ipynb))
   * Demand Forecasting and Price Effect Estimation for Multiple Entities Using Generalized Linear Models ([notebook](https://github.com/ikatsov/tensor-house/blob/master/demand-forecasting/demand-multivariate-glm.ipynb)) (🧪)
* Deep Learning Methods
   * Demand Forecasting for Multiple Entities Using DeepAR ([notebook](https://github.com/ikatsov/tensor-house/blob/master/demand-forecasting/demand-multivariate-deepar.ipynb))
   * Demand Forecasting for a Single Entity Using NeuralProphet ([notebook](https://github.com/ikatsov/tensor-house/blob/master/demand-forecasting/demand-univariate-neural-prophet.ipynb))
* Dynamic Learning 
   * Bayesian Demand Models ([notebook](https://github.com/ikatsov/tensor-house/blob/master/pricing/demand-univariate-bayesian.ipynb))
* Data Preprocessing
   * Demand Types Classification ([notebook](https://github.com/ikatsov/tensor-house/blob/master/demand-forecasting/demand-types-classification.ipynb))
   * Demand Unconstraining ([notebook](https://github.com/ikatsov/tensor-house/blob/master/demand-forecasting/demand-unconstraining.ipynb))

#### Pricing and Assortment
These notebooks can be used to create price optimization, promotion (markdown) optimization, and assortment optimization solutions.

* Static Price, Promotion, and Markdown Optimization
   * Market Response Functions ([notebook](https://github.com/ikatsov/tensor-house/blob/master/pricing/market-response-functions.ipynb)) (📚)
   * Price Optimization for Multiple Products ([notebook](https://github.com/ikatsov/tensor-house/blob/master/pricing/price-optimization-multiple-products.ipynb))
   * Price Optimization for Multiple Time Intervals ([notebook](https://github.com/ikatsov/tensor-house/blob/master/pricing/price-optimization-multiple-time-intervals.ipynb))
* Dynamic Pricing
   * Dynamic Pricing Using Thompson Sampling ([notebook](https://github.com/ikatsov/tensor-house/blob/master/pricing/dynamic-pricing-thompson.ipynb))
   * Dynamic Pricing with Limited Price Experimentation ([notebook](https://github.com/ikatsov/tensor-house/blob/master/pricing/dynamic-pricing-limited-experimentation.ipynb))
   * Price Optimization Using Reinforcement Learning (DQN) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/pricing/price-optimization-using-dqn-reinforcement-learning.ipynb)) (🚀)

#### Supply Chain
These notebooks and applications can be used to develop procurement and inventory allocation solutions, as well as provide supply chain managers with advanced decisions support and automation tools.

   * Single-echelon Inventory Optimization Using (s,Q) and (R,S) Policies ([notebook](https://github.com/ikatsov/tensor-house/blob/master/supply-chain/single-echelon-sQ-RS.ipynb))
   * Inventory Allocation Optimization ([notebook](https://github.com/ikatsov/tensor-house/blob/master/supply-chain/inventory-allocation.ipynb))
   * Multi-echelon Inventory Optimization Using Reinforcement Learning (DDPG, TD3) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/supply-chain/supply-chain-reinforcement-learning.ipynb)) (🚀)
   * Supply Chain Simulator for Reinforcement Learning Based Optimization (PPO) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/supply-chain/world_of_supply/world-of-supply.ipynb)) (🚀)
   * Supply Chain Control Tower Using LLMs ([notebook](https://github.com/ikatsov/tensor-house/blob/master/supply-chain/control_center_llm/control-center-llm.ipynb)) (🚀)
   
#### Smart Manufacturing
These notebooks can be used to prototype visual quality control and predictive maintenance solutions.

   * Noise Reduction in Multivariate Timer Series Using Linear Autoencoder (PCA) ([notebook](https://github.com/ikatsov/tensor-house/blob/master/smart-manufacturing/noise-reduction-pca.ipynb))
   * Remaining Useful Life Prediction Using Convolution Networks ([notebook](https://github.com/ikatsov/tensor-house/blob/master/smart-manufacturing/remaining-useful-life-prediction.ipynb))
   * Anomaly Detection in Time Series ([notebook](https://github.com/ikatsov/tensor-house/blob/master/smart-manufacturing/anomaly-detection-time-series.ipynb))
   * Anomaly Detection in Images Using Autoencoders ([notebook](https://github.com/ikatsov/tensor-house/blob/master/smart-manufacturing/visual-quality-control.ipynb))

## List of Questionnaires
These questionnaires can be used to assess readiness for typical AI/ML projects and collect the requirements for creating roadmaps and estimates.
   * Demand Sensing and Forecasting ([document](https://docs.google.com/document/d/1cd0n9L1pjCSGXgCS0CC3k9DpGiVzNhhmFlyhTXvnqws/edit))
   * Price and Promotion Optimization ([document](https://docs.google.com/document/d/1mHOANKSavhxCn3Y_R9WYnCSWavnmJJdoRj9aFPZlFqE/edit))
   * Next Best Action ([document](https://docs.google.com/document/d/10bo1wUAO8ctjaQqq2Y0mLo9k9itqwjyQysDOu_-C_rA/edit))

## More Documentation
* The most basic models are described the *Introduction to Algorithmic Marketing*. 
    * Book's website - https://www.algorithmicmarketingbook.com/
* More advanced models that use deep learning and reinforcement learning techniques are described in *The Theory and Practice of Enterprise AI*. 
    * Book's website - https://www.enterprise-ai-book.com/
* Templates for basic data science and ML task are available in [TensorHouseBasic](https://github.com/ikatsov/tensor-house-basic) repository. 
* Most notebooks contain references to specific research papers, industrial reports, and real-world case studies.
    * A manually curated list of [important papers](https://github.com/ikatsov/tensor-house/blob/master/_resources/papers.md) in enterprise AI.
    * A manually curated list of [public datasets](https://github.com/ikatsov/tensor-house/blob/master/_resources/datasets.md) related to enterprise use cases.
* Follow LinkedIn and X (Twitter) for notifications about new developments and releases.
<div id="badges" align="center">
  <a href="https://www.linkedin.com/in/ilya-katsov/">
    <img src="https://img.shields.io/badge/LinkedIn-blue?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn Badge"/>
  </a>
  <a href="https://twitter.com/ikatsov">
    <img src="https://img.shields.io/badge/Twitter-blue?style=for-the-badge&logo=twitter&logoColor=white" alt="Twitter Badge"/>
  </a>
</div>

## Contribution
We warmly welcome contributions, such as implementations of new use cases, advanced features and usability improvements for existing use cases, or enhancements to documentation.
