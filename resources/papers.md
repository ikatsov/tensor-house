# Imortant Publications in Enterprise Data Science / ML / AI
What follows is a manually curated list of papers in modern data science and operations research that are worth reading. 
* We mainly focus on industrial reports, papers, and case studies, not purely theoretical works. Many entries are explitly tagged with **\[CompanyYear\]** prefix to provide a clearer picture of industrial adoption or affiliation.
* Theoretical Foundations section focuses on papers that are most relevant in the context of operations research. There is no goal to create a comprehensive list of deep learning or reinforcement learning papers in general.  

---
# Customer Intelligence and Personalization

### Representation Learning and Semantic Spaces
1. **[ Microsoft2015 ]** Barkan O., Koenigstein N. -- Item2Vec: Neural Item Embedding for Collaborative Filtering
2. **[ Myntra2016 ]** Arora S., Warrier D. -- Decoding Fashion Contexts Using Word Embeddings, 2016
3. **[ Rakuten2016 ]** Phi V., Chen L., Hirate Y. -- Distributed Representation-based Recommender Systems in E-commerce, 2016
4. **[ RTBHouse2016 ]** Zolna K., Bartlomiej R. -- User2vec: user modeling using LSTM networks, 2016
5. **[ MediaGamma2017 ]** Stiebellehner S., Wang J, Yuan S. -- Learning Continuous User Representations through Hybrid Filtering with doc2vec, 2017
6. **[ Yandex2018 ]** Seleznev N., Irkhin I., Kantor V. -- Automated extraction of rider’s attributes based on taxi mobile application activity logs, 2018
7. **[ BBVA2018 ]** Baldassini L., Serrano J. -- client2vec: Towards Systematic Baselines for Banking Applications
8. **[ Santander2018 ]** Mancisidor R., Kampffmeyer M., Aas K., Jenssen R. -- Segment-Based Credit Scoring Using Latent Clusters in the Variational Autoencoder, 2018
9. **[ Zalando2017 ]** Lang T., Rettenmeier M. -- Understanding Consumer Behavior with Recurrent Neural Networks, 2017
10. Netzer O., Lattin J., Srinivasan V. -- A Hidden Markov Model of Customer Relationship Dynamics, 2008

### Personalized Recommendations, Ads, and Promotions

#### Basic Methods
1. **[ BookingCom2019 ]** Bernardi L., Mavridis T., Estevez P. -- 150 Successful Machine Learning Models: 6 Lessons Learned at Booking.com, 2019
2. **[ Amazon2003 ]** Linden G., Smith B., and York J. -- Amazon.com Recommendations: Item-to-Item Collaborative Filtering, 2003
3. **[ Netflix2009 ]** Koren Y. -- The BellKor Solution to the Netflix Grand Prize, 2009
4. **[ Netflix2009 ]** Koren Y., Bell R., and Volinsky C. -- Matrix Factorization Techniques for Recommender Systems, 2009
5. Pfeifer P., Carraway R. -- Modeling Customer Relationships as Markov Chains, 2000
6. Rendle S. -- Factorization Machines, 2010

#### Reinforcement Learning Methods
1. **[ Facebook2019 ]** Gauci J., et al -- Horizon: Facebook's Open Source Applied Reinforcement Learning Platform, 2019
2. **[ Adobe2015 ]** G. Theocharous, P. Thomas, and M. Ghavamzadeh -- Personalized Ad Recommendation Systems for Life-Time Value Optimization with Guarantees, 2015
3. **[ Criteo2018 ]** Rohde D., Bonner S., Dunlop T., Vasile F., Karatzoglou A. -- RecoGym: A Reinforcement Learning Environment for the Problem of Product Recommendation in Online Advertising, 2018
4. **[ Spotify2018 ]** McInerney J., Lacker B., Hansen S., Higley K., Bouchard H., Gruson A., Mehrotra R. -- Explore, Exploit, and Explain: Personalizing Explainable Recommendations with Bandits, 2018
5. **[ Google2018 ]** Chen M., Beutel A., Covington P., Jain S., Belletti F., Chi E. -- Top-K Off-Policy Correction for a REINFORCE Recommender System, 2018

#### Deep Learning Methods
1. **[ Airbnb2019 ]** Du G. -- Discovering and Classifying In-app Message Intent at Airbnb, 2019
2. **[ Google2016 ]** Covington P., Adams J., Sargin E. -- Deep Neural Networks for YouTube Recommendations, 2016
3. **[ Netflix2016 ]** Hidasi B., Karatzoglou A., Baltrunas L., Tikk D. -- Session-based Recommendations with Recurrent Neural Networks, 2016
4. **[ Google2017 ]** Wu C., Ahmed A., Beutel A., Smola A., Jing H. -- Recurrent Recommender Networks, 2017
5. **[ Snap2018 ]** Yang C., Shi X., Luo J. and Han J. -- I Know You'll Be Back: Interpretable New User Clustering and Churn Prediction on a Mobile Social Application, 2018
6. Zhang S., Yao L., Sun A., Tay Y. -- Deep Learning based Recommender System: A Survey and New Perspectives, 2019

#### Deep Graph Learning Methods
1. **[ Pinterest2018 ]** Ying R., He R., Chen K., Eksombatchai P., Hamilton W., Leskovec J. -- Graph Convolutional Neural Networks for Web-Scale Recommender Systems, 2018
2. **[ Pinterest2017 ]** Eksombatchai C., Jindal P., Liu J., Liu Y., Sharma R., Sugnet C., Ulrich M., Leskovec J. -- Pixie: A System for Recommending 3+ Billion Items to 200+ Million Users in Real-Time, 2017
3. **[ Uber2019 ]** Jain A., Liu I., Sarda A., and Molino P. -- Food Discovery with Uber Eats: Using Graph Learning to Power Recommendations, 2019

#### Evaluation and Measurement
1. **[ Netflix2018 ]** Steck H. -- Calibrated Recommendations, 2018
2. Chaney A., Stewart B., Engelhardt B. -- How Algorithmic Confounding in Recommendation Systems Increases Homogeneity and Decreases Utility, 2017

### Channel Attribution, Marketing Spend Optimization, and Ad Bidding
1. **[ Adobe2018 ]** N. Li, S. K. Arava, C. Dong, Z. Yan, and A. Pani -- Deep Neural Net with Attention for Multi-channel Multi-touch Attribution, 2018
2. **[ Miaozhen2018 ]** Ren K., et al. -- Learning Multi-touch Conversion Attribution with Dual-attention Mechanisms for Online Advertising, 2018
3. **[ Alibaba2018 ]** D. Wu, X. Chen, X. Yang, H. Wang, Q. Tan, X. Zhang, J. Xu, and K. Gai -- Budget Constrained Bidding by Model-free Reinforcement Learning in Display Advertising, 2018
4. **[ Alibaba2018]** Zhao J., Qiu G., Guan Z., Zhao W. and He X. -- Deep Reinforcement Learning for Sponsored Search Real-time Bidding, 2018
5. **[ iProspect2004 ]** Kitts B., Leblanc B. -- Optimial Bidding on Keyword Auctions, 2004
6. **[ Dstillery2012 ]** Dalessandro B., Perlich C., Stitelman O., Provost F. -- Causally motivated attribution for online advertising, 2012
7. **[ TurnInc2011 ]** Shao X., Li L. -- Data-driven Multi-touch Attribution Models, 2011
8. **[ IntegralAds2015 ]** Hill D., Moakler R., Hubbard A., Tsemekhman V., Provost F., Tsemekhman K. -- Measuring Causal Impact of Online Actions via Natural Experiments: Application to Display Advertising, 2015

---
# Price Management and Optimization

### Demand Analysis and Forecasting
1. **[ CVS2007 ]** Ailawadi K., Harlam B., César J., Trounce D. -- Quantifying and Improving Promotion Effectiveness at CVS, 2007
2. **[ Lexus2010 ]** van Heerde H., Srinivasan S., Dekimpe M. -- Estimating Cannibalization Rates for Pioneering Innovations, 2010
3. **[ AlbertHeijn2006 ]** Kök A., Fisher M. -- Demand Estimation and Assortment Optimization Under Substitution: Methodology and Application, 2006
4. **[ Uber2017 ]** Zhu L., Laptev N. -- Deep and Confident Prediction for Time Series at Uber, 2017
6. **[ Uber2017 ]** Laptev N., Yosinski J., Li L., Smyl S. -- Time-series Extreme Event Forecasting with Neural Networks at Uber, 2017
7. Rodrigues F., Markou I., Pereira F. -- Combining Time-Series and Textual Data for Taxi Demand Prediction in Event Areas: A Deep Learning Approach, 2018
8. Ghobbar A., Friend C. -- Evaluation of Forecasting Methods for Intermittent Parts Demand in the Field of Aviation: A Predictive Model, 2002

### Dynamic Pricing
1. **[ Groupon2017 ]** Cheung W., Simchi-Levi D., and Wang H. -- Dynamic Pricing and Demand Learning with Limited Price Experimentation, 2017
2. **[ Harward2017 ]** Ferreira K., Simchi-Levi D., and Wang H. -- Online Network Revenue Management Using Thompson Sampling, November 2017
3. **[ Walmart2018 ]** Ganti R., Sustik M., Quoc T., Seaman B. -- Thompson Sampling for Dynamic Pricing, February 2018
4. **[ RueLaLa2015 ]** Ferreira K. J., Lee B., and Simchi-Levi D. -- Analytics for an Online Retailer: Demand Forecasting and Price Optimization, November 2015
5. **[ Airbnb2018 ]** Srinivasan S. -- Learning Market Dynamics for Optimal Pricing, 2018
6. **[ Uber2017 ]** Chen L. -- Measuring Algorithms in Online Marketplaces, 2017
7. **[ Amazon2016 ]** Chen L., Mislove A., Wilson C. -- An Empirical Analysis of Algorithmic Pricing on Amazon Marketplace, 2016

### Macroeconomic Impact of Algorithmic Pricing
1. Cavallo A. -- More Amazon Effects: Online Competition and Pricing Behaviors, 2018

---
# Inventory and Supply Chain Management

1. Oroojlooyjadid A., Snyder L., Takáč M. -- Applying Deep Learning to the Newsvendor Problem, 2018
2. Kemmer L., et al. -- Reinforcement learning for supply chain optimization, 2018 

---
# Theoretical Foundations

### Foundations of Reinforcement Learning
1. Russo D., Roy B., Kazerouni A., Osband I., and Wen Z. -- A Tutorial on Thompson Sampling, November 2017
2. Riedmiller M. -- Neural Fitted Q Iteration - First Experiences with a Data Efficient Neural Reinforcement Learning Method, 2005
3. Mnih V., et al. -- Human-level Control Through Deep Reinforcement Learning, 2015
4. Silver D., Lever G., Heess N., Degris T., Wierstra D., Riedmiller M. -- Deterministic Policy Gradient Algorithms, 2014
5. Lillicrap T., Hunt J., Pritzel A., Heess N., Erez T., Tassa Y., Silver D., Wierstra D. -- Continuous Control with Deep Reinforcement Learning, 2015
6. Hessel M, et al. -- Rainbow: Combining Improvements in Deep Reinforcement Learning, 2017

### Reinforcement Learning in Operations 
1. Bello I., Pham H., Le Q., Norouzi M., Bengio S. -- Neural Combinatorial Optimization with Reinforcement Learning, 2017

### Foundation of Deep Learning
1. Hochreiter S., Schmidhuber J. -- Long short-term memory, 1997
2. Mikolov T., Chen K., Corrado G., Dean J. -- Efficient Estimation of Word Representations in Vector Space, 2013
3. Le Q., Mikolov T. -- Distributed Representations of Sentences and Documents, 2014 
4. Sutskever I., Vinyals O., Le Q. -- Sequence to Sequence Learning with Neural Networks, 2014
5. Vaswani A., Shazeer N., Parmar N., Uszkoreit J., Jones L., Gomez A., Kaiser L., Polosukhin I. -- Attention Is All You Need, 2017

---
# Books

### Customer Intelligence
1. Winston W. -- Marketing Analytics: Data-Driven Techniques with Microsoft Excel, Wiley, 2014
2. Grigsby M. -- Advanced Customer Analytics: Targeting, Valuing, Segmenting and Loyalty Techniques, Kogan Page, 2016
3. Katsov I. -- [Introduction to Algorithmic Marketing](https://algorithmicweb.wordpress.com/), 2017
4. Falk K. -- Practical Recommender Systems, Manning, 2019

### Price Management
1. Simon H., FassnachtM. -- Price Management: Strategy, Analysis, Decision, Implementation, Springer, 2018
2. Talluri K., van Ryzin G. -- The Theory and Practice of Revenue Management, Springer, 2004
3. Smith T. -- Pricing Strategy: Setting Price Levels, Managing Price Discounts and Establishing Price Structures, Cengage Learning, 2011
4. Phillips R. -- Pricing and Revenue Optimization, Stanford Business Books, 2005

### Supply Chain
1. Fisher M., Raman A. -- The New Science of Retailing: How Analytics are Transforming the Supply Chain and Improving Performance, Harvard Business Review Press, 2010
2. Jacobs R., Berry W., Whybark D., Vollmann T. -- Manufacturing Planning and Control for Supply Chain Management, McGraw-Hill Education, 2018
3. Vandeput N. -- Data Science for Supply Chain Forecast, 2018

### Econometrics
1. Shumway R., Stoffer D. -- Time Series Analysis and Its Applications: With R Examples, Springer, 2017
2. Mills T. -- Applied Time Series Analysis: A Practical Guide to Modeling and Forecasting, Academic Press, 2019

### Data Science and Machine Learning for Enterprise Use Cases
1. Provost F., Fawcett T. -- Data Science for Business, O'Reilly Media, 2013
2. Osinga D. -- Deep Learning Cookbook, O'Reilly Media, 2018
