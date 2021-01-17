# Knowledge-Graph-Embeddings-to-Implement-Explainability
Knowledge Graph Embeddings (KGE) to implement Explainable Artificial Intelligence. As AI develops users must know how algorithms make their decisions, especially for hazardous tasks such as driverless cars. Knowledge graphs are an inherently understandable form of text-based data created as an interconnected network of information. These can be converted into KGE by transforming the unqiue entites in the graph to vector representations. With these, predictions were made for missing/incorrect links in the network and further explainations were made by plotting the clusters of the data. Knowledge graphs and their embedded models were researched and four of these KGE were created and tested by their ability to rank the correct links from a Covid-19 dataset. This dataset was extracted from research papers about the virus to retrieve information quicker. The model which was most accurate was used to implement knowledge graph completion and explainability of the dataset using visual and textual interpretations. A 29,000-word thesis was written to describe the work done through the researching, testing and interpreting of this project.

CORD-19 dataset used to create knowledge graph - https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge

Preprocessing - Data set was stored as entities and their information, decided on six realtions to extract from the data: 
1. affiliated_with – Relationship between Author and their Institution
2. authored_by – Relationship between paper and its Author
3. associated_concept – Relationship between paper and related concepts, weighted with Amazon Medical confidence scores
4. similarity – Similarity between two papers based on the crossover in text
5. cites – Relationship between papers cited in another paper
6. associated_topic – Relationship between Papers and 1 of 10 labelled topics

These links were turned into triple format i.e. (First Entity, Link, 2nd Entity), to make a network of related information (Knowledge Graph). The data was filtered to only allow for entites which are found at least 5 times in the network, otherwise the KGE models would fail to learn effectively. NetworkX was used to create data visualisations of the KG.

Visual Sample of KG links:

![Knowledge Graph](https://github.com/ronanmmurphy/Knowledge-Graph-Embeddings-to-Implement-Explainability/blob/main/Images/KG.PNG?raw=true)

The above graph contains thousands of connected links and entites, heres a zoomed in view to understand the relationships.
Links Related to SARS:

![sample set](https://github.com/ronanmmurphy/Knowledge-Graph-Embeddings-to-Implement-Explainability/blob/main/Images/KG_sample.PNG?raw=true)


A knowledge graph embedded model embeds the entities and relations of a KG into continuous vector spaces. Trained 4 KGE models - TransE, DistMult, ConvE and ComplEx models. These models were trained using pretrained models made by Ampligraph(Open-source Python Library). Many different variants of hyperparameters to find the optimal settings for each model, evaluated for Mean Rank, Mean Reciporical Rank, and Hits @ rate. 
Best performing models: 
TransE - MR: 3609.37; MRR: 0.14; Hits@1: 0.09; Hits@3: 0.15; Hits@10: 0.22
DistMult - MR: 2,477.12; MRR: 0.17; Hits@1: 0.13; Hits@3: 0.19; Hits@10: 0.24
ComplEx - MR: 2104.23; MRR: 0.25; Hits@1: 0.16; Hits@3: 0.20; Hits@10: 0.27
ConvE - Unfortunatlely the dataset was too large for this model and it failed to perform as there was too much computation required on limited resources available(Google Colabs 15GB GPU).

There are many different applications for KGE including link prediction, cluster analysis, and Explainability. All three of these were performed to gain a better interpretation of the Knowledge Graph. Link Prediction is a form of knowledge graph completion to include and remove relations which have strong or weak correlation. This improves the performance of the knowledge graph as the predicted links can be added. Finding duplicates is another form of this to detect entites which are the same but may vary in spelling for example. Finding the optimal number of clusters in the dataset can be estimated using the elbow method, which determined 9 for the dataset. When testing output I decided that 10 fit it better by judging the seperation of clusters. This proved effective in when comparing the identify entities in the plot below.

Here is an example fo K-means clustering analysis:

![cluster analysis](https://github.com/ronanmmurphy/Knowledge-Graph-Embeddings-to-Implement-Explainability/blob/main/Images/clusters.PNG?raw=true)

To provide Explainability to the KGE using dimensionailty reduction algorithms - Principal Component Analysis, TSNE, and UMAP. These were implemented using TensorBoard to visualise the plots. These 3-D graphs provide an alternative perspective about the embeddings and can help to explain predictions.

Here is an example of PCA 3D model showing the relations with 'Lung' entity, which adds interpretation of predicted links.

![PCA analysis](https://github.com/ronanmmurphy/Knowledge-Graph-Embeddings-to-Implement-Explainability/blob/main/Images/pca.PNG?raw=true)
