# Machine Learning Engineer Nanodegree
## Heimdall - Email classifier tool

Jean Jung

February 22st, 2017

## I. Definitions

### Project Overview

In norse mythology, Heimdall or Heimdallr is the gatekeeper of Bitfröst, the bridge that connects Midgard (mans' earth)
to Asgard (Gods' realm). In other words he controls unwanted access to Gods' realm.

This project will try to do both, control unwanted emails received on a mail box and classify wanted emails to be
forwarded to someone who is capable of reading, understanding and resolving it as fast as possible.

### Problem Statement

All day in help desk systems there are a lot of emails coming from different sources stating about different problems.
Each of these require specific abilities and know-how of some part of the organization's process. A common way to avoid
every employee needing to know about all the organization's processes is to have different groups to check
different issues. With this layout there is a triage to check for what group an issue should be sent. This is
represented on the image bellow:

![Help desk](http://www.opensourcehelpdeskguide.com/images/help-desk.jpg)

It takes time to classify every incoming request, the operator needs to open the request, identify the requester, read
its content, see if it is a valid request, understand what the requester needs, identify the staff group it belongs to
finally classify and forward the request, and this is done to every request issued.

It would be a great time gain if this could be done automatically as son as any request arrives.

To solve this problem, Heimdall will integrate with help desk systems available on market to get access to email 
contents. It will analyze the already classified emails training a Supervised Learning Classification model. When ready,
it will start to inspect every incoming email and classify them using any type of tagging system available from the
chosen platform or forwarding the email to a previously defined list for each label.

### Metrics

#### Accuracy

Heimdall will use Accuracy metric to measure performance of models being used/tested to solve the problem.

Classification accuracy is the number of correct predictions made as a ratio of all predictions made. This is the most 
common method used to measure performance on classification models.

The formula can be expressed as:

    X = t / n * 100
    
Where `X` is the accuracy, `t` is the number of correctly classified samples and `n` is the total number of samples, 
thus the accuracy will be a number between 0 and 1 representing the percentage of right predictions made.

We will use this to estimate how many emails would be correctly classified per day.

#### Confusion matrix

During development a confusion matrix will be used to identify the exactly point we are failing and help us improve our
classification model.

A Confusion matrix consists of a table in form `N * N` where `N` is the number of possible values or classes. On every
cell you have the count of a `predicted vs actual` conflict happened with the respective values represented by row and
column positions.

Example:

        A   B   C
    A   10  2   3
    B   9   20  0   
    C   1   5   7

With the given matrix we can see that `A` was predicted correctly 10 times and there was 2 times it was predicted when
the actual value was `B`.

## II. Analysis

### Data Exploration

One of the best text example datasets available on internet currently is the 
[20 newsgroups dataset](http://scikit-learn.org/stable/datasets/twenty_newsgroups.html#newsgroups), freely distributed 
on scikit-learn python's library.

This dataset contains around 180000 newsgroups posts on 20 subtopics. The dataset itself is already split on train and 
test parts to be used on leaning models.

Even this dataset is not composed by emails, it can be used to train and measure a email classifier tool, since Heimdall
is supposed to use only general text classifying tools. 

Scikit-learn has functions to fetch and load the data into python arrays containing raw text and labels related. We 
start by using the `sklearn.datasets.fetch_20newsgroups` function to retrieve the data this will return a python object
with two attributes:
 
* **data**: A list containing raw texts from news posts.
* **target**: A list where each value is the label of the correspondent entry on the data list.

For Heimdall will be used only between 4 and 6 categories because this would be the average number of departments a
company will have so we don't need a model that is capable of identifying 20 categories.

According to the documentation, the categories are:

    'alt.atheism',
    'comp.graphics',
    'comp.os.ms-windows.misc',
    'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware',
    'comp.windows.x',
     'misc.forsale',
     'rec.autos',
     'rec.motorcycles',
     'rec.sport.baseball',
     'rec.sport.hockey',
     'sci.crypt',
     'sci.electronics',
     'sci.med',
     'sci.space',
     'soc.religion.christian',
     'talk.politics.guns',
     'talk.politics.mideast',
     'talk.politics.misc',
     'talk.religion.misc'
     
Exploring the data, we can see that the distribution of the categories are like:
 
![feature_dist](https://cloud.githubusercontent.com/assets/13054871/24388181/56fa9172-134f-11e7-9a01-b75a204d6a55.png)

We can see that data is well distributed, just some of the categories like `alt.atheism` and `talk.religion.misc` have 
less examples, other ones are balanced. We will choose the `sci` categories group to validate our model, since the 
distribution is well balanced on that classes and the contents will be very similar, the same as in a help desk system.

Filtering data to use the chosen ones we end up with:
 
    Size of train data: 2373 samples
    Size of test data: 1579 samples
    

### Exploratory Visualization

Looking on below image we can see the most repeatable words for 6 arbitrary documents:
 
![word_count](https://cloud.githubusercontent.com/assets/13054871/24683599/69805bc4-1976-11e7-9ee6-fb733b10db7f.png)

We've found that prepositions and articles like 'the', 'and', 'is', 'in' and etc. are too much bigger than other words 
in the great majority of the documents. With the exception of Topic 2, all other topics seems to be natural text so this
will happen with almost any email delivered on the internet today.

It can turn things difficult to classify since this words will appear on every document. To avoid this problem we 
will analyze a TF-IDF (Term Frequency–Inverse Document Frequency) to find the words that appear only in some documents.
We will explain the TF-IDF functionality later on this paper.

![tfidf](https://cloud.githubusercontent.com/assets/13054871/24684492/c4583d5e-197c-11e7-80a6-35982ec305e3.png)

We can see now that with TF-IDF all words for each topic have changed turning into more specific words, what will
make possible to identify a document by presence of some of these words.

### Algorithms and Techniques

In order to achieve his objective Heimdall will use the Bag of Words model, commonly used on natural language processing. One of 
the great advantages of using this algorithm is the simplicity and extensibility.
 
 #### Feature extraction algorithms

A raw text itself is not a good input source for a classifier algorithm. Classification models expect it's input data to
be something with measurable and comparable values, also called features. What do you can extract from "Hi, 
I have a problem", for us humans it's simple to say that it seems there is someone with problems, but what a computer
algorithm can extract from this data? The size? Maybe but how do you classify texts by size? Large, short? That is not
what we want, we need more.
 
This is what feature extraction extends for. This algorithms transform raw text documents into a matrix with relevant
features like the frequency a specific word appears for example.  

There are a lot of variations and with scikit-learn it's simple to do a benchmark through them, so this is what we will 
do. Bellow you can see a brief explanation of each technique we will measure. Later on 
[Data preprocessing](https://github.com/jejung/heimdall#data-preprocessing) section you will find more details of how we
have applied this techniques on our source dataset.

##### Count

The Count score is the most simple, it is just measures the frequency each word appears on the text.

##### TF-IDF

The Term Frequency - Inverse Document Frequency is more complex, it works giving each word the score as a relation 
between the term frequency over all documents and the inverse frequency of documents it appears.

##### Hashing

Almost the same as the Count approach, but use a token for indexing instead the word itself what save computational
resources.

All these three implementations result in a matrix of terms and his scores. This matrix can be forwarded to any 
classification model as all features will be numeric.

##### PCA

Principal Component Analysis is an algorithm used to discover which set of features imply on a higher data variance, 
ranking this way the best features that can be used on a classifier model. Heimdall will choose just N best features to 
save computational resources on query operations.

#### Classifier algorithm 

##### Multinomial Naive Bayes

This algorithm is a implementation of the Naive Bayes algorithm for multinomially distributed data. This algorithm is 
known to work well with text documents. Even this is designed for receiving a vectorized word-count data it also works
well with TF-IDF matrices.

It can performs well since it take advantage of probabilities, since the probability of certain words appear on a 
specific email request type is great, but it needs a lot of training samples.

##### KNN

The K-Nearest Neighbour classifier works by finding elements that approximate their values, in our scenario it can make
sense. We can think if we have the same words appearing in two different texts with almost the same frequency so they
should be talking of the same thing!

This algorithm has the advantage of being very fast on training but a bit slow on querying.

##### SVM

Support Vector Machines is an algorithm that classifies data separating their data points with vectors in some 
dimensions and calculating their distance to that vector. In order to discover all the different categories, the maximum
gap between data points will be found, so a point can be classified based on which side of the gap they fall.

This algorithm reduce the need for labeled training instances, with     

### Benchmark

Let's do a first checkout of how everything we mentioned applies on our simulated "real world". 
 
With a simple script we have executed the combinations of the algorithms mentioned earlier and created a graphic showing
the results:

![Benchmark](https://cloud.githubusercontent.com/assets/13054871/24840306/4032cb3e-1d41-11e7-9cb1-d6c664b3ec9b.png)

As predicted the KNN algorithm has the slowest testing time, the surprise is that it has
the worst score. The best combination for a KNN model is the TF-IDF vectorizer, it has a good score but a very high test
time.

The SVC algorithm has an impressive score, the greatest for Hashing and TF-IDF vectorizers, just staying behind Naive 
Bayes with the Count value, but in counter part has the slowest training time.

The Multinomial Naive Bayes algorithm performs very well. It has great scores and small training and test times.

This simple benchmark shows us that in combination with any model we are using the Hashing Vectorizer is not so good, so 
we will discard it from here through the final.

## III. Methodology

### Data Preprocessing

As discussed before on Algorithms and Techniques section, for raw text classification we always need some preprocessing.

The first thing to notice is the time that preprocessing takes, about 4 to 5 seconds on our benchmark. To improve the 
preprocessing time we've tried to reduce the number of features generated by all vectorizers to only 150 but that 
dropped the performance of all models to below the 50% mark. The time taken was only a little greater than 1 second, so 
we con continue from here.

To improve our processed data we've tried to use a PCA algorithm so we could choose the best features in a proper time. 
The idea behind that was to let the vectorizers generate more features and let a PCA decide what features to use, but
that was so slow we could not even measure.

We have found that when working with sparse matrices the PCA algorithm is very slow because for every feature it needs
to center all data points around. As the matrices returned by a raw text vectorized are known to be very large we have 
gave up. We have tried also the NMF - Non-negative Matrix Factorization, an algorithm that can be used for 
dimensionality reduction, also slow and the Truncated Singular Value - TruncatedSVD on scikit-learn - that performs a 
linear dimensionality reduction using the truncated singular value decomposition, contrary to PCA there is no need to 
center data on this algorithm, even that this algorithm was faster, he can return negative values, what is invalid for 
working with Naive Bayes based algorithms and there was no significant gain working with other classifiers, so we have 
discarded this algorithm too.

Thinking better of our task here, there is no need to a second algorithm of dimensionality reduction, since the feature
extraction algorithms we are using can do that!

For example, CountVectorizer has a parameter called `max_features` that causes the result matrix to have only a maximum
of `n` features. The features chosen will be the `n` top words that appear with more frequency, exactly what we want,
just the best. For any other vectorizer we are using, the parameter is the same.

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
