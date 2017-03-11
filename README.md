# Machine Learning Engineer Nanodegree
## Heimdall - Payments fraud checks

Jean Jung

February 22st, 2017

## I. Definitions

### Vocabulary

1. **Chargeback**: The reversal of a prior outbound transfer of funds from a consumer's bank account, line of credit, 
or credit card.
1. **IP**: An Internet Protocol address (IP address) is a numerical label assigned to each device (e.g., computer, 
printer) participating in a computer network that uses the Internet Protocol for communication.
1. **API**: The acronym for Application Programming Interface, it is the schema or contract a computer application have 
to follow to communicate with another application.
1. **Impersonation**: The act of acting like other person for getting access to individual data, archives or resources.
1. **Cash payment solutions**: Payment methods where the person buying online receives a voucher or a ticket and have to 
pay it with money.

### Project Overview

In norse mythology, Heimdall or Heimdallr is the gatekeeper of Bitfröst, the bridge that connects Midgard (mans' earth)
to Asgard (Gods' realm). In other words he controls unwanted access to Gods' realm.

For payment processing services confirming that a transaction is legitimate is vital to avoid fraud attacks that 
generates chargebacks[¹](https://github.com/jejung/heimdall#vocabulary).

The legitimacy of a transaction can be checked or monitored by analyzing a series of factors and patterns like 
customer IP Address[²](https://github.com/jejung/heimdall#vocabulary), customer buying frequency, hour of day, 
items category, customer age, customer documents, quantity of confirmed chargebacks and others. 

Heimdall intends to be the one who decides which transaction can be accepted or not before forwarding it to the bank 
or financial institution who will process the transaction. 

In this project a mini API[³](https://github.com/jejung/heimdall#vocabulary) capable of analyze transactions and classify them on three levels of risk was created. 

Risk levels:

1. **LOW**: Low risk transaction, can be forwarded to the bank.
1. **MEDIUM**: Medium risk level transactions can need a human intervention, making a review of the data and even 
contacting the customer to validate the legitimacy of the transaction depending on the amount.
1. **HIGH**: High risk transactions offers a real chargeback risk and should not be processed.
  

### Problem Statement

While processing online payments, it's very common to suffer fraud attacks, from people trying to validate credit 
card number stolen from anyone to elaborated attack plans that involve impersonation
[<sup>4</sup>](https://github.com/jejung/heimdall#vocabulary). There are a lot of tools to identify online fraud attacks and 
avoid chargebacks, the great majority of them focused on credit card attacks.

Even that great majority of attacks involve credit card transactions, bank transactions or cash payment 
solutions[<sup>5</sup>](https://github.com/jejung/heimdall#vocabulary) are also victims of this type of cyber crime.
Above there is a list of known fraud attacks types 
[according to HSBC Bank](http://www.hsbc.com/online-banking/online-security/types-of-online-attack):
 
* **Fraudulent supplier requests**: Attackers can call or email a employee asking for a payment on an order that 
does not exists, giving her bank accounts for deposit for example. This type of attack targets companies and use 
unattended people as a vulnerability to exploit.
* **Courier scams**: This type of attack focus on common people. An attacker can call or email the victim asking to 
him/her to return your credit card to bank in order to get a new one. Attackers can even dress themselves as courier
employees and go to victim's house taking the card from his hand. They also can ask you to cut your card in the half 
just to win credibility.
* **“Vishing”**: Victims of this type of attack will receive a call from the attacker that will talk as being from
bank staff or police investigation and try to obtain private information such as passwords, security code or even 
indirect information like full address, family, full name, documents and etc.
* **Keystroke capturing/logging**: This is a cyber attack. Victims of this type of attack will have an unwanted
software on their computer or another device like smartphone or tablet that will keep track of anything that is 
inputted like bank accounts and passwords for example.
* **Pharming**: Pharming occurs when a fraudster creates false websites in the hope that victims will visit them by 
mistake. People can sometimes do this by mistyping a website address or sometimes a fraudster can redirect traffic 
from a genuine website to their own. The 'pharmer' will then try to obtain your personal details when you enter them 
into the false website.

This list shows us that in the majority, fraudsters are looking for people data for posterior use. Since this is 
true, fraud detection tools should look for patterns and common data on every transaction, trying to find requests 
that does not follow the pattern for each customer and learn from identified attacks to predict new issues in the 
future. 

Heimdall will act inspecting every transaction sent to him by API and answering one of the three possible levels 
with some additional data that explains the response.
 
- API's clients will have to provide enough information to:
    1. Identify customer;
    1. Identify customer's supposed device;
    1. Identify product category;
    1. Identify payment method;
    1. Identify billing information;
    1. Identify transaction in future communication;
- Every divergence between Heimdall response and transaction final status needs to be notified via API call.
- Chargebacks needs to be notified via API call.
    
#### Transaction flow

![heimdallflow](https://cloud.githubusercontent.com/assets/13054871/23780514/3044b184-0526-11e7-8fa8-29005af09129.jpg)

### Metrics
In this section, you will need to clearly define the metrics or calculations you will use to measure performance of a model or result in your project. These calculations and metrics should be justified based on the characteristics of the problem and problem domain. Questions to ask yourself when writing this section:
- _Are the metrics you’ve chosen to measure the performance of your models clearly discussed and defined?_
- _Have you provided reasonable justification for the metrics chosen based on the problem and solution?_

## II. Analysis
_(approx. 2-4 pages)_

### Data Exploration
In this section, you will be expected to analyze the data you are using for the problem. This data can either be in the form of a dataset (or datasets), input data (or input files), or even an environment. The type of data should be thoroughly described and, if possible, have basic statistics and information presented (such as discussion of input features or defining characteristics about the input or environment). Any abnormalities or interesting qualities about the data that may need to be addressed have been identified (such as features that need to be transformed or the possibility of outliers). Questions to ask yourself when writing this section:
- _If a dataset is present for this problem, have you thoroughly discussed certain features about the dataset? Has a data sample been provided to the reader?_
- _If a dataset is present for this problem, are statistics about the dataset calculated and reported? Have any relevant results from this calculation been discussed?_
- _If a dataset is **not** present for this problem, has discussion been made about the input space or input data for your problem?_
- _Are there any abnormalities or characteristics about the input space or dataset that need to be addressed? (categorical variables, missing values, outliers, etc.)_

### Exploratory Visualization
In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Algorithms and Techniques
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_

### Benchmark
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_


## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

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
