# Data Analysis Process
In this post, I will be sharing with you the standard of procedures Data Analysts should think as they go about analyzing data. 
Be curious and ask alot of questions to your data!

# Data Pre-collection 
- Develop expectations, set questions
- Collect data 
- Compare expectations with data
- If the expectations and data don't match, revise your expectations or fix (change or wrangle) the data to match expectations

# Data Analysis
1. Stating the question.
  - What questions should I ask? What do I want to find out? What do I need to know?
  - What hypothesis do I want to test? What statistical methods should I use?
  - What kind of forecasting or predictions do I want to make?
2. Exploratory Data Analysis
  - Data cleaning and wrangling. Manipulating data to suit our question requirements.
  - Make exploratory plots of data
  - Refine question or collect more data
3. Model Building
  - Building formal statistical models
  - Primary model answers questions
  - Fit secondary model to include more predictors
4. Interpret Results
  - Interpret totality of analyses with focus on effect sizes & uncertainty
5. Communicate Results

# Example : Asthma prevalence in the U.S.
Let's say you are working for a medical drug production company.  
Your boss wants to understand the prevalence of asthma amongst adults,  
and how big the market for a new asthma drug might be.

You have a general question identified to you by your boss, and your role as a DA will be to...
1. Sharpen the question
- Determine what data to collect to match expectations.
- Otherwise, revise the question.
  - Ex. You could visit Centers for Disease Control (CDC) website to get recent data on asthma prevalence.
  - Why so we need to make a new drug, if there are already existing drugs in the market?
  - "Any new drug that will be developed should perhaps target those whose asthma was not controlled with the current medication."
  - "How many people in U.S. have asthma THAT is not currently controlled by the existing drugs?"
  - "What are the demographic predictors of uncontrolled asthma? Who will be more prone to this?
  - "To whom should we market this drug to?"
  - "How will the changing demographic patterns projected to occur in next 5-10 years will be expected to affect the prevalence of uncontrolled asthma?"
  - "What is the effect of changing demographics on the prevalence of uncontrolled asthma?" --> helps predict size of market in the future
- We have refined the question and identified a better question here.
- Repeat this process of collecting information and determine more questions to reveal informative insights.
3. Exploratory data analysis
- What data files do I need to collect? Do I have to append files?
- Estimate the prevalence of uncontrolled asthma among adults in US.
- What kind of characteristic predictors are common amongst this group?
- (Useful for making future predictions on who will need this drug)
4. Build a statistical model
  - To determine the demographic characteristics that best predict someone will have uncontrolled asthma.
      - Age, gender, race, BMI, smoking status, income, etc.
      - Association, correlation, standard scaler.
      - Changing categorical variables to 0 and 1. (k-1 dummy variables to avoid dummy variable trap)
  - Estimate specific parameters. Make predictions.
  - Challenge your findings and test your assumptions.
5. Interpret the results

6. Communicate the results
- Report building
- Brief summary of findings
- Gather feedback, what kind of extra questions do we need to find out? (Key! Answering new qns never end for a DA!)

# Types of question to ask
## Descriptive questions
A descriptive question is one that seeks to **summarize characteristics associated with the set of data.**
  - Ex. determining the proportion of males, the mean number of servings of fresh fruits and vegetables per day, or the frequency of viral illnesses in a set of data collected from the group.

There is no interpretation of the result itself as the result is a fact, an attribute of the set of data that you are
working with.

## Exploratory questions

**Analyze the data to see if there are patterns, trends, or relationships between variables.** They will help you generate potential hypothesis to test in the next step. (You are not testing a hypothesis here, but coming up with a general assumptions to test.)

  - Ex. From EDA, we found that people who ate healthier diets had fewer viral illnesses so you propose the hypothesis that among adults, eating at least 5 servings a day of fresh fruit and vegetables is associated with fewer viral
illnesses per year. 

## Inferential questions
An inferential question would be a restatement of this proposed hypothesis as a question and would be **answered by
analyzing a different set of data**, which in this example, is a representative sample of adults in the US. By analyzing
this different set of data you are both **determining if the association you observed in your exploratory analysis holds
in a different sample** and whether it holds in a sample that is representative of the adult US population, which would
suggest that the association is applicable to all adults in the US. 

## Predictive questions
A predictive question would be one where you ask what types of people will eat a diet high in fresh fruits and vegetables
during the next year. 

In this type of question you are less interested in what causes someone to eat a certain diet, just **what characteristics predicts whether someone will eat this certain diet.**
  - ex. For example, higher income may be one of the final set of predictors, and you may not know (or even care) why people with higher incomes are more likely to eat a diet high in fresh fruits and vegetables, but what is most important is that **income is a factor that predicts this behavior.**
    
## Causal questions
A causal question asks about whether changing one factor will change another factor, on average, in a population. 
## Mechanistic questions (HOW does A lead to B?)
None of the questions so far will tell us if a diet will significantly indeed cause a reduction in no. of viral illneses. 
- It doesn't tell us **HOW the diet leads to a reduction in the number of viral illnesses**
- A question that asks how a diet high in fresh fruits and vegetables leads to a reduction in the number of viral illnesses would be a mechanistic question.

## Alternative data
Sometimes, the data we want does not exist or is too costly to collect. If this type of data set does not exist,
then the best is to apply causal analysis methods to observational data and answer inferential questions about diet and viral illnesses.





