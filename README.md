# Sentiment Analysis of Skincare Product Reviews

# Introduction
The skincare industry has experienced exponential growth in recent years, driven by a heightened awareness of self-care and the pursuit of youthful, radiant skin. With an abundance of products available in the market, consumers are often faced with the challenge of selecting the right skincare regimen tailored to their needs. In this digital age, online reviews have become a crucial resource for consumers seeking guidance on skincare products. Understanding the sentiment behind these reviews is essential for both consumers and skincare companies. Positive sentiments indicate satisfaction with product efficacy, texture, scent, and overall experience, influencing potential buyers positively. Conversely, negative sentiments may signal issues such as allergic reactions, ineffective formulations, or poor customer service, potentially deterring others from purchasing the product. 

Therefore, the ability to analyze skincare reviews and accurately predict sentiment can provide invaluable insights. For consumers, it facilitates informed decision-making, allowing them to choose products that align with their preferences and avoid potential pitfalls. For skincare companies, sentiment analysis offers a means to gauge customer satisfaction, identify areas for improvement, and refine product offerings to better meet consumer needs.

# Objective
The main objective of this project is to develop a machine learning model that can accurately classify skincare reviews as positive or negative based on the sentiment expressed in the text.


# Code Snippets & Results
- Dataset
The dataset for this project is sourced from Ulta Skincare Reviews. It consists of textual reviews along with corresponding sentiment labels (positive/negative). The dataset is obtained through web scraping or provided by Ulta for research purposes. It is preprocessed to remove irrelevant information, ensure consistency in labeling, and split into training, validation, and test sets.

<img width="573" alt="Screenshot 2024-08-14 at 3 53 35 AM" src="https://github.com/user-attachments/assets/a3b2a507-36fc-4ac5-aff1-5bd62e7a694a">


# Data Preprocessing, Visualization and Exploratory Data Analysis (EDA)
Here are some details about the dataset:

• It has 4150 entries, which refers to the total number of rows in the dataset.
• It has 16 columns. These columns contain features or attributes of the data points.
  
The output shows the data type of each column. There are two integer data types (int64) and eight object data types.

• Integer data types are typically used for numerical data.
• Object data types are used for non-numerical data such as text strings.
  
In this dataset, the two integer data types likely represent the “Review_Upvotes” and “Review_Downvotes” columns, while the eight object data types likely represent the text data in the other columns.
Overall, the information in the screenshot suggests that the dataset likely contains reviews of products, possibly gathered from an online store.

<img width="452" alt="Screenshot 2024-08-14 at 3 56 04 AM" src="https://github.com/user-attachments/assets/926b8f3e-5866-4889-bd4b-9bc5ba88d2f9">

Lowercase conversion:
 This line converts all characters in the ‘Review_Text ’column to lowercase. This is a common data cleaning step because it can help improve the consistency of the text data and make it easier to process.

Remove punctuation:
 This line removes all characters except alphabets and numbers from the text in the ‘Review_Text ’column. This is a common data cleaning step because punctuation marks can add noise to the data and make it more difficult to analyze. The re.sub function is used for regular expression substitution.

Tokenization:
 This line splits the text in the ‘Review_Text ’column into a list of words. This is a common data preprocessing step for NLP tasks. The code likely uses the word_tokenize function from the NLTK library to perform tokenization.

<img width="558" alt="Screenshot 2024-08-14 at 3 57 10 AM" src="https://github.com/user-attachments/assets/3cbefceb-783c-4c14-8cf6-69968b3fb270">


By creating categorical labels for the review lengths, you can more easily analyze how review length is related to other variables in the dataset. For example, you might be interested in whether there is a correlation between review length and sentiment rating.

<img width="808" alt="Screenshot 2024-08-14 at 3 57 42 AM" src="https://github.com/user-attachments/assets/f5c0ff89-e233-4164-9631-69f001e95734">


# Product vs Count

<img width="663" alt="Screenshot 2024-08-14 at 3 58 47 AM" src="https://github.com/user-attachments/assets/805615ea-c537-481d-b478-b5db93cfee26">

The graph shows the number of units of four skincare products in stock at a company. The y-axis shows the number of units, and the x-axis shows the product name. The four products are Multi- Vitamin Thermafoliant, Hydro Masque Exfoliant, Daily Superfoliant, and Daily Microfoliant. The graph indicates that the Daily Microfoliant is the most popular product, with around 1400 units in stock. The other three products all have significantly fewer units in stock, all around 200-400 units.

# Review Upvotes Product-wise

<img width="725" alt="Screenshot 2024-08-14 at 3 59 41 AM" src="https://github.com/user-attachments/assets/fa4a5d9d-167b-4c2e-abc2-1cc4f8eed916">

The line graph shows the number of upvotes received by four different skincare products. The y-axis shows the number of upvotes and the x-axis shows the product name.

  • Daily Microfoliant appears to be the most popular product, with around 1400 upvotes.
  • Multi-Vitamin Thermafoliant, Hydro Masque Exfoliant and Daily Superfoliant all have significantly fewer upvotes,in the range of 200-400 upvotes.
  
It is important to consider that this graph only shows the upvote count and does not provide any context about the total number of reviews each product has received. Upvotes are a way for users to express their appreciation for a review, but they don't necessarily reflect the overall sentiment of the reviews.

# Review Downvotes Product-wise

<img width="861" alt="Screenshot 2024-08-14 at 4 00 44 AM" src="https://github.com/user-attachments/assets/c36ebe7c-957b-4297-adff-96f2213e29a9">

Here’s a breakdown of the graph:

  • Daily Microfoliant has the most downvotes, at around 300.
  • Multi-Vitamin Thermafoliant and Hydro Masque Exfoliant have the least downvotes, both around 50.
  • Daily Superfoliant has around 150 downvotes.
  
It is important to note that similar to upvotes, downvotes only reflect a portion of the user reviews and don't necessarily represent the overall sentiment. A high number of downvotes could indicate that some users find a product ineffective or irritating, but it could also mean there are simply more reviews for that product.

# Count vs Verified Buyer

<img width="482" alt="Screenshot 2024-08-14 at 4 01 51 AM" src="https://github.com/user-attachments/assets/4506c908-d82d-45ef-9314-283cb72c55ca">

The image you sent is a bar graph that shows the distribution of verified buyers compared to count buyers for a particular product. The x-axis shows whether the buyer was verified or not ("No" or "Yes") and the y-axis shows the count.

  • There are significantly more verified buyers than count buyers. This suggests that verification may be an important factor in making a purchase decision for this product.


TextBlob uses a machine learning algorithm to classify the sentiment of text as either positive or negative. It has a built-in feature that is used to classify text as positive or negative based on a set of predefined words and their associated polarities. Polarity is between -1 and 1 with negative words giving rise to negative values and positive words giving positive values.

<img width="698" alt="Screenshot 2024-08-14 at 4 02 56 AM" src="https://github.com/user-attachments/assets/f1a67f66-9d42-4800-a81f-d8373606257c">

# Count of Sentiment

The image you sent is a pie chart visualization of the sentiment distribution in a dataset, likely customer reviews [1]. It shows the proportion of positive and negative sentiment reviews. Here’s a breakdown of the information in the chart:

  • Positive Sentiment: The chart indicates that 94.7% of the reviews fall under the positive sentiment category.
  • Negative Sentiment: The chart indicates that 5.3% of the reviews fall under the negative sentiment category.

Pie charts are useful for visualizing categorical data where the slices represent proportions of the whole. In this case, the whole is all of the reviews in the dataset, and the slices represent the proportion of positive and negative reviews.

# Conclusion

The skincare industry is undergoing rapid expansion, fueled by consumers' increasing interest in skincare products and routines. In this digital age, online reviews play a pivotal role in shaping consumer perceptions and purchasing decisions. Understanding the sentiment expressed in these reviews is crucial for both consumers and skincare companies. 

This project aims to contribute to the skincare industry by developing a sentiment analysis model tailored specifically for skincare reviews sourced from Ulta. Through the application of Natural Language Processing (NLP) techniques, the model can accurately classify reviews as positive or negative, providing valuable insights for consumers and companies alike. 

For consumers, the sentiment analysis model offers a means to make more informed purchasing decisions. By analyzing reviews, consumers can gauge the effectiveness, quality, and overall satisfaction with skincare products before making a purchase. This empowers them to select products that align with their skincare goals and preferences, ultimately enhancing their overall skincare experience. 

For skincare companies, sentiment analysis provides invaluable feedback on product performance and customer satisfaction. By understanding consumer sentiments, companies can identify areas for improvement, refine product formulations, and tailor marketing strategies to better meet consumer needs. This can lead to enhanced brand loyalty, increased sales, and a stronger competitive edge in the market. 

In conclusion, this project aims to leverage the power of NLP to bridge the gap between consumers and skincare companies, facilitating a more transparent and informed skincare landscape. By accurately predicting the sentiment of skincare reviews, this model can contribute to a more seamless skincare shopping experience for consumers while empowering companies to deliver products that resonate with their target audience. Through continuous refinement and innovation, sentiment analysis holds the potential to revolutionize the skincare industry, driving improvements in product quality, customer satisfaction, and market competitiveness.
