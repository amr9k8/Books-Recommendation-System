
import numpy as np
import pandas as pd
import pickle
import pandas as pd
from collections import OrderedDict
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#nltk.download('stopwords')
#nltk.download('punkt')
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask,render_template,request


books = pd.read_csv('books.csv',low_memory=False)
users = pd.read_csv('users.csv')
ratings = pd.read_csv('ratings.csv')


def generate_pivot_table():
    
    # create new dataframe by merging rating and books on "ISBN" coulmn
    ratings_with_name = ratings.merge(books,on='ISBN')
    # Filter out users who have rated more than 300 books
    x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 300
    nerd_users = x.loc[x == True].index
    filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(nerd_users)]
    
    # Filter books that have been rated more than 50 times
    y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50
    famous_books = y.loc[y == True].index
    final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]
    
    # Create a pivot table with Book-Title as index and User-ID as columns
    pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
    
    # Fill any missing values with zeros
    pt.fillna(0, inplace=True)
    
    # Calculate the cosine similarity between each pair of books in the pivot table
    similarity_scores = cosine_similarity(pt)
    
    return pt, similarity_scores

def popular_books_recommendation():
    
    
    ## Popularity Based Recommender System
    # create new dataframe by merging rating and books on "ISBN" coulmn
    ratings_with_name = ratings.merge(books,on='ISBN')
    # Group by title instead of ISBN because different versions or editions of a book may have different ISBNs, but they would still have the same title
    # Then we make count() to count "rating" for each bookTitle 
    # Then we rename bookrate to number_of_rates
    num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
    num_rating_df.rename(columns={'Book-Rating':'num_ratings'}, inplace=True)
    
    # Calculate the average score for each title
    # Then we rename Book-Rating to avg_rating
    avg_rating_df = ratings_with_name.groupby('Book-Title')['Book-Rating'].mean().reset_index()
    avg_rating_df.rename(columns={'Book-Rating':'avg_rating'}, inplace=True)
    
    # Merge rating_counts and rating_avg in one dataframe called popular_df
    popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
    
    # Filter only books that have more than 100 ratings and an average rating of 4.8 or higher
    popular_df = popular_df[(popular_df['num_ratings'] >= 100) & (popular_df['avg_rating'] >= 4.8)].sort_values('avg_rating', ascending=False)
    
    # Add average rating, rating counts, author, and image URL to the books dataframe
    # Drop any duplicate rows that have the same book title
    # Filter only the columns we want to show: title, author, image URL, number of ratings, and average rating
    popular_df = popular_df.merge(books, on='Book-Title').drop_duplicates('Book-Title')[['Book-Title', 'Book-Author', 'Image-URL-M', 'num_ratings', 'avg_rating']]
    
    return popular_df




app = Flask(__name__)

# Load the pickled objects and store them as global variables in app.config
app.config['POPULAR_DF'] = pickle.load(open('popular.pkl', 'rb'))
app.config['PT'] = pickle.load(open('pt.pkl', 'rb'))
app.config['BOOKS'] = pickle.load(open('books.pkl', 'rb'))
app.config['SIMILARITY_SCORES'] = pickle.load(open('similarity_scores.pkl', 'rb'))
app.config['BOOKS_WITH_DESC'] = pickle.load(open('common_books_with_desc.pkl', 'rb'))



@app.route('/')
def index():
    popular_df_pk = app.config['POPULAR_DF']
    pt_pk = app.config['PT']
    books_pk = app.config['BOOKS']
    similarity_scores_pk = app.config['SIMILARITY_SCORES']

    return render_template('index.html',
                           book_name = list(popular_df_pk['Book-Title'].values),
                           author=list(popular_df_pk['Book-Author'].values),
                           image=list(popular_df_pk['Image-URL-M'].values),
                           votes=list(popular_df_pk['num_ratings'].values),
                           rating=list(popular_df_pk['avg_rating'].values)
                           )

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_books',methods=['post'])
def recommend():

### Collaborative Filtering Based   

    popular_df_pkl = app.config['POPULAR_DF']
    pt_pkl = app.config['PT']
    books_pkl = app.config['BOOKS']
    similarity_scores_pkl = app.config['SIMILARITY_SCORES']

    user_input = request.form.get('user_input') #get book name
    if user_input not in pt_pkl.index:
        return 'Invalid book name'
    # find book index [which row] 
    index = np.where(pt_pkl.index==user_input)[0][0]
    # return 1d array contain similiraty values with all other books
    scores_for_given_book = similarity_scores_pkl[index]
    # Create a list of tuples, where each tuple contains an index and a similarity score for the item at that index
    item_scores = list(enumerate(scores_for_given_book))
    # Sort the list of tuples in descending order based on the similarity scores
    #x:x[1] => for each tuple (index,simllarity_value)  use 2nd value as key to sort 
    sorted_scores = sorted(item_scores, key=lambda x: x[1], reverse=True)
    # Select the 15 items with the highest similarity scores (excluding the item itself)
    top_scores = sorted_scores[1:200] # skip first item as book with it's self is always 1
    #print(top_scores)


    #save all books title from collabrotive based filtering
    collabrotive_based_title = []
    for one_score in top_scores:
        book_index = one_score[0] #(book_index,similiarity_value)
        book_name  = pt_pkl.index[book_index] # get bookname
        collabrotive_based_title.append(book_name)

### Content Filtering Based   
    books_with_desc = app.config['BOOKS_WITH_DESC']
    # Remove keys with values less than 10 characters
    books_with_desc = {k: v for k, v in books_with_desc.items() if len(v) >= 10}
    desc_list = list(books_with_desc.values())
    title_list= list(books_with_desc.keys())
    # Preprocess the book descriptions by removing stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    preprocessed_descriptions = []
    for description in desc_list:
        tokens = word_tokenize(description.lower())
        filtered_tokens = [token for token in tokens if token not in stop_words and token.isalnum()]
        preprocessed_descriptions.append(" ".join(filtered_tokens))
    #TfidfVectorizer object to convert descriptions to  matrix 
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_descriptions)
    # Calculate the cosine similarity between each pair of book descriptions
    similarity_matrix_content_based = cosine_similarity(tfidf_matrix)    
    #get index of book
    book_index = np.where(np.array(title_list) == user_input)[0][0]
    # return 1d array contain similiraty values with all other books
    scores_for_given_book = similarity_matrix_content_based[book_index] #GET SCORES ROW OF BOOK[I] WITH OTHER BOOKS
    # Create a list of tuples, where each tuple contains an index and a similarity score for the item at that index
    similarity_scores = list(enumerate(scores_for_given_book))
    # Sort the list of tuples in descending order based on the similarity scores
    #x:x[1] => for each tuple (index,simllarity_value)  use 2nd value as key to sort 
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:200]
    #print(similarity_scores)
    bookindices = [i[0] for i in similarity_scores]

    #save all books title from content based filtering
    content_based_titles =[]
    for i in similarity_scores:
        book_index = i[0] #(book_index,similiarity_value)
        book_name  = title_list[book_index] # get bookname
        content_based_titles.append(book_name)


    all_books=content_based_titles+collabrotive_based_title
    common = list(OrderedDict.fromkeys([x for x in all_books if all_books.count(x) > 1]))# get common books in orderd way
    final_book_result = []
    final_book_result.extend(collabrotive_based_title[:15])
    final_book_result.extend(content_based_titles[:10])
    final_book_result.extend(common[:10])
    final_book_result = list(set(final_book_result)) # remove duplicates
    data =[]
    # get each book data from books.csv
    for one_title in final_book_result:
        item=[]
        temp_df = books[ books['Book-Title'] == one_title ] #get the book record from book.csv 
        temp_df  = temp_df.drop_duplicates('Book-Title') #drop duplicated rows
        item.extend(temp_df['Book-Title'].values)
        item.extend(temp_df['Book-Author'].values)
        item.extend(temp_df['Image-URL-M'].values)
        data.append(item)
    #app.logger.debug(data)
    #app.logger.debug(final_book_result) 
    return render_template('recommend.html',data=data)
 



if __name__ == '__main__':

    #popular_df = popular_books_recommendation()
    #pt, similarity_scores = generate_pivot_table()

    #books.drop_duplicates('Book-Title')
    ## write objects to binary file so we can store it and don't need to reprocess every time
    #pickle.dump(books,open('books.pkl','wb'))
    #pickle.dump(popular_df,open('popular.pkl','wb'))
    #pickle.dump(pt,open('pt.pkl','wb'))
    #pickle.dump(similarity_scores,open('similarity_scores.pkl','wb'))

    ##start flask app
    app.run(debug=True)
    



