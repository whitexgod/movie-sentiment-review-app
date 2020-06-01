from flask import Flask, render_template, request, url_for
import pickle
import pandas as pd
from fuzzywuzzy import process
import requests
import bs4
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#https://www.kaggle.com/stefanoleone992/imdb-extensive-dataset -->link of df1 dataset from kaggle

clf=pickle.load(open('model.pkl','rb'))
df1=pickle.load(open('df_new.pkl','rb'))

app=Flask(__name__)

movie_names=df1['original_title'].tolist()


@app.route('/predict', methods=['POST','GET'])
def predict():
    movie_title = request.form.get('movie-title')
    #movie_id=request.form.get('movie-id')
    try:

        sub = movie_title.upper()
        df1["Indexes"] = df1["original_title"].str.find(sub)
        index = df1[df1["Indexes"] == 0].index[0]
        movie_id = df1['imdb_title_id'].iloc[index]

        movie_title = []
        movie_year = []
        movie_genre = []
        movie_duration = []
        movie_imdb_rating = []
        def info_scraper(id):
            title_url = 'https://www.imdb.com/title/'
            res = requests.get(url=title_url + id).text
            soup = bs4.BeautifulSoup(res, 'html.parser')
            movie_imdb_rating.append(float(soup.find(name='span', attrs={"itemprop": "ratingValue"}).text))
            x = soup.find(name='h1', attrs={'class': ''}).text
            movie_title.append(x.split("\xa0")[0] + ' ' + x.split("\xa0")[1].strip())
            info = soup.find(name='div', class_='subtext').text
            info2 = soup.find(name='div', class_='subtext')
            movie_duration.append(info2.find(name='time').text.strip())
            movie_genre.append(info.split("|\n")[1].split("\n")[0] + info.split("|\n")[1].split("\n")[1])
            movie_year.append(info.split("\n")[-2].strip())

        info_scraper(movie_id)

        movie_title = movie_title[0]
        movie_year = movie_year[0]
        movie_genre = movie_genre[0]
        movie_duration = movie_duration[0]
        movie_imdb_rating = movie_imdb_rating[0]

        src = requests.get("https://www.imdb.com/title/{}/reviews".format(movie_id))
        soup = bs4.BeautifulSoup(src.content, 'html.parser')

        poster_container = soup.find('div', class_='subpage_title_block')
        poster = poster_container.find('a')
        src_tag = poster.find('img')
        pic = str(src_tag.attrs['src'])

        containers = soup.find_all('div', class_='review-container')
        ratings = []
        names = []
        dates = []
        reviews = []
        for i in containers:
            try:
                ratings.append(i.find('span').text.strip())
                names.append(i.find('a').text.strip())
                dates.append(i.find('span', class_='review-date').text.strip())
                review_div = i.find('div', class_='content')
                reviews.append(review_div.find('div').text.strip())
            except:
                break

        d = {'viewer_ratings': ratings, 'review_title': names, 'review_date': dates, 'reviews': reviews}
        df = pd.DataFrame(d)

        df['viewer_ratings'] = df['viewer_ratings'].str.replace(r'[^\d.]+', '')
        x = df[df['viewer_ratings'] == ''].index.tolist()
        df.drop(x, inplace=True)
        df=df.reset_index(drop=True)
        df['viewer_ratings'] = df['viewer_ratings'].astype(int)
        df['viewer_ratings'] = (round(df['viewer_ratings'] / 100))
        df_copy=df.copy()

        size = df.shape[0]
        rating_generated=[]
        if size>0:
            # Function to clean html tags
            def clean_html(text):
                clean = re.compile('<.*?>')
                return re.sub(clean, '', text)
            df_copy['reviews'] = df_copy['reviews'].apply(clean_html)
            # converting everything to lower
            def convert_lower(text):
                return text.lower()
            df_copy['reviews'] = df_copy['reviews'].apply(convert_lower)
            # function to remove special characters
            def remove_special(text):
                x = ''
                for i in text:
                    if i.isalnum():
                        x = x + i
                    else:
                        x = x + ' '
                return x
            df_copy['reviews'] = df_copy['reviews'].apply(remove_special)
            def remove_stopwords(text):
                x = []
                for i in text.split():
                    if i not in stopwords.words('english'):
                        x.append(i)
                y = x[:]
                x.clear()
                return y
            df_copy['reviews'] = df_copy['reviews'].apply(remove_stopwords)
            ps = PorterStemmer()
            y = []
            def stem_words(text):
                for i in text:
                    y.append(ps.stem(i))
                z = y[:]
                y.clear()
                return z
            df_copy['reviews'] = df_copy['reviews'].apply(stem_words)
            # Join back
            def join_back(list_input):
                return " ".join(list_input)
            df_copy['reviews'] = df_copy['reviews'].apply(join_back)



            from sklearn.feature_extraction.text import CountVectorizer
            cv = CountVectorizer(max_features=1000)
            x = cv.fit_transform(df_copy['reviews']).toarray()

            predict = clf.predict(x)
            predict = predict.tolist()
            df['prediction'] = predict

            rating_generated.append(round((df[df['prediction'] == 1].shape[0] / df.shape[0]) * 10,1))

        else :
            rating_generated.append(0)


        return render_template("predict.html", movie_title=movie_title, movie_id=movie_id, movie_year=movie_year,
                               movie_genre=movie_genre, movie_duration=movie_duration, movie_imdb_rating=movie_imdb_rating,
                               pic=pic, df=df, size=size, rating_generated=rating_generated[0])

    except:

        sub = movie_title.upper()
        def get_matches(query, choices, limit=10):
            results = process.extract(query, choices, limit=limit)
            return results
        def find_id(text):
            sub = text.upper()
            df1["Indexes"] = df1["original_title"].str.find(sub)
            index = df1[df1["Indexes"] == 0].index[0]
            return df1['imdb_title_id'].iloc[index]
        def find_year(text):
            sub = text.upper()
            df1["Indexes"] = df1["original_title"].str.find(sub)
            index = df1[df1["Indexes"] == 0].index[0]
            return df1['year'].iloc[index]

        movie_list = get_matches(sub, movie_names)
        new_list = []
        for i in range(10):
            if movie_list[i][1] > 90:
                new_list.append(movie_list[i])
        movie_nam = []
        movie_y = []
        movie_i = []
        for i in range(len(new_list)):
            movie_nam.append(str(new_list[i][0]))
            movie_y.append(find_year(new_list[i][0]))
            movie_i.append(find_id(new_list[i][0]))
        d = {'movie': movie_nam, 'movie_id': movie_i, 'movie_year': movie_y}
        data = pd.DataFrame(d)
        data.drop_duplicates(inplace=True)

        data_len=data.shape[0]

        return render_template("redirect.html", data=data, data_len=data_len)


@app.route('/imdb', methods=['POST'])
def imdb():
    movie_name = request.form.get('movie-title')
    src = requests.get("https://www.imdb.com/find?q={}".format(movie_name))
    soup = bs4.BeautifulSoup(src.text, "html.parser")
    div = soup.find('div', class_='findSection')
    table = div.find('table')
    names = []
    url = []
    for cell in table.find_all('td', class_='result_text'):
        a_tag = ""
        name = cell.get_text(strip=True)
        names.append(name)
        a_tag = cell.find('a')
        url.append(str(a_tag.attrs['href'].strip("/").strip('title').strip('/')))
    d = {'movie_names': names, 'movie_id': url}
    df = pd.DataFrame(d)
    size=df.shape[0]
    return render_template('imdb.html', movie_name=movie_name, df=df, size=size)

@app.route('/prediction', methods=['POST','GET'])
def prediction():
    #movie_title = request.form.get('movie-title')
    movie_id = request.form.get('movie-id')
    try:

        movie_title = []
        movie_year = []
        movie_genre = []
        movie_duration = []
        movie_imdb_rating = []
        def info_scraper(id):
            title_url = 'https://www.imdb.com/title/'
            res = requests.get(url=title_url + id).text
            soup = bs4.BeautifulSoup(res, 'html.parser')
            movie_imdb_rating.append(float(soup.find(name='span', attrs={"itemprop": "ratingValue"}).text))
            x = soup.find(name='h1', attrs={'class': ''}).text
            movie_title.append(x.split("\xa0")[0] + ' ' + x.split("\xa0")[1].strip())
            info = soup.find(name='div', class_='subtext').text
            info2 = soup.find(name='div', class_='subtext')
            movie_duration.append(info2.find(name='time').text.strip())
            movie_genre.append(info.split("|\n")[1].split("\n")[0] + info.split("|\n")[1].split("\n")[1])
            movie_year.append(info.split("\n")[-2].strip())

        info_scraper(movie_id)
        movie_title = movie_title[0]
        movie_year = movie_year[0]
        movie_genre = movie_genre[0]
        movie_duration = movie_duration[0]
        movie_imdb_rating = movie_imdb_rating[0]

        src = requests.get("https://www.imdb.com/title/{}/reviews".format(movie_id))
        soup = bs4.BeautifulSoup(src.content, 'html.parser')

        poster_container = soup.find('div', class_='subpage_title_block')
        poster = poster_container.find('a')
        src_tag = poster.find('img')
        pic = str(src_tag.attrs['src'])

        containers = soup.find_all('div', class_='review-container')
        ratings = []
        names = []
        dates = []
        reviews = []
        for i in containers:
            try:
                ratings.append(i.find('span').text.strip())
                names.append(i.find('a').text.strip())
                dates.append(i.find('span', class_='review-date').text.strip())
                review_div = i.find('div', class_='content')
                reviews.append(review_div.find('div').text.strip())
            except:
                break

        d = {'viewer_ratings': ratings, 'review_title': names, 'review_date': dates, 'reviews': reviews}
        df = pd.DataFrame(d)

        df['viewer_ratings'] = df['viewer_ratings'].str.replace(r'[^\d.]+', '')
        x = df[df['viewer_ratings'] == ''].index.tolist()
        df.drop(x, inplace=True)
        df = df.reset_index(drop=True)
        df['viewer_ratings'] = df['viewer_ratings'].astype(int)
        df['viewer_ratings'] = (round(df['viewer_ratings'] / 100))
        df_copy = df.copy()

        size = df.shape[0]
        rating_generated = []
        if size > 0:
            # Function to clean html tags
            def clean_html(text):
                clean = re.compile('<.*?>')
                return re.sub(clean, '', text)

            df_copy['reviews'] = df_copy['reviews'].apply(clean_html)

            # converting everything to lower
            def convert_lower(text):
                return text.lower()

            df_copy['reviews'] = df_copy['reviews'].apply(convert_lower)

            # function to remove special characters
            def remove_special(text):
                x = ''
                for i in text:
                    if i.isalnum():
                        x = x + i
                    else:
                        x = x + ' '
                return x

            df_copy['reviews'] = df_copy['reviews'].apply(remove_special)

            def remove_stopwords(text):
                x = []
                for i in text.split():
                    if i not in stopwords.words('english'):
                        x.append(i)
                y = x[:]
                x.clear()
                return y

            df_copy['reviews'] = df_copy['reviews'].apply(remove_stopwords)
            ps = PorterStemmer()
            y = []

            def stem_words(text):
                for i in text:
                    y.append(ps.stem(i))
                z = y[:]
                y.clear()
                return z

            df_copy['reviews'] = df_copy['reviews'].apply(stem_words)

            # Join back
            def join_back(list_input):
                return " ".join(list_input)

            df_copy['reviews'] = df_copy['reviews'].apply(join_back)

            from sklearn.feature_extraction.text import CountVectorizer
            cv = CountVectorizer(max_features=1000)
            x = cv.fit_transform(df_copy['reviews']).toarray()

            predict = clf.predict(x)
            predict = predict.tolist()
            df['prediction'] = predict

            rating_generated.append(round((df[df['prediction'] == 1].shape[0] / df.shape[0]) * 10, 1))

        else:
            rating_generated.append(0)

        length=""
        if df.shape[0]>0:
            length=df.shape[0]
        else:
            length=0


        return render_template("prediction.html", movie_title=movie_title, movie_id=movie_id, movie_year=movie_year,movie_genre=movie_genre,
                               movie_duration=movie_duration,movie_imdb_rating=movie_imdb_rating,pic=pic,
                               df=df, size=length, rating_generated=rating_generated[0])

    except:
        return render_template('index2.html')


@app.route('/')
def home():
    return render_template('index.html')


if __name__=="__main__":
    app.run(debug=True)