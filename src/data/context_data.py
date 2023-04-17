import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset

# Book DataFrame Preprocessing
from .books_data import book_language_change,publisher_change,category_change,category_fill

def age_map(x: int) -> int:
    x = int(x)
    if x < 20:
        return 1
    elif x >= 20 and x < 30:
        return 2
    elif x >= 30 and x < 40:
        return 3
    elif x >= 40 and x < 50:
        return 4
    elif x >= 50 and x < 60:
        return 5
    else:
        return 6
    
def year_map(x: int) -> int:
    x = int(x)
    if  x < 1950:
        return 1
    elif x >= 1950 and x < 1960:
        return 2
    elif x >= 1960 and x < 1970:
        return 3
    elif x >= 1970 and x < 1980:
        return 4
    elif x >= 1980 and x < 1990:
        return 5
    elif x >= 1990 and x < 2000:
        return 6
    elif x >= 2000 and x < 2010:
        return 6
    elif x >= 2010 and x < 2020:
        return 6
    else:
        return 7

def process_context_data(users, books, ratings1, ratings2):
    """
    Parameters
    ----------
    users : pd.DataFrame
        users.csv를 인덱싱한 데이터
    books : pd.DataFrame
        books.csv를 인덱싱한 데이터
    ratings1 : pd.DataFrame
        train 데이터의 rating
    ratings2 : pd.DataFrame
        test 데이터의 rating
    ----------
    """

    users['location_city'] = users['location'].apply(lambda x: x.split(',')[0])
    users['location_state'] = users['location'].apply(lambda x: x.split(',')[1])
    users['location_country'] = users['location'].apply(lambda x: x.split(',')[2])
    users = users.drop(['location'], axis=1)

    # location 전처리
    city_tab = pd.DataFrame([['iowa city','iowa','usa'],
    ['somerset', 'somerset','england'],
    ['milford','massachusetts','usa'],
    ['rockvale','tennessee','usa'],
    ['bronx','newyork','usa'],
    ['tustin','california','usa'],
    ['choctaw','choctaw','usa'],
    ['richmond hill','richmond hill','canada'],
    ['kuala lumpur','kuala lumpur','malaysia']])
    city_tab.columns = ['city','state','country']

    for _,row in city_tab.iterrows():
        location_idx = users['location_city'] == row.city
        users.loc[location_idx,'location_city'] = row.city
        users.loc[location_idx,'location_state'] = row.state
        users.loc[location_idx,'location_country'] = row.country

    # users.fillna('unknown',inplace=True)
    users['location_country'] = users['location_country'].str.replace('n/a','unknown')
    users['location_city'] = users['location_city'].str.replace('n/a','unknown')
    users['location_state'] = users['location_state'].str.replace('n/a','unknown')

    # isbn 데이터 오기입 수정
    # books['isbn'] = books['img_url'].apply(lambda x: x.split('P/')[1][:10])
    # books['isbn'] = books['isbn'].astype(int)

    # Item 
    ##################################### Category Preprocessing ##############################################
    # books['category'].str.lower()
    # books['category'] = books['category'].apply(lambda x: re.sub('[\W_]+',' ',str(x)).strip())
    # books['language'] = books['language'].astype(str)
    
    # categories = ['garden','crafts','physics','adventure','music','fiction','nonfiction','science','science fiction','social','homicide',
    # 'sociology','disease','religion','christian','philosophy','psycholog','mathemat','agricult','environmental',
    # 'business','poetry','drama','literary','travel','motion picture','children','cook','literature','electronic',
    # 'humor','animal','bird','photograph','computer','house','ecology','family','architect','camp','criminal','language','india']

    # books['category_high'] = books['category'].copy()

    # for category in categories:
    #     books.loc[books[books['category'].str.contains(category,na=False)].index,'category_high'] = category

    # category_high_df = pd.DataFrame(books['category_high'].value_counts()).reset_index()
    # category_high_df.columns = ['category','count']
    # others_list = category_high_df[category_high_df['count']<5]['category'].values

    # books.loc[books[books['category_high'].isin(others_list)].index, 'category_high'] = 'others'
    # books['category'] = books['category_high'].copy()

    # books = books.drop(['category_high'],axis=1)

    ############################################################################################

    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)

    # indexing
    loc_city2idx = {v:k for k,v in enumerate(users['location_city'].unique())}
    loc_state2idx = {v:k for k,v in enumerate(users['location_state'].unique())}
    loc_country2idx = {v:k for k,v in enumerate(users['location_country'].unique())}

    category2idx = {v:k for k,v in enumerate(books['category'].unique())}
    publisher2idx = {v:k for k,v in enumerate(books['publisher'].unique())}
    language2idx = {v:k for k,v in enumerate(books['language'].unique())}
    author2idx = {v:k for k,v in enumerate(books['book_author'].unique())}

    users['location_city'] = users['location_city'].map(loc_city2idx)
    users['location_state'] = users['location_state'].map(loc_state2idx)
    users['location_country'] = users['location_country'].map(loc_country2idx)
    
    # print(users.head())

    users['age'] = users['age'].fillna(int(users['age'].mean()))
    users['age'] = users['age'].apply(year_map)

    books['year_of_publication'] = books['year_of_publication'].fillna(int(books['year_of_publication'].mean()))
    books['year_of_publication'] = books['year_of_publication'].apply(age_map)
    year_of_publication2idx = {v:k for k,v in enumerate(books['year_of_publication'].unique())}

    books['year_of_publication'] = books['year_of_publication'].map(year_of_publication2idx)
    books['category'] = books['category'].map(category2idx)
    books['publisher'] = books['publisher'].map(publisher2idx)
    books['language'] = books['language'].map(language2idx)
    books['book_author'] = books['book_author'].map(author2idx)

    # print(books[['isbn', 'category', 'publisher', 'language', 'book_author']])
    # print(books[['isbn', 'category', 'publisher', 'language', 'book_author']].info)

    train_df = ratings1.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author', 'year_of_publication']], on='isbn', how='left')
    test_df = ratings2.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author', 'year_of_publication']], on='isbn', how='left')

    idx = {
        "loc_city2idx":loc_city2idx,
        "loc_state2idx":loc_state2idx,
        "loc_country2idx":loc_country2idx,
        "category2idx":category2idx,
        "publisher2idx":publisher2idx,
        "language2idx":language2idx,
        "author2idx":author2idx,
        'year_of_publication2idx':year_of_publication2idx
    }

    return idx, train_df, test_df


def context_data_load(args):
    """
    Parameters
    ----------
    Args:
        data_path : str
            데이터 경로
    ----------
    """

    ######################## DATA LOAD ########################
    users = pd.read_csv(args.data_path + 'users.csv')
    books = pd.read_csv(args.data_path + 'books.csv')
    train = pd.read_csv(args.data_path + 'train_ratings.csv')
    test = pd.read_csv(args.data_path + 'test_ratings.csv')
    sub = pd.read_csv(args.data_path + 'sample_submission.csv')

    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

    #### books 전처리
    books = books_preprocessing(books.copy(), args.data_path, save_csv_load = args.books)
    
    # 결측치 
    print(books.isna().sum())

    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

    user2idx = {id:idx for idx, id in idx2user.items()}
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}

    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)
    test['user_id'] = test['user_id'].map(user2idx)
    users['user_id'] = users['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)
    books['isbn'] = books['isbn'].map(isbn2idx)

    idx, context_train, context_test = process_context_data(users, books, train, test)
    field_dims = np.array([len(user2idx), len(isbn2idx),
                            6, len(idx['loc_city2idx']), len(idx['loc_state2idx']), len(idx['loc_country2idx']),
                            len(idx['category2idx']), len(idx['publisher2idx']), len(idx['language2idx']), len(idx['author2idx']), len(idx['year_of_publication2idx'])], dtype=np.uint32)

    data = {
            'train':context_train,
            'test':context_test.drop(['rating'], axis=1),
            'field_dims':field_dims,
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            }


    return data


def context_data_split(args, data):
    """
    Parameters
    ----------
    Args:
        test_size : float
            Train/Valid split 비율을 입력합니다.
        seed : int
            랜덤 seed 값
    ----------
    """

    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['train'].drop(['rating'], axis=1),
                                                        data['train']['rating'],
                                                        test_size=args.test_size,
                                                        random_state=args.seed,
                                                        shuffle=True
                                                        )
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    return data

def context_data_loader(args, data):
    """
    Parameters
    ----------
    Args:
        batch_size : int
            데이터 batch에 사용할 데이터 사이즈
        data_shuffle : bool
            data shuffle 여부
    ----------
    """

    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.data_shuffle)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=args.data_shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data

## -------------------------------------------- 추가한 내용 -------------------------------------------- ##
def books_preprocessing(books: pd.DataFrame, path : str, save_csv_load : bool = False) -> pd.DataFrame:
    if save_csv_load:
        books = pd.read_csv(path + 'books_process_data.csv')
    else:
        print('--------------- BOOKS DATA PROCESS ---------------')
        print('--------------- BOOKS LANGUAGE PROCESS #1/3 ---------------')
        books = book_language_change(books)

        print('--------------- BOOKS PUBLISHER PROCESS #2/3 ---------------')
        books = publisher_change(books)

        print('--------------- BOOKS CATEGORY PROCESS #3/3 ---------------')
        books = category_change(books)
        books = category_fill(books)
        print('--------------- to_csv : books_process_data ---------------')
        books.to_csv(path + 'books_process_data.csv', index=False)
        print('FIN')
    return books