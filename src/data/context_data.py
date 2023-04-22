import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from collections import Counter

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
    if x >= 1950 and x < 1960:
        return 1
    elif x >= 1960 and x < 1970:
        return 2
    elif x >= 1970 and x < 1980:
        return 3
    elif x >= 1980 and x < 1990:
        return 4
    elif x >= 1990 and x < 2000:
        return 5
    else:
        return 6

def process_context_data(users, books, ratings1, ratings2, args):
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
    """
    users['location'] = users['location'].str.replace(r'[^0-9a-zA-Z:,]', '',regex=True)

    users['location_city'] = users['location'].apply(lambda x: x.split(',')[0])
    users['location_state'] = users['location'].apply(lambda x: x.split(',')[1])
    users['location_country'] = users['location'].apply(lambda x: x.split(',')[2])

    users = users.replace('na', np.nan) 
    users = users.replace('', np.nan)

    users = users.drop(['location'], axis=1)
    """
    
    users = users_preprocessing(users.copy(), args.data_path, save_csv_load = True)
    
    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)
    

    # 인덱싱 처리된 데이터 조인 (문제 해결되면 year_of_publication 추가.)
    context_df = ratings.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author']], on='isbn', how='left')
    #context_df = context_table_preprocessing(context_df,args.data_path, save_csv_load=True, save_mode ='context_table')
    train_df = ratings1.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author']], on='isbn', how='left')
    #train_df = context_table_preprocessing(train_df,args.data_path, save_csv_load=True, save_mode = 'train')
    test_df = ratings2.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author']], on='isbn', how='left')
    #test_df = context_table_preprocessing(test_df,args.data_path, save_csv_load=True, save_mode = 'test')

    # 인덱싱 처리
    loc_city2idx = {v:k for k,v in enumerate(context_df['location_city'].unique())}
    loc_state2idx = {v:k for k,v in enumerate(context_df['location_state'].unique())}
    loc_country2idx = {v:k for k,v in enumerate(context_df['location_country'].unique())}

    train_df['location_city'] = train_df['location_city'].map(loc_city2idx)
    train_df['location_state'] = train_df['location_state'].map(loc_state2idx)
    train_df['location_country'] = train_df['location_country'].map(loc_country2idx)
    test_df['location_city'] = test_df['location_city'].map(loc_city2idx)
    test_df['location_state'] = test_df['location_state'].map(loc_state2idx)
    test_df['location_country'] = test_df['location_country'].map(loc_country2idx)

    train_df['age'] = train_df['age'].fillna(int(train_df['age'].mean()))
    train_df['age'] = train_df['age'].apply(age_map)
    test_df['age'] = test_df['age'].fillna(int(test_df['age'].mean()))
    test_df['age'] = test_df['age'].apply(age_map)

    # book 파트 인덱싱
    category2idx = {v:k for k,v in enumerate(context_df['category'].unique())}
    publisher2idx = {v:k for k,v in enumerate(context_df['publisher'].unique())}
    language2idx = {v:k for k,v in enumerate(context_df['language'].unique())}
    author2idx = {v:k for k,v in enumerate(context_df['book_author'].unique())}

    train_df['category'] = train_df['category'].map(category2idx)
    train_df['publisher'] = train_df['publisher'].map(publisher2idx)
    train_df['language'] = train_df['language'].map(language2idx)
    train_df['book_author'] = train_df['book_author'].map(author2idx)
    test_df['category'] = test_df['category'].map(category2idx)
    test_df['publisher'] = test_df['publisher'].map(publisher2idx)
    test_df['language'] = test_df['language'].map(language2idx)
    test_df['book_author'] = test_df['book_author'].map(author2idx)

    #train_df['year_of_publication'] = train_df['year_of_publication'].apply(year_map)
    #test_df['year_of_publication'] = test_df['year_of_publication'].apply(year_map)

    idx = {
        "loc_city2idx":loc_city2idx,
        "loc_state2idx":loc_state2idx,
        "loc_country2idx":loc_country2idx,
        "category2idx":category2idx,
        "publisher2idx":publisher2idx,
        "language2idx":language2idx,
        "author2idx":author2idx,
    }

    print(train_df.info())

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

    ######################## DATA LOAD
    users = pd.read_csv(args.data_path + 'users.csv')
    books = pd.read_csv(args.data_path + 'books.csv')
    train = pd.read_csv(args.data_path + 'train_ratings.csv')
    test = pd.read_csv(args.data_path + 'test_ratings.csv')
    sub = pd.read_csv(args.data_path + 'sample_submission.csv')

    books = books_preprocessing(books.copy(), args.data_path, save_csv_load = True)

    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

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

    idx, context_train, context_test = process_context_data(users, books, train, test, args)

    field_dims = np.array([len(user2idx), len(isbn2idx), 
                            6, len(idx['loc_city2idx']), len(idx['loc_state2idx']), len(idx['loc_country2idx']),
                            len(idx['category2idx']), len(idx['publisher2idx']), len(idx['language2idx']), len(idx['author2idx'])], dtype=np.uint32)
                            #문제 해결되면 year_of_publication에 해당하는 6 추가.
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
def users_preprocessing(users: pd.DataFrame, path : str, save_csv_load : bool = False) -> pd.DataFrame:

    if save_csv_load:
        users = pd.read_csv(path +'users_process_data.csv')
    else:
        print('--------------- USERS DATA PROCESS ---------------')
        print('--------------- USERS LOCATION PROCESS #1/1 ---------------')
        users = user_location_fill(users)

        print('--------------- to_csv : users_process_data ---------------')
        users.to_csv(path + 'users_process_data.csv', index=False)

        print('FIN')
    
    return users

def user_location_fill(users: pd.DataFrame) -> pd.DataFrame:

    users['location'] = users['location'].str.replace(r'[^0-9a-zA-Z:,]', '',regex=True)

    users['location_city'] = users['location'].apply(lambda x: x.split(',')[0])
    users['location_state'] = users['location'].apply(lambda x: x.split(',')[1])
    users['location_country'] = users['location'].apply(lambda x: x.split(',')[2])

    users = users.replace('na', np.nan) 
    users = users.replace('', np.nan)

    city_base_data = Counter(users['location']).most_common()

    for i in city_base_data:
        users.loc[users[users['location_city']==i[0].split(',')[0]].index,'location_state'] = i[0].split(',')[1]
        users.loc[users[users['location_city']==i[0].split(',')[0]].index,'location_country'] = i[0].split(',')[2]
        users.loc[users[users['location_city'].isna() & users['location_state']==i[0].split(',')[1]].index,'location_city'] = i[0].split(',')[0]

    users = users.drop(['location'], axis=1)

    return users

def context_table_preprocessing(context_table: pd.DataFrame, path : str, save_mode: str, save_csv_load : bool = False):

    if save_csv_load:
        if(save_mode == 'context_table'):
            context_table = pd.read_csv(path +'context_table_process_data.csv')
        elif(save_mode == 'train'):
            context_table = pd.read_csv(path +'train_process_data.csv')
        elif(save_mode == 'test'):
            context_table = pd.read_csv(path +'test_process_data.csv') 
    else:
        print('--------------- CONTEXT_TABLE DATA PROCESS ---------------')
        print('--------------- CONTEXT_TABLE AGE PROCESS #1/1 ---------------')
        context_table = context_table_age_fill(context_table)

        print('--------------- to_csv : context_table_process_data ---------------')
        if(save_mode == 'context_table'):
            context_table.to_csv(path + 'context_table_process_data.csv', index=False)
        elif(save_mode == 'train'):
            context_table.to_csv(path + 'train_process_data.csv', index=False)
        elif(save_mode == 'test'):
            context_table.to_csv(path + 'test_process_data.csv', index=False)    
        

        print('FIN')
    
    return context_table

def context_table_age_fill(context_table: pd.DataFrame) -> pd.DataFrame:
    category_age_table = list(context_table.groupby('category')['age'])

    category_age_dict=dict()

    for ca in category_age_table:
        category_age_dict[ca[0]] = Counter(list(ca[1])).most_common(1)[0]
    
    for i, v in context_table.iterrows():
        if(v['age']!=v['age'] and v['category']==v['category']):
            c = category_age_dict[v['category']][0]
            if(c == c):
                context_table.loc[i,'age'] = c
    
    return context_table



def books_preprocessing(books: pd.DataFrame, path : str, save_csv_load : bool = False) -> pd.DataFrame:
    if save_csv_load:
        books = pd.read_csv(path + 'books_process_data.csv')
    else:
        print('--------------- BOOKS DATA PROCESS ---------------')
        print('--------------- BOOKS LANGUAGE PROCESS #1/3 ---------------')
        books = book_language_change(books)

        print('--------------- BOOKS PUBLISHER PROCESS #2/3 ---------------')
        books = publisher_change(books)

        print('--------------- BOOKS PUBLISHER PROCESS #3/3 ---------------')
        books = category_change(books)
        books = category_fill(books)
        print('--------------- to_csv : books_process_data ---------------')
        books.to_csv(path + 'books_process_data.csv', index=False)
        print('FIN')
    return books

def book_year_of_publication_drop(books: pd.DataFrame) -> pd.DataFrame:

    books.drop(books[books['year_of_publication']<1950].index,axis=0,inplace=True)

    return books

def book_language_change(books: pd.DataFrame) -> pd.DataFrame:

    """
        books_isbn_country : dict()
        ISBN_COUNRY_NUMBER : int - 국가 번호 자리가 명확하지 않아서 수정하여 사용
    """
    books_isbn_country = {}
    ISBN_COUNRY_NUMBER = 1
    for i, v in books.iterrows():
        key = v['isbn'][:ISBN_COUNRY_NUMBER]
        if key not in books_isbn_country:
            books_isbn_country[key] = Counter([v['language']])
        else:
            books_isbn_country[key].update([v['language']])

    idx2isbn_country = {}
    for i, v in books_isbn_country.items():
        new_list = (filter(lambda x: x == x, v)) # nan을 제거한 리스트
        try:
            idx2isbn_country[i] = Counter(new_list).most_common(1)[0][0]
        except:
            idx2isbn_country[i] = np.nan
            
    for i in books[books['language'].isna()].index:
        books.loc[i, 'language']=idx2isbn_country[books.loc[i]['isbn'][:ISBN_COUNRY_NUMBER]]
    return books

def publisher_change(books: pd.DataFrame) -> pd.DataFrame:
    from collections import Counter
    """
    
    """
    books_publisher = {}
    PUBLISHER_NUMBER = 4
    for i, v in books.iterrows():
        key = v['isbn'][:PUBLISHER_NUMBER]
        if key not in books_publisher:
            books_publisher[key] = Counter([v['publisher']])
        else:
            books_publisher[key].update([v['publisher']])

    book_pub = {}
    for i, v in books_publisher.items():
        new_list = (filter(lambda x: x == x, v)) # nan을 제거한 리스트
        try:
            book_pub[i] = Counter(new_list).most_common(1)[0][0]
        except:
            book_pub[i] = np.nan

    for i in books.index:
        books.loc[i, 'publisher']=book_pub[books.loc[i]['isbn'][:PUBLISHER_NUMBER]]
    return books

def category_change(books: pd.DataFrame) -> pd.DataFrame:
    import re
    """
    
    """
    books.loc[books[books['category'].notnull()].index, 'category'] = books[books['category'].notnull()]['category'].apply(lambda x: re.sub('[\W_]+',' ',x).strip())
    books['category'] = books['category'].str.lower()

    # category_df 데이터프레임 생성: category별 책 개수를 저장함
    category_df = pd.DataFrame(books['category'].value_counts()).reset_index()
    # category_df 데이터프레임의 컬럼 이름을 'category'와 'count'로 변경함
    category_df.columns = ['category', 'count']

    categories = list(reversed(category_df[category_df['count']>=50]['category']))
    books['category'] = books['category'].fillna('')
    for category in categories:
        books.loc[books['category'].str.contains(category, case=False), 'category'] = category
    
    others_list = category_df[category_df['count']<50]['category'].values
    books.loc[books[books['category'].isin(others_list)].index, 'category']='others'
    return books

def category_fill(books: pd.DataFrame) -> pd.DataFrame:
    df_ = books[books['category']!='']

    author_df = pd.DataFrame(books['book_author'].value_counts()).reset_index()
    # category_df 데이터프레임의 컬럼 이름을 'category'와 'count'로 변경함
    author_df.columns = ['book_author', 'count']
    author_list = list(author_df[author_df['count']>=10]['book_author'])

    name_dict = {}
    for name in author_list:
        try:
            name_dict[name] = df_[df_['book_author'] == name]['category'].value_counts().index[0]
        except:
            name_dict[name] = np.nan

    for idx, row in books.iterrows():
        if row['category'] == '':
            author = row['book_author']
            if author in name_dict:
                books.loc[idx, 'category'] = name_dict[author]
    empty_category_idx = books[books['category'] == ''].index
    books.loc[empty_category_idx, 'category'] = 'others'
    return books
