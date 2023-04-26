import numpy as np
import pandas as pd

ISBN_COUNRY_NUMBER = 2
PUBLISHER_NUMBER = 4

def book_language_change(books: pd.DataFrame) -> pd.DataFrame:
    from collections import Counter
    """
        books_isbn_country : dict()
        ISBN_COUNRY_NUMBER : int - 국가 번호 자리가 명확하지 않아서 수정하여 사용
    """
    books_isbn_country = {}
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
    for i, v in books.iterrows():
        key = v['isbn'][ISBN_COUNRY_NUMBER:ISBN_COUNRY_NUMBER+PUBLISHER_NUMBER]
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
        books.loc[i, 'publisher']=book_pub[books.loc[i]['isbn'][ISBN_COUNRY_NUMBER:ISBN_COUNRY_NUMBER+PUBLISHER_NUMBER]]

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
    
    others_list = category_df[category_df['count']<10]['category'].values
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
            name_dict[name] = df_[df_['book_author']== name]['category'].value_counts().index[0]
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