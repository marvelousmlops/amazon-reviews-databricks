import os
import urllib
import gzip
import pandas as pd
import codecs
import re


class AmazonReviewsDataLoader:
    """
    Downloads & returns data from Amazon reviews public dataset for the following categories:
    'Gourmet_Foods', 'Pet_Supplies', 'Health', 'Baby'
    """
    def __init__(self, path):
        """
        :param path: path to download the data
        """
        self.path = path
        self.list_to_download = ['Gourmet_Foods', 'Pet_Supplies', 'Health', 'Baby', 'categories']


    def download_data(self):
        """
        Downloads & returns data from Amazon reviews public dataset for the following categories:
        'Gourmet_Foods', 'Pet_Supplies', 'Health', 'Baby'
        :return:
        """

        os.makedirs(self.path, exist_ok=True)

        for cat in self.list_to_download:
            if not os.path.exists(f'{os.path.join(self.path, cat)}.txt'):
                data = urllib.request.urlopen(
                    f'https://archive.org/download/amazon-reviews-1995-2013/{cat}.txt.gz').read()
                with open(f'{os.path.join(self.path, cat)}.txt.gz', 'wb') as f:
                    f.write(data)

                g = gzip.open(f'{os.path.join(self.path, cat)}.txt.gz', 'rb')
                with open(f'{os.path.join(self.path, cat)}.txt', 'wb') as outfile:
                    for line in g:
                        outfile.write(line)
                print(f"File {cat} is downloaded")

    def create_reviews_dataframe(self):
        """
        Creates reviews dataframe
        :return: pd.Dataframe
        """
        self.download_data()

        df_list = []

        for cat in self.list_to_download[0:-1]:

            with open(f'{os.path.join(self.path, cat)}.txt', 'r') as f:
                reviews = f.read()

            reviews_list = reviews.split('\n\n')
            headers = ['productId', 'Title', 'userId', 'Helpfulness', 'Score', 'Text']

            all_records = []
            ## the last record is empty
            for i in range(len(reviews_list) - 1):
                record = []
                for line in [0, 1, 3, 5, 6, 9]: # only get relevant information
                    record.append(reviews_list[i].split('\n')[line].split(': ')[1])
                all_records.append(record)

            df = pd.DataFrame(all_records, columns=headers)
            df_list.append(df)

        df_reviews = pd.concat(df_list)

        with codecs.open(os.path.join(self.path, 'categories.txt'), "r", "latin_1") as f:
            categories = f.read()

        categories_list = categories.split('\n')
        headers = ['productId', 'Cat1', 'Cat2', 'Cat3']

        all_records = []
        for i in range(len(categories_list)):
            if 'B00' in categories_list[i - 1]:
                record = [categories_list[i - 1]] + categories_list[i].split(", ")
                if len(record) < 4:
                    record = record + ['Unknown'] * (4 - len(record))
                else:
                    record = record[0:4]
                all_records.append(record)

        df_categories = pd.DataFrame(all_records, columns=headers)

        df = df_reviews.merge(df_categories, on="productId", how='inner')
        df = df[pd.notnull(df.Text)].drop_duplicates().reset_index(drop=True)
        for col in ['Cat1', 'Cat2', 'Cat3']:
            df[col] = df[col].apply(lambda x: str.lower(re.sub("[^a-zA-Z]", ' ', x)).replace('  ', ' ').strip())
        df = df[df.Cat1.isin(['pet supplies', 'baby products', 'beauty', 'grocery  gourmet food'])]

        return df