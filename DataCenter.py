import numpy as np
from sklearn import preprocessing
import os
import mysql.connector
from datetime import datetime
from AppUtils import logger


class DataCenter:
    def __init__(self):
        """
        ['paypal', 'store', 'education', 'travel', 'food', 'other', 'entertainment', 'health',
         'mobile', 'transportation', 'service', 'parking']
        """
        self.review_sample_rate = {
            'paypal': [0.1, 0.9],
            'store': [0.1, 0.9],
            'education': [0.6, 0.99],
            'travel': [0.1, 0.7],
            'food': [0.1, 0.9],
            'other': [0.3, 0.7],
            'entertainment': [0.1, 0.6],
            'health': [0.3, 0.9],
            'mobile': [0.2, 0.8],
            'transportation': [0.1, 0.9],
            'service': [0.2, 0.8],
            'parking': [0.1, 0.7]
        }

    def mock_data(self, number_records=320):

        inputs = np.random.randint(low=0, high=5, size=number_records)
        lb = preprocessing.LabelBinarizer()
        lb.fit(np.arange(5))
        inputs_one_hot = lb.transform(inputs)

        # Create a 20*4 matrix
        number_feature_continuous = 5
        input_continous_feature = np.random.rand(number_records, number_feature_continuous)
        inputs = np.concatenate([inputs_one_hot, input_continous_feature], axis=1)

        targets = np.random.randint(low=0, high=3, size=number_records)
        lb = preprocessing.LabelBinarizer()
        lb.fit(np.arange(3))
        targets_one_hot = lb.transform(targets)
        targets = targets_one_hot

        return inputs, targets

    def __get_category_data(self):
        '''
        Get Category Merchant data, should save into backend to keep order same every time. (Using set not ensure order)
        :return:
        '''
        path_category_to_int = "{}/category_to_int.npy".format("./DataSet")
        path_int_to_category = "{}/int_to_category.npy".format("./DataSet")

        # if merchant category data not exist, query from db again and save to backend
        if not os.path.exists(path_int_to_category) or not os.path.exists(path_category_to_int):
            # cnx = mysql.connector.connect(user='root', database='CapitalOne', password='root')
            cnx = mysql.connector.connect(host="18.216.190.153",
                                          user='cap1app',
                                          database='CapitalOne',
                                          password='cap1app')
            cursor = cnx.cursor()
            cursor.execute("select * from MERCHANT")
            merchant_data = cursor.fetchall()

            # print(type(merchant_data))
            category = set(map(lambda x: x[1], merchant_data))
            category_to_int = {key: idx for idx, key in enumerate(category)}
            int_to_category = dict(enumerate(category))
            np.save(path_category_to_int, category_to_int)
            np.save(path_int_to_category, int_to_category)
            cursor.close()
            cnx.close()
        else:
            category_to_int = np.load(path_category_to_int).item()
            int_to_category = np.load(path_int_to_category).item()
        return category_to_int, int_to_category

    def __record_to_vector(self, record, category_to_int):
        """
        c.gender, c.dob, c.is_primary, c.is_married, t.amount, t.year, (t.country), (t.review), m.Category
        Mock probability [Pos][Negative][Positive]

        :param record:
        :return:
        """
        record = list(record)
        category_ont_hot = [0 for _ in range(len(category_to_int))]
        # handle data not associated, if not exist then assign as 'other' which is 5.
        category_ont_hot[category_to_int.get(record[-1], category_to_int['other'])] = 1
        record[0] = 1 if record[0] == "Male" else 0
        # Dob to age
        record[1] = datetime.now().year - int(record[1].split("/")[-1])
        record[2] = int(record[2])
        record[3] = int(record[3])
        record[4] = int(record[4])
        # mock the review based on sample probability
        pos, neutral = self.review_sample_rate.get(record[-1], [0.3, 0.7])
        # print(pos, neutral)
        random_sample = np.random.uniform(low=0.0, high=1.0)
        # print(random_sample)
        if random_sample < pos:
            review_mock = 0
        elif random_sample < neutral:
            review_mock = 1
        else:
            review_mock = 2
        # print(review_mock)
        return record[:-2] + category_ont_hot + [review_mock]

    def __fetch_train_data(self):
        inputs, targets = None, None
        category_to_int, int_to_category = self.__get_category_data()
        # print(category_to_int)
        cnx = mysql.connector.connect(host="18.216.190.153",
                                      user='cap1app',
                                      database='CapitalOne',
                                      password='cap1app')
        cursor = cnx.cursor()
        # t.country, t.review,
        # where c.customer_id = 100110000
        sql = """
               select c.gender, c.dob, c.is_primary, c.is_married, t.amount, t.year, m.merchant_name, m.Category
             from CUSTOMER as c
             inner join TRANSACTION as t
             on c.customer_id = t.customer_id
             left join MERCHANT as m
             on TRIM(t.merchant_name) = TRIM(m.merchant_name)
             ;
              """
        cursor.execute(sql)
        query_data = cursor.fetchall()
        logger.info("size of query data:{}".format(len(query_data)))
        logger.info("First five query results")

        firstK = len(query_data)
        # for id in range(firstK):
        #     logger.info(query_data[id])
        if query_data:
            processed_data = [self.__record_to_vector(record, category_to_int) for record in query_data]
            targets = [record[-1] for record in processed_data]
            inputs = [record[:-1] for record in processed_data]
            lb = preprocessing.LabelBinarizer()
            lb.fit(np.arange(3))
            targets = lb.transform(targets)

            # logger.info("First five processed inputs")
            # for id in range(firstK):
            #     logger.info(inputs[id])
            # logger.info("First five targets")
            # for id in range(firstK):
            #     logger.info(targets[id])
        return inputs, targets

    def __fetch_new_data(self):
        inputs, targets = None, None
        category_to_int, int_to_category = self.__get_category_data()
        # print(category_to_int)
        cnx = mysql.connector.connect(host="18.216.190.153",
                                      user='cap1app',
                                      database='CapitalOne',
                                      password='cap1app')
        cursor = cnx.cursor()
        # t.country, t.review,
        # where c.customer_id = 100520000
        sql = """
             select c.gender, c.dob, c.is_primary, c.is_married, t.amount, t.year, m.merchant_name, m.Category
             from CUSTOMER as c
             inner join TRANSACTION as t
             on c.customer_id = t.customer_id
             left join MERCHANT as m
             on TRIM(t.merchant_name) = TRIM(m.merchant_name)
             where c.customer_id = 100610000
             limit 1
             ;
              """
        cursor.execute(sql)
        query_data = cursor.fetchall()
        logger.info(query_data[0])
        if query_data:
            processed_data = [self.__record_to_vector(record, category_to_int) for record in query_data]

            targets = [record[-1] for record in processed_data]
            inputs = [record[:-1] for record in processed_data]
            lb = preprocessing.LabelBinarizer()
            lb.fit(np.arange(3))
            targets = lb.transform(targets)

            logger.info("First five processed inputs")
            firstK = 1
            for id in range(firstK):
                logger.info(inputs[id])
            logger.info("First five targets")
            for id in range(firstK):
                logger.info(targets[id])
        return np.array(inputs), np.array(targets)

    def run(self, task):
        inputs, targets = None, None
        if task == "train":
            inputs, targets = self.__fetch_train_data()
            # inputs, targets = DataCenter().mock_data(4)
        if task == "predict":
            inputs, targets = self.__fetch_new_data()
            # inputs, targets = DataCenter().mock_data(320)
        return inputs, targets
