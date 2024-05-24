from sqlalchemy import create_engine, Column, Integer, String, DateTime,Float
from sqlalchemy.orm import sessionmaker,relationship

from sqlalchemy.orm import declarative_base
Base = declarative_base()

class Train(Base):
    __tablename__='train_datas'

    x=Column(Float,nullable=False,primary_key=True)
    y1=Column(Float,nullable=False)
    y2=Column(Float,nullable=False)
    y3=Column(Float,nullable=False)
    y4=Column(Float,nullable=False)

class Test(Base):
    __tablename__='test_datas'


    x=Column(Float,nullable=False,primary_key=True)
    y=Column(Float,nullable=False)
    delta_x=Column(Float,nullable=True)
    func_num=Column(Float,nullable=True)


class IdealFunctions(Base):
    __tablename__ = 'ideal_functions'
    x=Column(Float,nullable=False,primary_key=True)
    y1 = Column(Float)
    y2 = Column(Float)
    y3 = Column(Float)
    y4 = Column(Float)
    y5 = Column(Float)
    y6 = Column(Float)
    y7 = Column(Float)
    y8 = Column(Float)
    y9 = Column(Float)
    y10 = Column(Float)
    y11 = Column(Float)
    y12 = Column(Float)
    y13 = Column(Float)
    y14 = Column(Float)
    y15 = Column(Float)
    y16 = Column(Float)
    y17 = Column(Float)
    y18 = Column(Float)
    y19 = Column(Float)
    y20 = Column(Float)
    y21 = Column(Float)
    y22 = Column(Float)
    y23 = Column(Float)
    y24 = Column(Float)
    y25 = Column(Float)
    y26 = Column(Float)
    y27 = Column(Float)
    y28 = Column(Float)
    y29 = Column(Float)
    y30 = Column(Float)
    y31 = Column(Float)
    y32 = Column(Float)
    y33 = Column(Float)
    y34 = Column(Float)
    y35 = Column(Float)
    y36 = Column(Float)
    y37 = Column(Float)
    y38 = Column(Float)
    y39 = Column(Float)
    y40 = Column(Float)
    y41 = Column(Float)
    y42 = Column(Float)
    y43 = Column(Float)
    y44 = Column(Float)
    y45 = Column(Float)
    y46 = Column(Float)
    y47 = Column(Float)
    y48 = Column(Float)
    y49 = Column(Float)
    y50 = Column(Float)

engine = create_engine('sqlite:///assignment_data.db')

Base.metadata.create_all(engine)

import pandas as pd

train = pd.read_csv('train.csv')

train.to_sql('train_datas', con=engine, if_exists='replace', index=False)


df = pd.read_sql('SELECT * FROM train_datas', con=engine)

print(df)

test = pd.read_csv('test.csv')

test.to_sql('test_datas', con=engine, if_exists='replace', index=False)


df = pd.read_sql('SELECT * FROM test_datas', con=engine)
print(df)

ideal = pd.read_csv('ideal.csv')

ideal.to_sql('ideal_functions', con=engine, if_exists='replace', index=False)


df = pd.read_sql('SELECT * FROM ideal_functions', con=engine)
print(df)

def load_data_to_dataframe(table_name, engine):
    return pd.read_sql_table(table_name, con=engine)

train_data = load_data_to_dataframe('train_datas', engine)
ideal_functions = load_data_to_dataframe('ideal_functions', engine)
test_data = load_data_to_dataframe('test_datas', engine)

import pandas as pd
import numpy as np

def calculate_least_squares(y_train, y_ideal):
    return np.sum((y_train - y_ideal) ** 2)

def find_best_ideal_functions(training_data, ideal_functions, n=4):
    if not isinstance(training_data, pd.DataFrame) or not isinstance(ideal_functions, pd.DataFrame):
        raise ValueError("Both training_data and ideal_functions must be pandas DataFrames.")
    if not isinstance(n, int):
        raise ValueError("n must be an integer.")

    best_funcs = {}
    
    for i in range(1, n + 1):
        min_deviation = float('inf')
        best_func = None
        
        for ideal in ideal_functions.columns[1:]:
            deviation = calculate_least_squares(training_data[f'y{i}'], ideal_functions[ideal])
            
            if deviation < min_deviation:
                min_deviation = deviation
                best_func = ideal
        
        best_funcs[f'IdealFunc{i}'] = best_func
    
    return best_funcs

best_funcs = find_best_ideal_functions(train_data, ideal_functions)
print(best_funcs)


import numpy as np
import pandas as pd

def map_test_data(test_data, ideal_functions, best_funcs):
    if not isinstance(test_data, pd.DataFrame) or not isinstance(ideal_functions, pd.DataFrame):
        raise ValueError("test_data and ideal_functions must be pandas DataFrame objects")
    
    if not 'x' in test_data or not 'y' in test_data:
        raise ValueError("test_data must contain 'x' and 'y' columns")
    
    if not 'x' in ideal_functions:
        raise ValueError("ideal_functions must contain an 'x' column")
    ideal_functions = ideal_functions.set_index('x')
    
    results = []
    max_deviation_threshold = float('inf')  

    for index, row in test_data.iterrows():
        x = row['x']
        y_test = row['y']
        closest_func = "none"  
        min_deviation = float('inf')
        
        for train_col, ideal_col in best_funcs.items():
            if x in ideal_functions.index:
                y_ideal = ideal_functions.at[x, ideal_col] if pd.notna(ideal_functions.at[x, ideal_col]) else np.nan
                if pd.notna(y_ideal):
                    deviation = (y_test - y_ideal) ** 2
                    if deviation < min_deviation:
                        min_deviation = deviation
                        closest_func = ideal_col
        
        delta_x = np.sqrt(min_deviation) if min_deviation != float('inf') and min_deviation <= max_deviation_threshold else np.nan
        
        results.append({'x': x, 'y': y_test, 'delta_x': delta_x, 'func_num': closest_func})

    return pd.DataFrame(results)

try:
    mapped_test_data = map_test_data(test_data, ideal_functions, best_funcs)
    mapped_test_data.to_sql('test_datas', con=engine, if_exists='replace', index=False)
    test_data = pd.read_sql('SELECT * FROM test_datas', con=engine)
    print(test_data)
except Exception as e:
    print(f"An error occurred: {e}")



import matplotlib.pyplot as plt
import seaborn as sns

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
Base = declarative_base()

def plot_train_data(session):
    train_data = session.query(Train).all()
    x = [data.x for data in train_data]
    y1 = [data.y1 for data in train_data]
    y2 = [data.y2 for data in train_data]
    y3 = [data.y3 for data in train_data]
    y4 = [data.y4 for data in train_data]

    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, label='y1')
    plt.plot(x, y2, label='y2')
    plt.plot(x, y3, label='y3')
    plt.plot(x, y4, label='y4')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Train Data')
    plt.legend()
    plt.show()

def plot_test_data(session):
    test_data = session.query(Test).all()
    x = [data.x for data in test_data]
    y = [data.y for data in test_data]
    func_num=[data.func_num for data in test_data]
    delta_x=[data.delta_x for data in test_data]
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Test Data', color='r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Test Data')
    plt.legend()
    plt.show()
    plt.scatter(delta_x, func_num, label='Test Results', color='r')
    plt.xlabel('delta_x')
    plt.ylabel('func_num')
    plt.title('Test Results')
    plt.legend()
    plt.show()


engine = create_engine('sqlite:///assignment_data.db') 
Session = sessionmaker(bind=engine)
session = Session()

plot_train_data(session)
plot_test_data(session)


import unittest
import pandas as pd
import numpy as np

class TestIdealFunctions(unittest.TestCase):
    def setUp(self):
       
        self.train_data = pd.DataFrame({
            'x': np.linspace(-20.0, 19.9, 100), 
            'y1': np.linspace(-10, 10, 100), 
            'y2': np.linspace(10, -10, 100), 
            'y3': np.linspace(-5, 5, 100),   
            'y4': np.linspace(5, -5, 100)     
        })
        self.ideal_functions = pd.DataFrame({
            'x': np.linspace(-20.0, 19.9, 100),  
            'func1': np.linspace(-10, 10, 100), 
            'func2': np.linspace(10, -10, 100), 
            'func3': np.linspace(-5, 5, 100),  
            'func4': np.linspace(5, -5, 100)    
        })

    def test_valid_input(self):
        best_funcs = find_best_ideal_functions(self.train_data, self.ideal_functions, 4)
        expected_funcs = {
            'IdealFunc1': 'func1',
            'IdealFunc2': 'func2',
            'IdealFunc3': 'func3',
            'IdealFunc4': 'func4'  
        }
        self.assertEqual(best_funcs, expected_funcs)

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)


import unittest

class TestDataMapping(unittest.TestCase):
    
    def test_map_test_data(self):
       
        test_data = pd.DataFrame({'x': [1, 2, 3], 'y': [10, 20, 30]})
        ideal_functions = pd.DataFrame({'x': [1, 2, 3], 'func1': [10, 20, 30], 'func2': [9, 19, 29]})
        best_funcs = {'y': 'func1'}
        
        result = map_test_data(test_data, ideal_functions, best_funcs)
        
        expected_output = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [10, 20, 30],
            'delta_x': [0.0, 0.0, 0.0],
            'func_num': ['func1', 'func1', 'func1']
        })
        pd.testing.assert_frame_equal(result, expected_output)
if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)




