import pandas as pd
raw_data = {'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
        'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'],
        'age': [42, 52, 36, 24, 73],
        'preTestScore': [4, 24, 31, 2, 3],
        'postTestScore': [25, 94, 57, 62, 70]}
raw_data = {'BoardState': [[1,2,3]]}
df = pd.DataFrame(raw_data, columns = ['BoardState'])
print(df)
df.to_csv('example.csv')



# with open('my_csv.csv', 'a') as f:
#     df.to_csv(f, header=False)


# df = pd.read_csv('foo.csv', index_col=0)

# with open('foo.csv', 'a') as f:
#     (df + 6).to_csv(f, header=False)
