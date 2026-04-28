import pandas as pd
df = pd.read_excel(r'c:\Users\Admin\Desktop\Новая папка\result\avito.xlsx')
with open(r'c:\Users\Admin\Desktop\Новая папка (2)\avito_cols.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(df.columns))
    f.write('\n\n')
    f.write(str(df.head(2).to_dict('records')))
