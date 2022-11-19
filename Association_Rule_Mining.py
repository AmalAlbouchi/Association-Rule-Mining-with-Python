import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


# Convert text file to dataframe.
# delimiter : Delimiter to use.
# header : Row number(s) to use as the column names, and the start of the data.
# names : List of column names to use.
df = pd.read_csv('market_basket.txt', delimiter="\t", header=0, names=["ID", "Product"])


# Return the first 10 rows.
print("\nFirst 10 rows of the DataFrame\n")
print(df.head(10))


# Return DataFrame dimensions.
print("\nDataFrame dimensions")
print("Number of rows : ", df.shape[0], '\nNumber of columns : ', df.shape[1],'\n')


# Change the DataFrame into a Binary Table.
df_binary = pd.get_dummies(df, prefix='', prefix_sep='').groupby("ID").sum()
df_binary.to_csv("binaryTable.csv")


# Return the first 30 transactions.
# Only returning the first 3 in each Itemset.
# Group by the first column (ID) and get second column (Products) as lists.
df_chariots = df.groupby("ID")['Product'].apply(list)
print("\nFirst 30 transactions\n")
for i in range(0,30) :
    j = 0
    print(i ,' - | ', end="")
    for k in df_chariots.iloc[i]:
        print(k, "| ", end="")
        j=j+1
        if (j==3) : 
            break
    print() # NewLine.


# Frequent Itemsets.
print("\nFirst 15 Frequent Itemsets\n")
frequent_itemsets = apriori(df_binary.astype('bool'), min_support = 0.025, max_len=4, use_colnames = True)
print(frequent_itemsets.head(15))


# Printing frequent Itemsets containing Aspirin.
print("\nFrequent Itemsets containing Aspirin\n")
A = {'Aspirin'}
for i in frequent_itemsets.itertuples():
    if (A.issubset(i.itemsets)) :
        print(list(i.itemsets))


# Printing frequent Itemsets containing Aspirin & Eggs.
print("\nFrequent Itemsets containing Aspirin & Eggs\n")
A = {'Aspirin','Eggs'}
for i in frequent_itemsets.itertuples():
    if (A.issubset(i.itemsets)) :
        print(list(i.itemsets))


# Association Rules.
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.75)
rules.to_csv("rules.csv")
print('\nFirst 5 association rules\n')
print(rules.head(5))


# Print rules with lift >= 7.
print('\nRules with lift >= 7\n')
for i in rules.itertuples():
    if (i.lift >= 7) :
        print('Antecedants : ',list(i.antecedents),' ==> Consequents : ',list(i.consequents))


# Print Rules with Consequent = {'2pct_Milk'}.
print('\nRules with Consequent = 2pct_Milk\n')
B={'2pct_Milk'}
for i in rules.itertuples():
    if (B == i.consequents) :
        print('Antecedants : ',list(i.antecedents),' ==> Consequents : ',list(i.consequents))
