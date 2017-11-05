import mysql.connector
import time
cnx = mysql.connector.connect(user='root', database='CapitalOne', password='root')
cursor = cnx.cursor()
cursor.execute("DROP TABLE IF EXISTS Merchant")
sql = """CREATE TABLE MerchantCategory (Merchant varchar(50), Category varchar(50))"""
cursor.execute(sql)
with open("./DataSet/TabelMerchantCategoryNew.txt") as f:
    text = f.readlines()

for line in text[1:]:
    fields = line.split("\t")
    merchant, category = fields[0], fields[-1]
    cursor.execute("""Insert into MerchantCategory values (%s,%s)""", (merchant, category))
cnx.commit()
print("Done")
