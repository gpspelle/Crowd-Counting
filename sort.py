import csv

with open('truth.csv', 'r') as t1, open('truth_sorted.csv', 'w', newline='') as t2:
   
    f1 = csv.reader(t1, delimiter=',')

    sortedlist = sorted(f1, key=lambda row: row[0])
    print(sortedlist)
    wr = csv.writer(t2, quoting=csv.QUOTE_ALL)
    wr.writerows(sortedlist)
