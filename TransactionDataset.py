from itertools import chain
from collections import Counter

class TransactionDataset:
    def __init__(self):
        self.transactions = []

    def add_transaction(self, transaction):
        self.transactions.append(transaction)

    def get_frequent_items(self, min_support):
        item_counts = Counter(chain(*self.transactions))
        num_transactions = len(self.transactions)
        frequent_items = []
        for item, count in item_counts.items():
            support = count / num_transactions
            if support >= min_support:
                frequent_items.append((item, support))
        return frequent_items

import numpy as np
from collections import Counter
from itertools import chain, combinations

class Apriori:
    def __init__(self, dataset, min_support):
        self.dataset = dataset
        self.min_support = min_support
        self.frequent_items = self._generate_frequent_items()

    def _generate_frequent_items(self):
        print('1')
        return self.dataset.get_frequent_items(self.min_support)

    def _generate_candidate_itemsets(self, frequent_itemsets, k):
        candidate_itemsets = []
        for i in range(len(frequent_itemsets)):
            for j in range(i + 1, len(frequent_itemsets)):
                itemset1 = frequent_itemsets[i]
                itemset2 = frequent_itemsets[j]

                print(itemset1)
                print(itemset2)
                if itemset1[:k - 2] == itemset2[:k - 2]:
                    candidate_itemsets.append(sorted(set(itemset1).union(itemset2)))
        return candidate_itemsets

    def _calculate_support(self, itemset):
        count = 0
        for transaction in self.dataset.transactions:
            if itemset.issubset(transaction):
                count += 1
        return count / len(self.dataset.transactions)

    def _generate_frequent_itemsets(self, k):
        candidate_itemsets = self._generate_candidate_itemsets(self.frequent_items, k)
        for itemset, _ in candidate_itemsets:
            support = self._calculate_support(itemset)
            if support >= self.min_support:
                self.frequent_items.append((itemset, support))
        return len(candidate_itemsets) > 0

    def _generate_association_rules(self, itemset, remaining_itemset):
        rules = []
        for item in remaining_itemset:
            rule = (itemset - item, item)
            rules.append(rule)
        return rules

    def mine(self):
        k = 5
        while self._generate_frequent_itemsets(k):
            k += 1

        association_rules = []
        for itemset, _ in self.frequent_items[1:]:
            subset_itemsets = map(frozenset, [combinations(itemset, r) for r in range(1, len(itemset))])
            for subset_itemset in subset_itemsets:
                remaining_itemset = itemset - subset_itemset
                confidence = self._calculate_support(itemset) / self._calculate_support(subset_itemset)
                if confidence >= self.min_support:
                    rules = self._generate_association_rules(itemset, remaining_itemset)
                    association_rules.extend(rules)

        return self.frequent_items, association_rules


dataset = TransactionDataset()

dataset.add_transaction(['A', 'B', 'C'])
dataset.add_transaction(['A', 'B'])
dataset.add_transaction(['A', 'C'])
dataset.add_transaction(['B', 'C'])
dataset.add_transaction(['A', 'B', 'C', 'D'])
dataset.add_transaction(['A', 'D'])
dataset.add_transaction(['B', 'D'])
dataset.add_transaction(['C', 'D'])

# Create an instance of the Apriori algorithm with a minimum support of 0.3
apriori = Apriori(dataset, min_support=0.3)
print(apriori.mine())