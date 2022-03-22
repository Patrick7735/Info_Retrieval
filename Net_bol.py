#!/usr/bin/env python
# coding: utf-8

# In[ ]:


print("this")


# In[ ]:


import math
import string
import nltk
string.punctuation
stop_list=nltk.corpus.stopwords.words('english')
def rem_punctuation(text):
    txt_np="".join([c for c in text if c not in string.punctuation])
    return txt_np
class BooleanModel():
    
    @staticmethod
    def and_operation(left_operand, right_operand):
        # perform 'merge'
        result = []                                 # results list to be returned
        l_index = 0                                 # current index in left_operand
        r_index = 0                                 # current index in right_operand
        jump_l = int(math.sqrt(len(left_operand)))  # skip pointer distance for l_index
        jump_r = int(math.sqrt(len(right_operand))) # skip pointer distance for r_index

        while (l_index < len(left_operand) and r_index < len(right_operand)):
            l_item = left_operand[l_index]  # current item in left_operand
            r_item = right_operand[r_index] # current item in right_operand
            
            # case 1: if match
            if (l_item == r_item):
                result.append(l_item)   # add to results
                l_index += 1            # advance left index
                r_index += 1            # advance right index
            
            # case 2: if left item is more than right item
            elif (l_item > r_item):
                # if r_index can be skipped (if new r_index is still within range and resulting item is <= left item)
                if (r_index + jump_r < len(right_operand)) and right_operand[r_index + jump_r] <= l_item:
                    r_index += jump_r
                # else advance r_index by 1
                else:
                    r_index += 1

            # case 3: if left item is less than right item
            else:
                # if l_index can be skipped (if new l_index is still within range and resulting item is <= right item)
                if (l_index + jump_l < len(left_operand)) and left_operand[l_index + jump_l] <= r_item:
                    l_index += jump_l
                # else advance l_index by 1
                else:
                    l_index += 1

        return result

    @staticmethod
    def or_operation(left_operand, right_operand):
        result = []     # union of left and right operand
        l_index = 0     # current index in left_operand
        r_index = 0     # current index in right_operand

        # while lists have not yet been covered
        while (l_index < len(left_operand) or r_index < len(right_operand)):
            # if both list are not yet exhausted
            if (l_index < len(left_operand) and r_index < len(right_operand)):
                l_item = left_operand[l_index]  # current item in left_operand
                r_item = right_operand[r_index] # current item in right_operand
                
                # case 1: if items are equal, add either one to result and advance both pointers
                if (l_item == r_item):
                    result.append(l_item)
                    l_index += 1
                    r_index += 1

                # case 2: l_item greater than r_item, add r_item and advance r_index
                elif (l_item > r_item):
                    result.append(r_item)
                    r_index += 1

                # case 3: l_item lower than r_item, add l_item and advance l_index
                else:
                    result.append(l_item)
                    l_index += 1

            # if left_operand list is exhausted, append r_item and advance r_index
            elif (l_index >= len(left_operand)):
                r_item = right_operand[r_index]
                result.append(r_item)
                r_index += 1

            # else if right_operand list is exhausted, append l_item and advance l_index 
            else:
                l_item = left_operand[l_index]
                result.append(l_item)
                l_index += 1

        return result

    @staticmethod
    def not_operation(right_operand, indexed_docIDs):
        # complement of an empty list is list of all indexed docIDs
        if (not right_operand):
            return indexed_docIDs
        
        result = []
        r_index = 0 # index for right operand
        for item in indexed_docIDs:
            # if item do not match that in right_operand, it belongs to compliment 
            if (item != right_operand[r_index]):
                result.append(item)
            # else if item matches and r_index still can progress, advance it by 1
            elif (r_index + 1 < len(right_operand)):
                r_index += 1
        return result


# In[ ]:


from __future__ import print_function


class Node(object):
    """Tree node: left and right child + data which can be any object
    """
    def __init__(self, data):
        """Node constructor
        @param data node data object
        """
        self.left = None
        self.right = None
        self.data = data

    def insert(self, data):
        """Insert new node with data
        @param data node data object to insert
        """
        if self.data:
            if data < self.data:
                if self.left is None:
                    self.left = Node(data)
                else:
                    self.left.insert(data)
            elif data > self.data:
                if self.right is None:
                    self.right = Node(data)
                else:
                    self.right.insert(data)
        else:
            self.data = data

    def lookup(self, data, parent=None):
        """Lookup node containing data
        @param data node data object to look up
        @param parent node's parent
        @returns node and node's parent if found or None, None
        """
        if data < self.data:
            if self.left is None:
                return None, None
            return self.left.lookup(data, self)
        elif data > self.data:
            if self.right is None:
                return None, None
            return self.right.lookup(data, self)
        else:
            return self, parent

    def delete(self, data):
        """Delete node containing data
        @param data node's content to delete
        """
        # get node containing data
        node, parent = self.lookup(data)
        if node is not None:
            children_count = node.children_count()
            if children_count == 0:
                # if node has no children, just remove it
                if parent:
                    if parent.left is node:
                        parent.left = None
                    else:
                        parent.right = None
                else:
                    self.data = None
            elif children_count == 1:
                # if node has 1 child
                # replace node by its child
                if node.left:
                    n = node.left
                else:
                    n = node.right
                if parent:
                    if parent.left is node:
                        parent.left = n
                    else:
                        parent.right = n
                else:
                    self.left = n.left
                    self.right = n.right
                    self.data = n.data
            else:
                # if node has 2 children
                # find its successor
                parent = node
                successor = node.right
                while successor.left:
                    parent = successor
                    successor = successor.left
                # replace node data by its successor data
                node.data = successor.data
                # fix successor's parent node child
                if parent.left == successor:
                    parent.left = successor.right
                else:
                    parent.right = successor.right

    def compare_trees(self, node):
        """Compare 2 trees
        @param node tree to compare
        @returns True if the tree passed is identical to this tree
        """
        if node is None:
            return False
        if self.data != node.data:
            return False
        res = True
        if self.left is None:
            if node.left:
                return False
        else:
            res = self.left.compare_trees(node.left)
        if res is False:
            return False
        if self.right is None:
            if node.right:
                return False
        else:
            res = self.right.compare_trees(node.right)
        return res

    def print_tree(self):
        """Print tree content inorder
        """
        if self.left:
            self.left.print_tree()
        print(self.data, end=" ")
        if self.right:
            self.right.print_tree()

    def tree_data(self):
        """Generator to get the tree nodes data
        """
        # we use a stack to traverse the tree in a non-recursive way
        stack = []
        node = self
        while stack or node:
            if node:
                stack.append(node)
                node = node.left
            else:
                # we are returning so we pop the node and we yield it
                node = stack.pop()
                yield node.data
                node = node.right

    def children_count(self):
        """Return the number of children
        @returns number of children: 0, 1, 2
        """
        cnt = 0
        if self.left:
            cnt += 1
        if self.right:
            cnt += 1
        return cnt


# In[ ]:


import nltk
import collections

#from boolean import BooleanModel

# Build from sources: https://github.com/laurentluce/python-algorithms


class IRSystem():

    def __init__(self, docs=None, stop_words=stop_list):
        if docs is None:
            raise UserWarning('Docs should not be none')
        self._docs = docs
        self._stemmer = nltk.stem.porter.PorterStemmer()
        self._inverted_index = self._preprocess_corpus(stop_words)
        self._print_inverted_index()

    def _preprocess_corpus(self,stop_words=stop_list):
        index = {}
        for i, doc in enumerate(self._docs):
            for word in doc.split():
                token = self._stemmer.stem(word.lower())
                if token in stop_words:
                    continue
                token=rem_punctuation(token)    
                #token = self._stemmer.stem(word.lower())
                if index.get(token, -244) == -244:
                    index[token] = Node(i + 1)
                elif isinstance(index[token], Node):
                    index[token].insert(i + 1)
                else:
                    raise UserWarning('Wrong data type for posting list')
        return index

    def _print_inverted_index(self):
        print('INVERTED INDEX:\n')
        for word, tree in self._inverted_index.items():
            print('{}: {}'.format(word, [doc_id for doc_id in tree.tree_data() if doc_id != None]))
        print()

    def _get_posting_list(self, word):
        return [doc_id for doc_id in self._inverted_index[word].tree_data() if doc_id != None]

    @staticmethod
    def _parse_query(infix_tokens):
        """ Parse Query 
        Parsing done using Shunting Yard Algorithm 
        """
        precedence = {}
        precedence['NOT'] = 3
        precedence['AND'] = 2
        precedence['OR'] = 1
        precedence['('] = 0
        precedence[')'] = 0    

        output = []
        operator_stack = []

        for token in infix_tokens:
            if (token == '('):
                operator_stack.append(token)
            
            # if right bracket, pop all operators from operator stack onto output until we hit left bracket
            elif (token == ')'):
                operator = operator_stack.pop()
                while operator != '(':
                    output.append(operator)
                    operator = operator_stack.pop()
            
            # if operator, pop operators from operator stack to queue if they are of higher precedence
            elif (token in precedence):
                # if operator stack is not empty
                if (operator_stack):
                    current_operator = operator_stack[-1]
                    while (operator_stack and precedence[current_operator] > precedence[token]):
                        output.append(operator_stack.pop())
                        if (operator_stack):
                            current_operator = operator_stack[-1]
                operator_stack.append(token) # add token to stack
            else:
                output.append(token.lower())

        # while there are still operators on the stack, pop them into the queue
        while (operator_stack):
            output.append(operator_stack.pop())

        return output

    def process_query(self, query):
        # prepare query list
        query = query.replace('(', '( ')
        query = query.replace(')', ' )')
        query = query.split(' ')

        indexed_docIDs = list(range(1, len(self._docs) + 1))

        results_stack = []
        postfix_queue = collections.deque(self._parse_query(query)) # get query in postfix notation as a queue

        while postfix_queue:
            token = postfix_queue.popleft()
            result = [] # the evaluated result at each stage
            # if operand, add postings list for term to results stack
            if (token != 'AND' and token != 'OR' and token != 'NOT'):
                token = self._stemmer.stem(token) # stem the token
                # default empty list if not in dictionary
                if (token in self._inverted_index):
                    result = self._get_posting_list(token)
            
            elif (token == 'AND'):
                right_operand = results_stack.pop()
                left_operand = results_stack.pop()
                result = BooleanModel.and_operation(left_operand, right_operand)   # evaluate AND

            elif (token == 'OR'):
                right_operand = results_stack.pop()
                left_operand = results_stack.pop()
                result = BooleanModel.or_operation(left_operand, right_operand)    # evaluate OR

            elif (token == 'NOT'):
                right_operand = results_stack.pop()
                result = BooleanModel.not_operation(right_operand, indexed_docIDs) # evaluate NOT

            results_stack.append(result)                        

        # NOTE: at this point results_stack should only have one item and it is the final result
        if len(results_stack) != 1: 
            print("ERROR: Invalid Query. Please check query syntax.") # check for errors
            return None
        
        return results_stack.pop()


# In[ ]:


import argparse
import timeit
from easydict import EasyDict

import os
import pandas as pd
from nltk.tokenize import word_tokenize


inverted = dict()
docs=list()
for filename in os.listdir("E:\Info_Retreival\data"):
    with open(os.path.join("E:\Info_Retreival\data", filename), 'r') as f:
        
        text = f.read()
        docs.append(text)
        
#docs = ['hello i m a machine learning engineer haha', 
        #'hello bad world machine engineering people', 
        #'the world is a bad place',
        #'engineering a great machine that learns']


#stop_words = ['is', 'a', 'for', 'the', 'of']



def main():
    
    
    ir = IRSystem(docs, stop_words=stop_list)

    while True:
        query = input('Enter boolean query: ')

        
        results = ir.process_query(query)
        
        if results is not None:
            
            print('\nDoc IDS: ')
            print(results)
        print()
        
        
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        print('EXIT')
   


# In[ ]:





# In[ ]:





# In[ ]:




