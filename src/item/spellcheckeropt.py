# Keep some interesting statistics
NodeCount = 0

class SpellcheckerOpt:

    def __init__(self):
        self.word_count = 0
        self.trie = TrieNode()


    def load_words(self, words):

        trie_ = self.trie
        # read list of words into a trie
        for word in words:
            self.word_count += 1
            trie_.insert(word)

        print("Read %d words" % self.word_count)


    def search(self, word, max_cost):
        '''
            The search function returns a list of all words that are less than the
            given maximum distance from the target word
        '''

        # build first row
        current_row = range(len(word) + 1)

        results = []
        trie_ = self.trie

        # recursively search each branch of the trie
        for letter in trie_.children:
            self.search_recursive(trie_.children[letter], letter, word,
                                  current_row, results, max_cost)

        return results


    def search_recursive(self, node, letter, word, previous_row, results, max_cost):
        '''
            This recursive helper is used by the search function. It assumes that
            the previous row has been filled in already.
        '''

        columns = len(word) + 1
        current_row = [previous_row[0] + 1]

        # Build one row for the letter, with a column for each letter in the target
        # word, plus one for the empty string at column 0
        for column in range(1, columns):

            insert_cost = current_row[column - 1] + 1
            delete_cost = previous_row[column] + 1

            if word[column - 1] != letter:
                replace_cost = previous_row[column - 1] + 1
            else:
                replace_cost = previous_row[column - 1]

            current_row.append(min(insert_cost, delete_cost, replace_cost))

        # if the last entry in the row indicates the optimal cost is less than the
        # maximum cost, and there is a word in this trie node, then add it.
        if current_row[-1] <= max_cost and node.word != None:
            results.append((node.word, current_row[-1]))

        # if any entries in the row are less than the maximum cost, then
        # recursively search each branch of the trie
        if min(current_row) <= max_cost:
            for letter in node.children:
                self.search_recursive(node.children[letter], letter, word,
                                      current_row, results, max_cost)


class TrieNode:
    '''
        The Trie data structure keeps a set of words, organized with one node for
        each letter. Each node has a branch for each letter that may follow it in the
        set of words.
    '''

    def __init__(self):
        self.word = None
        self.children = {}

        global NodeCount
        NodeCount += 1

    def insert(self, word):
        node = self
        for letter in word:
            if letter not in node.children:
                node.children[letter] = TrieNode()
            node = node.children[letter]

        node.word = word
