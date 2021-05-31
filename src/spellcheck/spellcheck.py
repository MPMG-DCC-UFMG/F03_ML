from .spellcheckeropt import SpellcheckerOpt
import multiprocessing


def get_ranges(num_words, n_threads):
    '''
        It gets the ranges of the words. This is done in order to the processes work
        on.
    '''

    if(n_threads == 1):
        return 0, (num_words - 1)

    total_len = num_words
    num_threads = n_threads
    lower = []
    upper = []
    step = int(total_len/num_threads)

    for k in range(num_threads):
        lower.append(0)
        upper.append(0)

    lower[0] = 0
    upper[0] = step

    i = 1
    j = 0
    while (i < num_threads):
        upper[i]  = upper[j] + step
        lower[i]  = upper[j] +  1
        if(i%2 != 0):
            upper[i] = upper[i] + 1

        i = i + 1
        j = j + 1

    upper[n_threads - 1] = num_words - 1

    return lower, upper


def run_spellchecker_thread(spellchecker, tokens, distance, it_thread,
                            lower, upper, results_threads):
    '''
        It runs the spellchecker algorithm for each token in a range.

        spellchecker (SpellcheckerOpt object): spellchecker to be used to correct
                                               the tokens.
        tokens (list): tokens that should be corrected by the spellchecker.
        distance (int): levenshtein distance to be used.
        it_thread (int): thread number.
        results_threads (dict): where the results of the thread should be stored.
    '''

    token_similar = {}

    for token in tokens[lower:upper]:
        words_list = spellchecker.search(token, distance)
        if len(words_list) > 0:
            words_list.sort(key=lambda x:(x[1], x[0]))
            token_similar[token] = words_list[0][0]

    results_threads[it_thread] = token_similar


def run_spellchecker(words_set, tokens, distance=2, n_threads=10):
    '''
        It corrects each word in the list of tokens passed.

        words_set (list): right words to be used by the spellchecker.
        tokens (list): tokens that should be corrected by the spellchecker.
        distance (int): levenshtein distance to be used.
        n_thread (int): number of threads.
    '''

    spellchecker = SpellcheckerOpt()
    spellchecker.load_words(list(words_set))

    # It defines the ranges (of the items) the threads will work on:
    thread_ranges = get_ranges(len(tokens), n_threads)
    print('Read ranges')
    print(thread_ranges)

    manager = multiprocessing.Manager()
    results_threads = manager.dict()
    jobs = []

    for i in range(n_threads):
        p = multiprocessing.Process(target=run_spellchecker_thread,
        args = (spellchecker, tokens, distance, i, thread_ranges[0][i], \
                thread_ranges[1][i], results_threads))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    token_similar = {}
    for i in range(n_threads):
        token_similar.update(results_threads[i])

    return token_similar
