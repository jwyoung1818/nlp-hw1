import numpy as np
from scipy import spatial
from scipy import stats
from numpy import dot
from numpy.linalg import norm
import sys
def read_corpus(filename,v, vh, vc, vch, w):
    results = []
    ch = {}
    print "initialize results"
    for i in range(len(v)):
        r = []
        # print i
        for j in range(len(vc)):
            r.append(0)
        results.append(r)
    print "initialized results"
    l = 2 * w + 1
    corpus_file = open(filename, 'r')
    corpus = []
    index = w
    count = 0
    start = []
    for line in corpus_file:
        line = line.strip()
        words = line.split(' ')
        word = '<s>'
        count += 1
        insert(corpus, index, vh, vch, results, w, ch, word)
        if len(corpus) == l - 1 and len(start) == 0:
            start = list(corpus)
        for word in words:
            insert(corpus, index, vh, vch, results, w, ch, word)
            if len(corpus) == l - 1 and len(start) == 0:
                start = list(corpus)            
            count += 1
        word = '</s>'
        count += 1
        insert(corpus, index, vh, vch, results, w, ch, word)
        if len(corpus) == l - 1 and len(start) == 0:
            start = list(corpus)
    # process start and end, seems no influence
    for i in range(w):
        vw = start[i]
        if vw in vh:
            vw_index = vh[vw]
            for j in range(i + w + 1):
                  vcw = start[j]
                  if i == j:
                      continue
                  if vcw in vch:
                      vcw_index = vch[vcw] # get the index of the word in vc
                      results[vw_index][vcw_index] += 1
    for i in range(len(corpus) - w, len(corpus)):
        vw = corpus[i]
        if vw in vh:
            vw_index = vh[vw]
            for j in range(i - w, len(corpus)):
                  vcw = corpus[j]
                  if i == j:
                      continue
                  if vcw in vch:
                      vcw_index = vch[vcw] # get the index of the word in vc
                      results[vw_index][vcw_index] += 1
    np.savetxt(filename + '.array', corpus,  fmt='%s')
    np.savetxt(filename + '.vector', results,  fmt='%d')
    return results, ch, count
def adv_read_corpus(filename,v, vh, vc, vch, w):
    results = [] 
    for i in range(len(v)):
        r = {}
        results.append(r)
    ch = {}
    l = 2 * w + 1
    corpus_file = open(filename, 'r')
    corpus = []
    index = w
    count = 0
    start = []
    for line in corpus_file:
        line = line.strip()
        words = line.split(' ')
        word = '<s>'
        count += 1
        adv_insert(corpus, index, vh, vch, results, w, ch, word)
        if len(corpus) == l - 1 and len(start) == 0:
            start = list(corpus)
        for word in words:
            adv_insert(corpus, index, vh, vch, results, w, ch, word)
            if len(corpus) == l - 1 and len(start) == 0:
                start = list(corpus)            
            count += 1
        word = '</s>'
        count += 1
        adv_insert(corpus, index, vh, vch, results, w, ch, word)
        if len(corpus) == l - 1 and len(start) == 0:
            start = list(corpus)
    # process start and end, seems no influence
    for i in range(w):
        vw = start[i]
        if vw in vh:
            vw_index = vh[vw]
            for j in range(i + w + 1):
                  vcw = start[j]
                  if i == j:
                      continue
                  if vcw in vch:
                      vcw_index = vch[vcw] # get the index of the word in vc
                      if vcw_index in results[vw_index]:
                          results[vw_index][vcw_index] += 1
                      else:
                          results[vw_index][vcw_index] = 1
    for i in range(len(corpus) - w, len(corpus)):
        vw = corpus[i]
        if vw in vh:
            vw_index = vh[vw]
            for j in range(i - w, len(corpus)):
                  vcw = corpus[j]
                  if i == j:
                      continue
                  if vcw in vch:
                      vcw_index = vch[vcw] # get the index of the word in vc
                      if vcw_index in results[vw_index]:
                          results[vw_index][vcw_index] += 1
                      else:
                          results[vw_index][vcw_index] = 1
    # np.savetxt(filename + '.array', corpus,  fmt='%s')
    # np.savetxt(filename + '.vector', results,  fmt='%d')
    return results, ch, count
    
def insert(corpus, index, vh, vch, results, w, ch, word):
    l = 2 * w + 1
    corpus.append(word)
    if word in ch:
        ch[word] += 1
    else:
        ch[word] = 1
    if len(corpus) >= l :
        # print "length is enough", len(corpus)
        vw = corpus[index]
        if vw in vh:
            vw_index = vh[vw]
            for i in range(2 * w + 1):
              vcw = corpus[i]
              if i == w :
                  continue
              if vcw in vch:
                  vcw_index = vch[vcw] # get the index of the word in vc
                  results[vw_index][vcw_index] += 1
        corpus.remove(corpus[0])
def adv_insert(corpus, index, vh, vch, results, w, ch, word):
    l = 2 * w + 1
    corpus.append(word)
    if word in ch:
        ch[word] += 1
    else:
        ch[word] = 1
    if len(corpus) >= l :
        vw = corpus[index]
        if vw in vh:
            vw_index = vh[vw]
            for i in range(2 * w + 1):
              vcw = corpus[i]
              if i == w :
                  continue
              if vcw in vch:
                  vcw_index = vch[vcw] # get the index of the word in vc
                  if vcw_index in results[vw_index]:
                      results[vw_index][vcw_index] += 1
                  else:
                      results[vw_index][vcw_index] = 1
                  #print "re", vw_index, vcw_index, results[vw_index][vcw_index]
        corpus.remove(corpus[0])

def calculatePMI(results, ch, N, v, vc, w):
    PMI = []
    results = np.array(results)
    for i in range(len(results)):
        r = []
        for j in range(len(results[i])):
            r.append(0)
        PMI.append(r)
    sum_gram = (N - 2 * w) * 2 * w + (3 * w - 1) * (w - 1)
    sum_gram = float(np.sum(results))
    j_sum = []
    for j in range(len(results[0])):
        j_sum.append(np.sum(results[:,j]))
    for i in range(len(results)):
        vector = results[i]
        x = v[i]
        if x in ch:
            p_x = float(ch[x]) / N 
        else:
            p_x = 0
        p_x = np.sum(results[i]) / sum_gram
        count = 0
        for j in range(len(vector)):
            y = vc[j]
            if y in ch:
                p_y = float(ch[y]) / N 
            else:
                p_y = 0
            p_y = j_sum[j] / sum_gram
            p_xy = float(results[i][j]) / sum_gram
            if p_xy == 0:
                pmi = 0
                count += 1
            else:
                pmi =  p_xy / (p_x * p_y)
		pmi = np.log(pmi)
		if pmi > 0:
		    PMI[i][j] = pmi
        if count == len(vector):
            print i, v[i], count, np.linalg.norm(vector)
    print "finished PMI"
    # np.savetxt('pmi', PMI, fmt='%f')
    return PMI     
def adv_calculatePMI(results, ch, N, v, vc, w):
    PMI = []
    for i in range(len(v)):
        r = {}
        PMI.append(r)
    sum_gram = (N - 2 * w) * 2 * w + (3 * w - 1) * (w - 1) 
    j_sum = []
    for j in range(len(results)):
	j_sum.append(np.sum(results[j].values()))
    sum_gram = float(np.sum(j_sum))
    # sum_gram = 0
    # for hash_fre in results:
    #     sum_gram += np.sum(hash_fre.values())
    print "sum_gram is :", sum_gram
    for i in range(len(results)):
        hash_fre = results[i]
        # print i, len(v)
        x = v[i]
        if x in ch:
            p_x = ch[x] / float(N)
        else:
            p_x = 0
        p_x = j_sum[i] / sum_gram
        count = 0
        for j in  hash_fre:
            y = vc[j]
            if y in ch:
                p_y = ch[y] / float(N)
            else:
                p_y = 0
            p_y = j_sum[j] / sum_gram
            p_xy = float(results[i][j]) / sum_gram
            if p_xy == 0 or p_x == 0 or p_y == 0:
                # print i, j , p_x, p_y, p_xy
                pmi = 0
                count += 1
            else:
                pmi =  p_xy / (p_x * p_y)
		pmi = np.log(pmi)
		if pmi > 0:
		    PMI[i][j] = pmi
    #np.savetxt('pmi', PMI, fmt='%f')
    return PMI   
def readV(filename):
    words = []
    words_hash = {}
    words_f = open(filename, 'r')
    i = 0
    for word in words_f:
        word = word.replace('\n','')
        words.append(word)
        words_hash[word] = i
        i = i + 1
        
    return words, words_hash

def getSim(filename, results, vh):
    file = open(filename, 'r')
    scores = []
    sims = []
    for line in file:
        words = line.strip().split('\t')
        # print words
        if words[0] == 'word1':
            continue;
        if len(words) == 3:
            w1 = words[0]
            w2 = words[1]
            score = float(words[2])
            #scores.append(score)
            i1 = vh[w1]
            i2 = vh[w2]
            v1 = results[i1]
            v2 = results[i2]
            # similarity = 1 - spatial.distance.cosine(results[i1], results[i2])
            similarity = 0
            if norm(v1) != 0 and norm(v2) != 0:
                similarity = dot(v1, v2) / (norm(v1) * norm(v2))
		if w1 == 'sun' and w2 == 'sunlight':
                    print w1, similarity
            	sims.append(similarity)
		scores.append(score)
    return stats.spearmanr(scores, sims)
def adv_getSim(filename, results, vh):
    file = open(filename, 'r')
    scores = []
    sims = []
    for line in file:
        words = line.strip().split('\t')
        # print words
        if words[0] == 'word1':
            continue;
        if len(words) == 3:
            w1 = words[0]
            w2 = words[1]
            score = float(words[2])
            scores.append(score)
            i1 = vh[w1]
            i2 = vh[w2]
            v1 = hash2vector(results[i1], len(vh))
            v2 = hash2vector(results[i2], len(vh))
            # similarity = 1 - spatial.distance.cosine(v1, v2)
            if norm(v1) != 0 and norm(v2) != 0:
                similarity = dot(v1, v2) / (norm(v1) * norm(v2))
            sims.append(similarity)
    return stats.spearmanr(scores, sims)

def topN(word, PMI, vc, vh, n, length):
    print "Top", n
    neighbors = [0] * n
    tops = [0] * n
    tops = np.array(tops)
    if word in vh:
        word_index = vh[word]
        v1 = hash2vector(PMI[word_index], length)
        for i in range(len(PMI)):
            if i != word_index:
                v2 = hash2vector(PMI[i], length)
                #similarity = 1 - spatial.distance.cosine(v1, v2)
                if norm(v1) != 0 and norm(v2) != 0:
                    similarity = dot(v1, v2) / (norm(v1) * norm(v2))
                if similarity > neighbors[0] :
                    neighbors[0] = similarity
                    tops[0] = i
                    tops = tops[np.argsort(neighbors)]
                    neighbors.sort()
                    # print "similarity,", similarity, vc[i]
    return neighbors, tops
def hash2vector(hash_fre, length):
    vector = []
    for i in range(length):
        vector.append(0)
    for j in hash_fre:
        vector[j] = hash_fre[j]
    return vector
def main():
    
    filename = '31190-a1-files/vocab-wordsim.txt'
    v, vh = readV(filename)
    filename = '31190-a1-files/vocab-25k.txt'
    vc, vch = readV(filename)
    filename = 'wiki-1percent.txt'
    #filename = 'test.txt'
    if sys.argv[1] == 'd':
        w = int(sys.argv[2]) 
        # ch is the count vector, N is the number of words in the corpus
        results, ch, N = read_corpus(filename, v, vh, vc, vch, w)
        PMI = calculatePMI(results, ch, N, v, vc, w)
        print "window size: ", w
        # filename = '31190-a1-files/men.txt'
        # s = getSim(filename, results, vh)
        # print "similarity: ", s
       
        # filename = '31190-a1-files/simlex-999.txt'
        # s = getSim(filename, results, vh)
        # print "similarity: ", s
        
        filename = '31190-a1-files/men.txt'
        s = getSim(filename, PMI, vh)
        print "similarity: ", s
       
        filename = '31190-a1-files/simlex-999.txt'
        s = getSim(filename, PMI, vh)
        print "similarity: ", s

    elif sys.argv[1] == 'n':        
        word = 'monster'
        ws = [1,6]
        n = 10
        for w in ws:
            print "Calculate,", n, "nearest neighborgh of '", word, "' window size is :", w
            results, ch, N = adv_read_corpus(filename, vc, vch, vc, vch, w)
            PMI = adv_calculatePMI(results, ch, N, vc, vc, w)
            nei, tops = topN(word, PMI, vc, vch, n, len(vc)) 
            print nei
            print np.array(vc)[tops]
    elif sys.argv[1] == 'p':        
        words = ['defeating', 'the', 'monsters','and', 'rescuing', 'the', 'children']
        ws = [1, 6]
	words = ['on', 'above', 'well', 'of', 'pretty']
        n = 10
        for w in ws:
            results, ch, N = adv_read_corpus(filename, vc, vch, vc, vch, w)
            PMI = adv_calculatePMI(results, ch, N, vc, vc, w)
            for word in words:
                print "Calculate,", n, "nearest neighborgh of '", word, "' window size is :", w
                nei, tops = topN(word, PMI, vc, vch, n, len(vc)) 
                print nei
                print np.array(vc)[tops]
    elif sys.argv[1] == 'm':        
        words = ['bank', 'cell', 'apple', 'apples', 'axes', 'frame', 'light', 'well']
        ws = [6]
        n = 10
	print 'multiple senses'
        for w in ws:
            results, ch, N = adv_read_corpus(filename, vc, vch, vc, vch, w)
            PMI = adv_calculatePMI(results, ch, N, vc, vc, w)
            for word in words:
                print "Calculate,", n, "nearest neighborgh of '", word, "' window size is :", w
                nei, tops = topN(word, PMI, vc, vch, n, len(vc)) 
                print nei
                print np.array(vc)[tops]
    else:
        print "d word resprentation"
        
        print "n nearest neighbor of monster"
        
        
        print "p part-of-speech"
        
        print "m multi-meaning"
    
if __name__ == '__main__':
    main()

