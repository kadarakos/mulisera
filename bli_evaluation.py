import numpy as np
import argparse


def load_vectors(vectors_file):
    '''
    Load the word vectors into memory.

    Assumes the file has this format:
    <TOKEN> <DIM1> ... <DIM300>
    '''
    tokens = np.loadtxt(open(vectors_file), usecols=0, dtype=np.str)
    values = np.loadtxt(open(vectors_file), usecols=range(1, 300+1))
    
    en_vectors = dict()
    de_vectors = dict()

    for x,y in zip(tokens, values):
        if x.startswith('en'):
            en_vectors[x] = y
        elif x.startswith('de'):
            de_vectors[x] = y

    return en_vectors, de_vectors


def load_gold_data(gold_file):
    '''
    Load the gold standard evaluation data into memory.

    Assumes the file has this format:
    WORD1 WORD2 POS
    '''
    gold = np.loadtxt(open(gold_file), delimiter='\t', dtype=np.str)

    # Create separate dictionaries for each POS
    adjectives = dict()
    nouns = dict()
    verbs = dict()

    # Populate those dictionaries
    for x in gold:
        src = x[0]
        tgt = x[1]
        pos = x[2]
        if pos == 'A':
            adjectives[src] = tgt
        if pos == 'V':
            verbs[src] = tgt
        if pos == 'N':
            nouns[src] = tgt

    return adjectives, nouns, verbs


def e2d(en_vectors, de_vectors, gold, npts=None, n=1, return_ranks=False):
    npts = len(gold.keys())

    ranks = []
    top1 = []

    de_matrix = np.array([x for x in de_vectors.values()])
    de_tokens = [x for x in de_vectors.keys()]
    skipped_en = 0
    skipped_de = 0
    for token in gold.keys():

        #  Get query vector
        try:
            en = en_vectors['en_'+token]
        except:
            skipped_en += 1
            continue

        #  Get the target vector
        try:
            de = de_tokens.index('de_'+gold[token])
        except:
            pass
        try:
            #  Backoff for when our model vocabulary is lowercased
            de = de_tokens.index('de_'+gold[token].lower())
        except:
            skipped_de += 1
            continue

        # Compute scores
        sim = np.dot(en, de_matrix.T).flatten()
        inds = np.argsort(sim)[::-1]

        # Score
        rank = np.where(inds == de)[0][0]
        ranks.append(rank)
        top1.append(inds[0])

    ranks = np.array(ranks)
    
    # Compute metrics
    p1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    p5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    p10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medp = np.floor(np.median(ranks)) + 1
    mrr = np.mean([1/(x+1) for x in ranks])  # + 1 to avoid division by zero

    return {'p1':p1, 'p5':p5, 'p10':p10, 'medp':medp, 'mrr':mrr,
            'skipped':skipped_en + skipped_de, 'skip_en':skipped_en,
            'skip_de':skipped_de, 'total': len(gold.keys())}


def bli_evaluate(vectors_file, gold_file):
    en_vectors, de_vectors = load_vectors(vectors_file)
    adjectives, nouns, verbs = load_gold_data(gold_file)
    
    a = e2d(en_vectors, de_vectors, adjectives)
    n = e2d(en_vectors, de_vectors, nouns)
    v = e2d(en_vectors, de_vectors, verbs)

    print('{}/{} Nouns: P@1 {:.2f} P@10 {:.2f} MRR {:.2f}'.format(n['total']-n['skipped'], n['total'], n['p1'], n['p10'], n['mrr']))
    print('{}/{} Verbs: P@1 {:.2f} P@10 {:.2f} MRR {:.2f}'.format(v['total']-v['skipped'], v['total'], v['p1'], v['p10'], v['mrr']))
    print('{}/{} Adjectives: P@1 {:.2f} P@10 {:.2f} MRR {:.2f}'.format(a['total']-a['skipped'], a['total'], a['p1'], a['p10'], a['mrr']))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", type=str, required=True)
    parser.add_argument("--gold", type=str, required=True)
    args = parser.parse_args()
    bli_evaluate(args.vectors, args.gold)
