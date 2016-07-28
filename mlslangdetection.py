import collections
import enchant
import metaphone
import string
import sys

UNIGRAM_FILE = "data/unigram"
BIGRAM_FILE = "data/bigram"
TRIGRAM_FILE = "data/trigram"
WORDS_TO_METAPHONE = "data/words_to_metaphone.csv"
SPLIT_TOKEN = '@$@$'
ERROR_MODEL_PROB_ONE_EDIT_DISTANCE = 0.6
ERROR_MODEL_PROB_TWO_EDIT_DISTANCE = 0.4
ERROR_MODEL_PROB_PHONEMIC_ONE_DISTANCE = 0.6

class MLSlangDetection():

    is_dict_set = False

    unigram_probs = {}
    bigram_probs = {}
    trigram_probs = {}
    metaphone_to_words = {}
    dict = None
    delimiters_list = ['!', '.', ',']

    def __init__(self, dict):
        #print "__init__(): "
        if(MLSlangDetection.is_dict_set==False):
            self.init_dicts(dict)
            MLSlangDetection.is_dict_set = True
        #print "__init__(): leaving"


    def init_dicts(self, dict):
        #print "init_dicts(): "
        self.dict = dict
        self.loadUnigramProbs()
        self.loadBigramProbs()
        self.loadTrigramProbs()
        self.loadMetaphones()
        #print "init_dicts(): leaving"

    #give most probable  word of 'current_token'
    def matchedWord(self, current_token, prev_one, prev_two):
        token_list = self.parseWords(current_token, prev_one, prev_two)
        self.token = token_list[0]
        self.token_prev_one = token_list[1]
        self.token_prev_two = token_list[2]

        words_edit_distance_one = self.wordsOneDistance(self.token)
        words_edit_distance_two = self.wordsTwoDistance(self.token)
        phonemic_tuple = self.computePhonemic(self.token)
        word_phonemic = phonemic_tuple[0]
        phonemics_edit_distance_one = self.wordsOneDistance(word_phonemic, False)
        words_phonemic_distance_one = self.wordsFromPhonemic(phonemics_edit_distance_one)

        max_prob = sys.float_info.min
        max_prob_word = None

        for word in words_edit_distance_one:
            prob = self.computeProbability(word, self.token_prev_one, self.token_prev_two, ERROR_MODEL_PROB_ONE_EDIT_DISTANCE)
            if(prob>max_prob):
                max_prob = prob
                max_prob_word = word

        for word in words_edit_distance_two:
            prob = self.computeProbability(word, self.token_prev_one, self.token_prev_two, ERROR_MODEL_PROB_TWO_EDIT_DISTANCE)
            if(prob>max_prob):
                max_prob = prob
                max_prob_word = word

        for word in words_phonemic_distance_one:
            prob = self.computeProbability(word, self.token_prev_one, self.token_prev_two, ERROR_MODEL_PROB_PHONEMIC_ONE_DISTANCE)
            if(prob>max_prob):
                max_prob = prob
                max_prob_word = word

        if(max_prob_word!=None and max_prob>0.2):
            return True, max_prob_word
        else:
            return False, current_token


    #compute word(i-2), word(i-1), word(i) to perform statistical analysis
    def parseWords(self,current_token, prev_one, prev_two):
        token_list = []
        token_list.append(current_token.strip(string.punctuation))

        if(len(current_token)-len(current_token.lstrip(string.punctuation))>0):
            delimiter = current_token[current_token.find(token_list[0])-1]
            if(delimiter in self.delimiters_list):
                token_list.append(delimiter)

        prev_one_stripped = prev_one.strip(string.punctuation)
        if(len(prev_one)-len(prev_one.rstrip(string.punctuation))>0):
            delimiter = prev_one[prev_one.find(prev_one_stripped)+len(prev_one_stripped)]
            if(delimiter in self.delimiters_list):
                token_list.append(delimiter)

        if(len(token_list)==3):
            return token_list

        token_list.append(prev_one_stripped)

        if (len(token_list) == 3):
            return token_list

        prev_two_stripped = prev_two.strip(string.punctuation)
        if(len(prev_two)-len(prev_two.rstrip(string.punctuation))>0):
            delimiter = prev_two[prev_two.find(prev_two_stripped)+len(prev_two_stripped)]
            if(delimiter in self.delimiters_list):
                token_list.append(delimiter)

        if(len(token_list)==3):
            return token_list

        token_list.append(prev_two_stripped)
        return token_list

    #load unigram probabilities from file
    def loadUnigramProbs(self):

        with open(UNIGRAM_FILE, 'r') as unigram_file:
            for line in unigram_file:
                tokens = line.strip('\n\r').split(SPLIT_TOKEN)
                if(len(tokens)==2):
                    MLSlangDetection.unigram_probs[tokens[0]] = float(tokens[1])
                else:
                    MLSlangDetection.unigram_probs[SPLIT_TOKEN] = float(tokens[len(tokens)-1])


    #load bigram probabilities from file
    def loadBigramProbs(self):

        with open(BIGRAM_FILE, 'r') as bigram_file:
            for line in bigram_file:
                tokens = line.strip('\n\r').split(SPLIT_TOKEN)
                if(len(tokens)==3):
                    MLSlangDetection.bigram_probs[(tokens[0], tokens[1])] = float(tokens[2])
                else:
                    if(tokens[0]=='' and tokens[len(tokens)-2]==''):
                        MLSlangDetection.bigram_probs[(SPLIT_TOKEN, SPLIT_TOKEN)] = float(tokens[len(tokens)-1])
                    elif(tokens[0]==''):
                        MLSlangDetection.bigram_probs[(SPLIT_TOKEN, tokens[len(tokens)-2])] = float(tokens[len(tokens)-1])
                    else:
                        MLSlangDetection.bigram_probs[(tokens[0], SPLIT_TOKEN)] = float(tokens[len(tokens)-1])


    #load trigram probabilities from file
    def loadTrigramProbs(self):

        with open(TRIGRAM_FILE, 'r') as trigram_file:
            for line in trigram_file:
                tokens = line.strip('\n\r').split(SPLIT_TOKEN)
                if(len(tokens)==4):
                    MLSlangDetection.trigram_probs[(tokens[0], tokens[1], tokens[2])] = float(tokens[3])
                else:
                    if(len(tokens)==5):
                        if(tokens[0]==''):
                            MLSlangDetection.trigram_probs[(SPLIT_TOKEN, tokens[2], tokens[3])] = float(tokens[4])
                        elif(tokens[1]==''):
                            MLSlangDetection.trigram_probs[(tokens[0], SPLIT_TOKEN, tokens[3])] = float(tokens[4])
                        else:
                            MLSlangDetection.trigram_probs[(tokens[0], tokens[1], SPLIT_TOKEN)] = float(tokens[4])
                    elif(len(tokens)==6):
                        if(tokens[0]!=''):
                            MLSlangDetection.trigram_probs[(tokens[0], SPLIT_TOKEN, SPLIT_TOKEN)] = float(tokens[5])
                        elif(tokens[len(tokens)-2]!=''):
                            MLSlangDetection.trigram_probs[(SPLIT_TOKEN, SPLIT_TOKEN, tokens[len(tokens)-2])] = float(tokens[5])
                        else:
                            MLSlangDetection.trigram_probs[(SPLIT_TOKEN, tokens[2], SPLIT_TOKEN)] = float(tokens[5])
                    else:
                        print line
                        MLSlangDetection.trigram_probs[(SPLIT_TOKEN, SPLIT_TOKEN, SPLIT_TOKEN)] = float(tokens[6])



    #load phonemics of words from file
    def loadMetaphones(self):

        with open(WORDS_TO_METAPHONE, 'r') as metaphones_file:
            for line in metaphones_file:
                tokens = line.strip('\n\r').split(',')
                if(self.metaphone_to_words.get(tokens[1], None)==None):
                    self.metaphone_to_words[tokens[1]] = list(tokens[0])
                else:
                    self.metaphone_to_words[tokens[1]].append(tokens[0])


    #compute words having one levenshtein distance from 'word'
    def wordsOneDistance(self, word, checkInDict=True):

        alphabet = 'abcdefghijklmnopqrstuvwxyz'

        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [a + b[1:] for a, b in splits if b]
        transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
        replaces = [a + c + b[1:] for a, b in splits for c in alphabet if b]
        inserts = [a + c + b for a, b in splits for c in alphabet]

        words_set = set()

        if(checkInDict):
            for w in deletes:
                try:
                    if len(w)>0 and self.dict.check(w)==True:
                        words_set.add(w)
                except enchant.errors.Error:
                    pass

            for w in transposes:
                try:
                    if len(w)>0 and self.dict.check(w)==True:
                        words_set.add(w)
                except enchant.errors.Error:
                    pass

            for w in replaces:
                try:
                    if len(w)>0 and self.dict.check(w)==True:
                        words_set.add(w)
                except enchant.errors.Error:
                    pass

            for w in inserts:
                try:
                    if len(w)>0 and self.dict.check(w)==True:
                        words_set.add(w)
                except enchant.errors.Error:
                    pass
        else:
            words_set = set(deletes + transposes + replaces + inserts)

        return words_set


    #compute words having two levenshtein distance from 'word'
    def wordsTwoDistance(self, word):
        words_set =  set(e2 for e1 in self.wordsOneDistance(word) for e2 in self.wordsOneDistance(e1) if (len(e2)>0 and self.dict.check(e2)))
        return words_set

    #compute phonemic of word using double metaphophonemic algorithm
    def computePhonemic(self, word):
        try:
            metaphone_tuples = metaphone.dm(unicode(word))
        except Exception:
            try:
                metaphone_tuples = metaphone.dm(word.decode('utf-8', 'ignore'))
            except Exception:
                metaphone_tuples = metaphone.dm(word.decode('latin-1', 'ignore'))
        return metaphone_tuples

    #get words from phonemics
    def wordsFromPhonemic(self, phonemics_list):
        words_set = set()

        for phonemic in phonemics_list:
            list_ = self.metaphone_to_words.get(phonemic, None)
            if(list_!=None):
                for word in list_:
                    words_set.add(word)

        return words_set

    #compute probability P(w/c)*P(c)
    def computeProbability(self, token, prev_one, prev_two, error_model_probability):
        prob = 0.0
        tri_prob = self.trigram_probs.get((prev_two, prev_one, token), None)
        if(tri_prob!=None):
            prob = tri_prob*0.7
        bi_prob = self.bigram_probs.get((prev_one, token), None)
        if(bi_prob!=None):
            prob = prob + bi_prob*0.3
        #uni_prob = self.unigram_probs.get((token), None)
        #if(uni_prob!=None):
        #    prob = prob + uni_prob*0.1
        return prob*error_model_probability
