from nltk.tokenize import WhitespaceTokenizer, sent_tokenize
from HTMLParser import HTMLParser
import string
import re
from mlslangdetection import MLSlangDetection

SLANG_WORDS_FILE = "data/slangdict.csv"
DOMAIN_NAMES_FILE = "data/domain_names"
OUTPUT_FILE = "data/output"
UNTRANSLATED_FILE = "untranslated"

class RuleSlangDetection():

    is_initiation_done = False
    dict = None
    slang_dict = {}
    domain_names = []
    ml_slang_parser = None

    def __init__(self, vocb):
        if(self.is_initiation_done==False):
            self.init(vocb)
            self.ml_slang_parser = MLSlangDetection(self.dict)
            self.is_initiation_done = True

    def init(self, vocb):
        # initiate slang dictionary from slangs file.
        with open(SLANG_WORDS_FILE, 'r') as slang_file:
            for line in slang_file:
                token = line.strip('\n\r').split(',', 1)
                if (self.slang_dict.get(token[0], None) is None):
                    try:
                        self.slang_dict[token[0]] = token[1]
                    except IndexError:
                        print "Error in reading from slangdict.csv: ", "line: ", line

        # init domain_names list
        with open(DOMAIN_NAMES_FILE, 'r') as domains:
            for dom in domains:
                self.domain_names.append(dom.strip('\n\r'))

        #print "length of slang_dict: ", len(slang_dict)
        #print "no of domains: ", len(domain_names)

        # init PyEnchant lib
        self.dict = vocb

    # segment paragraph into sentences and parse each sentence to remove slang words
    def parseParagraph(self, paragraph):
        if (paragraph is ''):
            return

        words = WhitespaceTokenizer().tokenize(paragraph)
        for index in xrange(0, len(words)):
            if (words[index] != ''
                and (len(words[index]) == 1
                     or (len(words[index])>0 and self.dict.check(words[index]) is False))):  # if a word is not found in vocabulary

                # handle case: HTML entity '&amp;'
                is_corrected, word = self.isHTMLEntity(words[index])
                if (is_corrected):
                    words[index] = word
                    continue

                # handle case: 'thanks,'
                is_corrected, word = self.isWordWithPunctuations(words[index])
                if (is_corrected):
                    words[index] = word
                    continue

                # handle case: '-32.06%'
                is_corrected, word = self.isNumber(words[index])
                if (is_corrected):
                    words[index] = word
                    continue

                # handle case: 'install/uninstall'
                is_corrected, word = self.isMultipleWords(words[index])
                if (is_corrected):
                    words[index] = word
                    continue
                #loopkup in slang dict
                is_corrected, word = self.isInSlangDict(words[index])
                if (is_corrected):
                    words[index] = word
                    continue

                if(len(words[index].strip(string.punctuation))>1):
                    if(index>=2):
                        is_corrected, word = self.ml_slang_parser.matchedWord(words[index], words[index-1], words[index-2])
                    elif(index==1):
                        is_corrected, word = self.ml_slang_parser.matchedWord(words[index], words[index-1], ".")
                    elif(index==0):
                        is_corrected, word = self.ml_slang_parser.matchedWord(words[index], ".", ".")

                    if(is_corrected):
                        words[index] = word
                        continue

        self.writeToOutputFile(words)



    # handle case: HTML entity '&amp;'
    def isHTMLEntity(self, word):
        if (len(word) == 1):
            return False, word

        if (word[0] == '&'):
            h = HTMLParser()
            if (h.unescape(word) != '' and self.dict.check(h.unescape(word)) == True):
                word = unicode.encode(h.unescape(word), 'utf-8')
                return True, word
            else:
                return False, word
        else:
            return False, word



    # handle case: 'thanks,'
    def isWordWithPunctuations(self, word):
        if (len(word) == 1):
            return False, word

        if (word.strip(string.punctuation) != '' and self.dict.check(word.strip(string.punctuation)) == True):
            right_stripped = word.rstrip(string.punctuation)
            stripped = word.strip(string.punctuation)
            suffix_diff = len(word) - len(right_stripped)

            if (suffix_diff > 0):
                stripped = stripped + word[len(right_stripped)]

            return True, stripped
        else:
            return False, word



    # handle case: '-32.06%'
    def isNumber(self, word):
        if (len(word) == 1):
            return False, word

        try:
            number = float(word.strip(string.punctuation))
            is_number = True
        except ValueError:
            is_number = False
            return False, word

        if (is_number):
            right_stripped = word.rstrip(string.punctuation)
            left_stripped = word.lstrip(string.punctuation)
            stripped = left_stripped.strip(string.punctuation)
            prefix_diff = len(word) - len(left_stripped)
            suffix_diff = len(word) - len(right_stripped)

            if (prefix_diff > 0):
                stripped = word[prefix_diff - 1] + stripped
            if (suffix_diff > 0):
                stripped = stripped + word[len(right_stripped)]

            word = stripped
            return True, word
        else:
            return False, word

    def isMultipleWords(self, word):
        if (len(word) == 1):
            return False, word

        tokens = re.findall(r"[\w']+", word.strip(string.punctuation))
        is_string_updated = False

        if (len(tokens) > 0):
            parsed_word = ''
            beg = 0
            for idx in xrange(0, len(tokens)):
                # handle case: 'thanks,'
                is_corrected, updated_word = self.isWordWithPunctuations(tokens[idx])
                if (is_corrected):
                    is_string_updated = True
                    parsed_word = parsed_word + updated_word
                    parsed_word, beg = self.modifyParsedString(word, tokens, parsed_word, updated_word, idx, beg)
                    continue

                # handle case: '-32.06%'
                is_corrected, updated_word = self.isNumber(tokens[idx])
                if (is_corrected):
                    is_string_updated = True
                    parsed_word = parsed_word + updated_word
                    parsed_word, beg = self.modifyParsedString(word, tokens, parsed_word, updated_word, idx, beg)
                    continue

                # handle case: HTML entity '&amp;'
                is_corrected, updated_word = self.isHTMLEntity(tokens[idx])
                if (is_corrected):
                    is_string_updated = True
                    parsed_word = parsed_word + updated_word
                    parsed_word, beg = self.modifyParsedString(word, tokens, parsed_word, updated_word, idx, beg)
                    continue

                # look into table
                is_corrected, updated_word = self.isInSlangDict(tokens[idx])
                if (is_corrected):
                    is_string_updated = True
                    parsed_word = parsed_word + updated_word
                    if (idx < (len(tokens) - 1)):
                        non_alpha_numeric_chr_index = word.find(tokens[idx], beg) + len(tokens[idx])
                        if (word[non_alpha_numeric_chr_index] == '.' and len(updated_word) > 1 and tokens[idx + 1] not in self.domain_names):
                            parsed_word = parsed_word + word[non_alpha_numeric_chr_index] + ' '
                        elif (word[non_alpha_numeric_chr_index] == '?'):
                            parsed_word = parsed_word + word[non_alpha_numeric_chr_index] + ' '
                        else:
                            parsed_word = parsed_word + word[non_alpha_numeric_chr_index]
                        beg = non_alpha_numeric_chr_index
                    else:
                        if (len(tokens[idx]) == 1 and len(word) > len(tokens[idx])):  # u...emoji
                            try:
                                if (word[word.find(tokens[idx], beg) + 1] == '.' or word[
                                        word.find(tokens[idx], beg) + 1] == '?'):
                                    parsed_word = parsed_word + word[word.find(tokens[idx], beg) + 1] + ' '
                                    beg = word.find(tokens[idx], beg) + 1
                                else:
                                    parsed_word = parsed_word + word[word.find(tokens[idx], beg) + 1]
                                    word.find(tokens[idx], beg) + 1
                            except IndexError:
                                pass

                        suffix_diff = len(word) - len(word.rstrip(string.punctuation))
                        if (suffix_diff > 0):
                            parsed_word = parsed_word + word[len(word.rstrip(string.punctuation))]
                    continue

                parsed_word = parsed_word + tokens[idx]
                if (idx < (len(tokens) - 1)):
                    non_alpha_numeric_chr_index = word.find(tokens[idx], beg) + len(tokens[idx])
                    if (word[non_alpha_numeric_chr_index] == '.' and len(tokens[idx]) > 1 and tokens[idx + 1] not in self.domain_names):
                        parsed_word = parsed_word + word[non_alpha_numeric_chr_index] + ' '
                    elif (word[non_alpha_numeric_chr_index] == '?'):
                        parsed_word = parsed_word + word[non_alpha_numeric_chr_index] + ' '
                    else:
                        parsed_word = parsed_word + word[non_alpha_numeric_chr_index]
                    beg = non_alpha_numeric_chr_index
                else:
                    suffix_diff = len(word) - len(word.rstrip(string.punctuation))
                    if (suffix_diff > 0):
                        parsed_word = parsed_word + word[len(word.rstrip(string.punctuation))]

            if (is_string_updated):
                return True, parsed_word
            else:
                return False, word
        else:
            return False, word




    def modifyParsedString(self, word, tokens, parsed_word, updated_word, idx, beg):
        if (idx < (len(tokens) - 1)):
            non_alpha_numeric_chr_index = word.find(tokens[idx], beg) + len(tokens[idx])
            if (word[non_alpha_numeric_chr_index] == '.' and len(updated_word) > 1 and tokens[idx + 1] not in self.domain_names):
                parsed_word = parsed_word + word[non_alpha_numeric_chr_index] + ' '
            elif (word[non_alpha_numeric_chr_index] == '?'):
                parsed_word = parsed_word + word[non_alpha_numeric_chr_index] + ' '
            else:
                parsed_word = parsed_word + word[non_alpha_numeric_chr_index]
            beg = non_alpha_numeric_chr_index
        else:
            suffix_diff = len(word) - len(word.rstrip(string.punctuation))
            if (suffix_diff > 0):
                parsed_word = parsed_word + word[len(word.rstrip(string.punctuation))]

        return parsed_word, beg

    # look into table
    def isInSlangDict(self, word):
        if (self.slang_dict.get(word.strip(string.punctuation).lower(), None) != None):
            return True, self.slang_dict.get(word.strip(string.punctuation).lower(), None)
        elif (self.slang_dict.get(word.lower(), None) != None):
            return True, self.slang_dict.get(word, None)
        else:
            return False, word

    def writeToOutputFile(self, words):
        paragraph = ' '.join(words)

        try:
            sentences = sent_tokenize(paragraph)
        except Exception:
            try:
                sentences = sent_tokenize(paragraph.decode('utf-8'))
            except Exception:
                sentences = sent_tokenize(paragraph.decode('latin-1'))

        output_file = open(OUTPUT_FILE, 'a')
        for sent in sentences:
            try:
                output_file.write(sent + "\n")
            except Exception:
                output_file.write(sent.encode('ascii', 'ignore'))
        output_file.write("\n@@@@@\n")
        output_file.close()
