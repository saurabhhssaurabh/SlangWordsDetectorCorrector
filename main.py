import enchant
from ruleslangdetection import RuleSlangDetection


SLANG_WORDS_FILE = "data/slangdict.csv"
TEST_PARAGRAPHS_FILE = "data/test"
TEST_SLANG_WORDS_FILE = "data/testdata"

PARAGRAPH_SEPARATION_TOKEN = "@@@@@"

def main():
    slang_parser = RuleSlangDetection(enchant.Dict("en_US"))
    para = ''
    input_file = open(TEST_SLANG_WORDS_FILE, 'r')
    for line in input_file:
        if(line.find(PARAGRAPH_SEPARATION_TOKEN)>=0):
            slang_parser.parseParagraph(para)
            para = ''
        else:
            para = para + line

    if(para is not ''):
        slang_parser.parseParagraph(para)

    input_file.close()

if __name__ == '__main__':
    main()