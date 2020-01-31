import os
import json
import nltk
import numpy as np
import config as c
import get_data
from tqdm import tqdm


def data_from_json(json_path):
    with open(json_path, encoding='utf8') as data_file:
        data = json.load(data_file)
    return data


def write_to_file(out_file, line):
    out_file.write(line + '\n')


def tokenize(sequence):
    tokens = [token.replace("``", '"').replace("''", '"').lower()
              for token in nltk.word_tokenize(sequence)]
    return tokens


def get_char_word_loc_mapping(context, context_tokens):
    acc = ''
    current_token_idx = 0
    mapping = {}

    # step through original characters
    for char_idx, char in enumerate(context):
        if char != u' ' and char != u'\n':  # if it's not a space:
            acc += char  # add to accumulator
            context_token = str(
                context_tokens[current_token_idx])  # current word token
            if acc == context_token:  # if the accumulator now matches the current word token
                # char loc of the start of this word
                syn_start = char_idx - len(acc) + 1
                for char_loc in range(syn_start, char_idx+1):
                    mapping[char_loc] = (
                        acc, current_token_idx)  # add to mapping
                acc = ''  # reset accumulator
                current_token_idx += 1

    if current_token_idx != len(context_tokens):
        return None
    else:
        return mapping


def preprocess_and_write(json_path, tier):
    dataset = data_from_json(json_path)
    num_exs = 0
    num_mappingprob, num_tokenprob, num_spanalingnprob, num_noanswer = 0, 0, 0, 0
    examples = []

    for articles_id in tqdm(range(len(dataset['data'])), desc='Preprocessing {}'.format(tier)):
        article_paragraphs = dataset['data'][articles_id]['paragraphs']
        for pid in range(len(article_paragraphs)):

            # context processing
            context = str(article_paragraphs[pid]['context'])
            # BidAF suggestions
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')
            context = context.lower()
            context_tokens = tokenize(context)

            # charloc2wordloc maps the character location (int) of a context token to a pair giving (word (string), word loc (int)) of that token
            charloc2wordloc = get_char_word_loc_mapping(
                context, context_tokens)

            # qas processing
            # multiple question and answers
            qas = article_paragraphs[pid]['qas']

            if charloc2wordloc is None:
                num_mappingprob += len(qas)
                continue  # skipping this context

            for qn in qas:
                question = str(qn['question'])
                question_tokens = tokenize(question)
                try:
                    ans_text = str(qn['answers'][0]['text']).lower()
                except IndexError:
                    'no answer found'
                    num_noanswer += 1
                    continue
                ans_start_charloc = qn['answers'][0]['answer_start']
                ans_end_charloc = ans_start_charloc + len(ans_text)

                if context[ans_start_charloc:ans_end_charloc] != ans_text:
                    num_spanalingnprob += 1
                    continue

                ans_start_wordloc = charloc2wordloc[ans_start_charloc][1]
                ans_end_wordloc = charloc2wordloc[ans_end_charloc-1][1]

                assert ans_start_wordloc <= ans_end_wordloc

                ans_tokens = context_tokens[ans_start_wordloc:ans_end_wordloc+1]
                if "".join(ans_tokens) != "".join(ans_text.split()):
                    num_tokenprob += 1
                    continue

                examples.append((' '.join(context_tokens), ' '.join(question_tokens), ' '.join(
                    ans_tokens), ' '.join([str(ans_start_wordloc), str(ans_end_wordloc)])))
                num_exs += 1

    print("Number of (context, question, answer) triples discarded due to char -> token mapping problems: ", num_mappingprob)
    print("Number of (context, question, answer) triples discarded because character-based answer span is unaligned with tokenization: ", num_tokenprob)
    print("Number of (context, question, answer) triples discarded due to no answer: ", num_noanswer)
    print("Number of (context, question, answer) triples discarded due character span alignment problems (usually Unicode problems): ", num_spanalingnprob)
    print("Processed %i examples of total %i\n" % (
        num_exs, num_exs + num_mappingprob + num_tokenprob + num_spanalingnprob + num_noanswer))

    indices = list(range(len(examples)))
    np.random.shuffle(indices)

    with open(os.path.join(c.data_dir, tier + '.context'), 'w', encoding='utf-8') as context_file,  \
            open(os.path.join(c.data_dir, tier + '.question'), 'w', encoding='utf-8') as question_file,\
            open(os.path.join(c.data_dir, tier + '.answer'), 'w', encoding='utf8') as ans_text_file, \
            open(os.path.join(c.data_dir, tier + '.span'), 'w', encoding='utf8') as span_file:

        for i in indices:
            (context, question, answer, answer_span) = examples[i]
            # write tokenized data to file
            write_to_file(context_file, context)
            write_to_file(question_file, question)
            write_to_file(ans_text_file, answer)
            write_to_file(span_file, answer_span)


if __name__ == "__main__":
    train_path = os.path.join(c.data_dir, c.train_filename)
    dev_path = os.path.join(c.data_dir, c.dev_filename)

    if not os.path.exists(train_path):
        get_data.main()
    preprocess_and_write(train_path, 'train')
    preprocess_and_write(dev_path, 'dev')
