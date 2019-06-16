from collections import Counter
import sys
import os
from whoosh.fields import *
from whoosh.index import create_in
from whoosh.index import open_dir
from whoosh.qparser import QueryParser, OrGroup
from whoosh.query import Term, And, Phrase, Or
from whoosh import scoring
from whoosh.collectors import TimeLimitCollector, TimeLimit


threshold_info = {
    "yelp": 15,
    "amazon": 5.5,
    "caption": 5,
    "GYAFC_EM": 5,
    "GYAFC_FR": 5,
}

num_words_info = {
    "yelp": 7000,
    "amazon": 10000,
    "caption": 3000,
    "GYAFC_EM": 10000,
    "GYAFC_FR": 10000,
}

style_gap = '#####'


def get_style_ngrams(src_path_prefix, threshold=10, num_words=10000, save_path_prefix=None, dataset=None):
    path0, path1 = src_path_prefix + '.0', src_path_prefix + '.1'

    def get_count_dict(path):
        c = Counter()
        with open(path) as f:
            for line in f:
                words = line.strip().split(' ')
                n_grams = []
                for n in range(1, 5):
                    for l in range(0, len(words) - n + 1):
                        n_grams.append(' '.join(words[l:l + n]))
                c.update(n_grams)
        return c

    c0 = get_count_dict(path0)
    c1 = get_count_dict(path1)
    words = set(c0)

    tfidf0 = {}
    tfidf1 = {}
    for w in words:
        tfidf0[w] = (c0[w] + 1.0) / (c1[w] + 1.0)
        tfidf1[w] = 1.0 / tfidf0[w]

    tfidf0 = sorted(tfidf0.iteritems(), key=lambda (k,v): (v,k), reverse=True)
    tfidf1 = sorted(tfidf1.iteritems(), key=lambda (k,v): (v,k), reverse=True)

    if dataset:
        if dataset in threshold_info:
            threshold = threshold_info[dataset]
        if dataset in num_words_info:
            num_words = num_words_info[dataset]

    tfidf0 = filter(lambda (k, v): v >= threshold, tfidf0)
    tfidf1 = filter(lambda (k, v): v >= threshold, tfidf1)

    tfidf0 = tfidf0[:num_words]
    tfidf1 = tfidf1[:num_words]

    if save_path_prefix:
        with open(save_path_prefix + '.0', 'w') as f:
            for k, v in tfidf0:
                f.write('%s\t%s\n' % (k, v))
        with open(save_path_prefix + '.1', 'w') as f:
            for k, v in tfidf1:
                f.write('%s\t%s\n' % (k, v))

    return (dict(tfidf0), dict(tfidf1))


def process_data(src_path, word_dict, dst_path):
    with open(src_path) as f, open(dst_path, 'w') as fw:
        for line in f:
            words = line.strip().split(' ')
            content = ''
            style = ''
            style_dict = []
            for i, w in enumerate(words):
                for n in range(4, 0, -1):
                    if i + n > len(words):
                        continue
                    n_gram = ' '.join(words[i:i + n])
                    # find a longest n_gram
                    if word_dict.get(n_gram, False) and (style_dict == [] or i + n - 1 > style_dict[-1]):
                        style += ' '.join(words[i:i + n]) + ' '
                        style_dict.append(i)
                        style_dict.append(i + n - 1)
                        break

            # merge n_grams like: [0, 3, 4, 6] => [0, 6]
            style_dict_merge = []
            i = 0
            while i < len(style_dict):
                start = style_dict[i]
                end = style_dict[i + 1]
                n = 2
                while i + n < len(style_dict):
                    if style_dict[i + n] <= end:
                        if style_dict[i + n + 1] > end:
                            end = style_dict[i + n + 1]
                    elif style_dict[i + n] == end + 1:
                        end = style_dict[i + n + 1]
                    else:
                        break
                    n += 2
                style_dict_merge.append(start)
                style_dict_merge.append(end)
                i = i + n

            start = 0
            style1 = ''
            for i in range(0, len(style_dict_merge), 2):
                style1 += ' '.join(words[style_dict_merge[i]:style_dict_merge[i + 1] + 1]) + style_gap
            if len(style_dict) > 0 and style_dict[0] == 0:
                content = '<styleNgrams> '
            for i in range(0, len(style_dict), 2):
                if start < style_dict[i]:
                    content += ' '.join(words[start:style_dict[i]]) + ' '
                    content += '<styleNgrams> '
                start = style_dict[i + 1] + 1
            if start < len(words):
                content += ' '.join(words[start:len(words)]) + ' '

            style = style.strip()
            content = content.strip()
            contents = content.strip().split(' ')
            if len(contents) < 5 and 'train' in src_path:
                continue
            if style != '':
                fw.write(content + '\t' + style1[:-len(style_gap)] + '\n')
            else:
                if 'train' not in src_path:
                    fw.write(content + '\t' + 'SELF' + '\n')


def cal_sim(train_data_path, test_data_path, dst_result_path=None, save_n_best_search=1):
    schema = Schema(context=TEXT(stored=True), response=STORED, post=TEXT(stored=True))
    index_i = re.findall('\d', train_data_path)[0]

    index_path = "../tmp/ix_index/" + index_i
    if not os.path.exists(index_path):
        os.makedirs(index_path)

    ix = create_in(index_path, schema)
    writer = ix.writer()

    def get_cpr(line):
        lines = line.lower().strip().split('\t')
        context = ''
        post = lines[0]
        response = lines[1]
        return context.strip().decode('utf-8'), response.decode('utf-8'), post.decode('utf-8')

    def load_train_data(file_name, writer):
        f = open(file_name)
        for line in f:
            context, response, post = get_cpr(line)
            if context != '':
                writer.add_document(context=context, response=response, post=post)
            else:
                writer.add_document(response=response, post=post)
        writer.commit()

    def get_query(line, ix):
        lines = line.strip().split('\t')
        post = lines[0].decode('utf-8')
        q2 = QueryParser("post", ix.schema).parse(post)
        terms = list(q2.all_terms())
        query = Or([Term(*x) for x in terms])
        return query

    load_train_data(train_data_path, writer)

    f = open(test_data_path, 'r')
    fw_search = open(dst_result_path, 'w')
    with ix.searcher(weighting=scoring.TF_IDF()) as searcher:
        c = searcher.collector(limit=10)
        tlc = TimeLimitCollector(c, timelimit=10.0)
        for line in f:
            try:
                query = get_query(line, ix)
                searcher.search_with_collector(query, tlc)
                results = tlc.results()
                for i in range(min(len(results), save_n_best_search)):
                    fw_search.write(
                        line.strip() + '\t' + str(results[i]["post"]) + '\t' + str(results[i]["response"]) + '\n')
            except Exception as e:
                print('TimeLimit, ignore it!')
                print(line)
    fw_search.close()


def get_final_result(search_result_path, final_result_path, orignal_data_path):
    f = open(search_result_path, 'r')
    fw = open(final_result_path, 'w')
    tmp = ''

    def write_sen(sens0, style1):
        tmp = []
        num = 0
        for i in sens0:
            if i.lower() == '<styleNgrams>'.lower():
                if num < len(style1):
                    tmp.append(style1[num])
                    num += 1
                else:
                    pass
            else:
                tmp.append(i)

        if tmp[-1] in ".,?!();'":
            tmp2 = tmp[:-1]
            punc = tmp[-1:]
        else:
            tmp2 = tmp[:]
            punc = []

        while num < len(style1):
            tmp2.append(style1[num])
            num += 1
        return ' '.join(tmp2 + punc)

    rule_dict = {}
    for line in f:
        lines = line.strip().split('\t')
        # if '.' in lines[3]:
        #     print lines
        #     continue
        if tmp != lines[0]:
            tmp = lines[0]
            style1 = lines[1].split(style_gap)
            style2 = lines[3].split(style_gap)
            words = lines[0].split(' ')
            if style1[0] == 'SELF':
                sen1 = tmp
            else:
                sen1 = write_sen(words, style1)
            sen2 = write_sen(words, style2)
            rule_dict[sen1] = sen2
    f.close()
    f = open(orignal_data_path, 'r')
    for line in f:
        line = line.strip()
        if rule_dict.get(line) is not None:
            fw.write(line + '\t' + rule_dict.get(line) + '\n')
        else:
            fw.write(line + '\t' + line + '\n')
    fw.close()


def main():
    # dataset = "yelp"
    dataset = "GYAFC"

    src_train_path_prefix = '../data/%s/train' % dataset
    src_dev_path_prefix = '../data/%s/dev' % dataset
    src_test_path_prefix = '../data/%s/test' % dataset

    template_data_path = '../data/%s/template_result/' % dataset
    if not os.path.exists(template_data_path):
        os.makedirs(template_data_path)
    save_path_prefix = template_data_path + 'style_n_grams'
    replace_path_prefix = template_data_path + 'replace_result'
    search_path_prefix = template_data_path + 'search_result'
    # final_path_prefix = template_data_path + 'final_result'
    final_path_prefix = '../data/%s/tsf_template/' % dataset

    print("Build stylistic n_gram corpus...")
    n_gram_dicts = get_style_ngrams(src_train_path_prefix,
                                    save_path_prefix=save_path_prefix,
                                    dataset=dataset)
    print("Emotional saved at: %s, %s" % (save_path_prefix+'.0', save_path_prefix+'.1'))

    print("Remove stylistic n_gram...")
    for src_path_prefix in [src_train_path_prefix, src_dev_path_prefix, src_test_path_prefix]:
        mode = src_path_prefix.split('/')[-1]
        for i in range(2):
            f_end = '.' + mode + '.' + str(i)
            process_data(src_path_prefix + '.' + str(i), n_gram_dicts[i], replace_path_prefix + f_end)
            print("Processed data saved at: %s" % (replace_path_prefix + f_end))

    print("Generate template-based results ...")
    for src_path_prefix in [src_test_path_prefix, src_train_path_prefix, src_dev_path_prefix]:
        mode = src_path_prefix.split('/')[-1]
        for i in range(2):
            f_end = mode + '.' + str(i) + '-' + str(1-i) + '.tsf'
            print("Starting search for file: %s" % (replace_path_prefix + f_end))
            cal_sim(train_data_path=replace_path_prefix + '.train.' + str(1-i),
                    test_data_path=replace_path_prefix + f_end,
                    dst_result_path=search_path_prefix + f_end)
            print("End search...")
            get_final_result(search_path_prefix + f_end,
                             final_path_prefix + f_end,
                             src_path_prefix + '.' + str(i))
            print("Final result saved at: %s" % (final_path_prefix + f_end))


if __name__ == "__main__":
    main()

