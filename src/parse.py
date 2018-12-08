import sys

def parse_txt(path):
    l_dict = {'c': 0, 'i': 1}
    with open(path, 'r') as f:
        sens = f.read().split('\n\n')
        res_list = []
        for sen in sens:
            word_list = []
            label_list = []
            for pair in sen.split('\n'):
                pair = pair.split()
                if len(pair) == 2:
                    label_list.append(str(l_dict[pair[1]]))
                    word_list.append(pair[0])
            words = ' '.join(word_list)
            label_list.append(words)
            res_list.append('\t'.join(label_list))
    return res_list

if __name__ == '__main__':
    path = sys.argv[1]

    data = parse_txt(path)

    with open('../data/parsed_output.txt', 'w') as f:
        for item in data:
            f.write("%s\n" % item)
