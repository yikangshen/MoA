# -*- coding: utf-8 -*-
import argparse
import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
# reload(sys)
# sys.setdefaultencoding('utf8')
parser = argparse.ArgumentParser()


parser.add_argument("--generate_file", type=str, default='generate/generate-test.txt')
parser.add_argument("--output_file", type=str, default='generate/generate-test.cleaned')
arg = parser.parse_args()

with open(arg.generate_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
# print(lines[0])
generate_dict = {}
for line in lines:
    # line = line.strip()
    if line[0] == 'H':
        # print(line)
        sentences = line.split('\t')
        # print(int(sentences[0].split('-')[1]))
        generate_dict[int(sentences[0].split('-')[1])] = sentences[-1]
print(generate_dict[0])
with open(arg.output_file, 'w', encoding='utf-8') as f:
    for i in range(len(generate_dict)):
        # print(generate_dict[i])
        f.write(generate_dict[i])

