import sys

str_line = []
for line in open(sys.argv[1]):
    fields = line.strip().split()
    if len(fields) != 2:
        print(" ".join(str_line))
        str_line = []
        continue

    word = fields[0]
    tag = fields[1]

    str_line.append(tag)