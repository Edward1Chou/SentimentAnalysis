#! /bin/env python
# -*- coding: utf-8 -*-

with open("n_pos.csv", "w") as n:
    with open("pos.csv", "r") as p:
        for line in p.readlines():
            if line == "\"\n":
                continue
            n.write(line)

line = "\""
print len(line)