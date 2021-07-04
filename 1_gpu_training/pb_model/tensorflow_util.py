# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import math
import sys
import os
import argparse
from logging import *
import logging
import tensorflow as tf
from google.protobuf import text_format
pathSep = os.path.sep


def getPrefix(filepath):
    filename = filepath.split(pathSep)[-1]
    prefix = filename.split(".")[0]
    return prefix


def convert_pb_to_pbtxt(filename):
    prefix = getPrefix(filename)
    dirname = os.path.dirname(filename)
    with tf.gfile.GFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # tf.import_graph_def(graph_def, name='')
        tf.train.write_graph(graph_def, dirname, prefix + ".pbtxt", as_text=True)
    return


def convert_pbtxt_to_pb(filename):
    prefix = getPrefix(filename)
    dirname = os.path.dirname(filename)
    print(dirname)
    with tf.gfile.GFile(filename, 'r') as f:
        graph_def = tf.GraphDef()
        text_format.Merge(f.read(), graph_def)
        # tf.import_graph_def(graph_def, name='')
        tf.train.write_graph(graph_def, dirname, prefix + ".pb", as_text=False)
    return


def main(args):
    print(args)
    file = args.file
    filepath = file.name
    abspath = os.path.abspath(filepath)
    prefix = getPrefix(abspath)
    pb_to_txt = args.pb_to_txt
    txt_to_pb = args.txt_to_pb
    if pb_to_txt:
        convert_pb_to_pbtxt(abspath)
    elif txt_to_pb:
        convert_pbtxt_to_pb(abspath)


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s", level=logging.DEBUG)
    argsParse = argparse.ArgumentParser(
        prog=sys.argv[0], description="This script a util about tensorflow", epilog="Enjoy it.")
    argsParse.add_argument("file", help="file path",
                           type=argparse.FileType('r', encoding="utf-8"))
    actions = argsParse.add_mutually_exclusive_group(required=True)
    actions.add_argument("-t", "--pb_to_txt", help="protobuf file to protobuf text", action="store_true", default=False)
    actions.add_argument("-p", "--txt_to_pb", help="protobuf text to protobuf file", action="store_true", default=False)
    args = argsParse.parse_args()
    main(args)
