#
# Copyright 2023 The EA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################
"""Benchmarking script for EA."""

from __future__ import print_function
import argparse
import sys
import logging
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.table as pltt
import pathlib
import subprocess


benchmark = './benchmark/distance/benchmark_in_one'

benchmark_file = 'benchmark_in_one.json'

def run_benchmark(args):

    print("run bench", benchmark_file)
    if args.skip == 'true':
        print("skip bench")
        return
    s = subprocess.run([benchmark, '--benchmark_out=' + benchmark_file, '--benchmark_format=json'])
    print("run bench done", s)


logging.basicConfig(format="[%(levelname)s] %(message)s")

METRICS = [
    "real_time",
    "cpu_time",
    "bytes_per_second",
    "items_per_second",
    "iterations",
]
TRANSFORMS = {"": lambda x: x, "inverse": lambda x: 1.0 / x}


def get_default_ylabel(args):
    """Compute default ylabel for commandline args"""
    label = ""
    if args.transform == "":
        label = args.metric
    else:
        label = args.transform + "(" + args.metric + ")"
    if args.relative_to is not None:
        label += " relative to %s" % args.relative_to
    return label


def parse_args():
    """Parse commandline arguments"""

    print("parse args add arg")
    parser = argparse.ArgumentParser(description="Visualize google-benchmark output")
    parser.add_argument(
        "-m",
        metavar="METRIC",
        choices=METRICS,
        default=METRICS[0],
        dest="metric",
        help="metric to plot on the y-axis, valid choices are: %s" % ", ".join(METRICS),
    )

    parser.add_argument(
        "-s",
        metavar="SKIP",
        choices=['true', 'false'],
        default=['false'],
        dest="skip",
        help="metric to plot on the y-axis, valid choices are: %s" % ", ".join(METRICS),
    )

    parser.add_argument(
        "-t",
        metavar="TRANSFORM",
        choices=TRANSFORMS.keys(),
        default="",
        help="transform to apply to the chosen metric, valid choices are: %s"
             % ", ".join(list(TRANSFORMS)),
        dest="transform",
    )
    parser.add_argument(
        "-r",
        metavar="RELATIVE_TO",
        type=str,
        default=None,
        dest="relative_to",
        help="plot metrics relative to this label",
    )
    parser.add_argument(
        "--xlabel", type=str, default="input size", help="label of the x-axis"
    )
    parser.add_argument("--ylabel", type=str, help="label of the y-axis")
    parser.add_argument("--title", type=str, default="", help="title of the plot")
    parser.add_argument(
        "--logx", action="store_true", help="plot x-axis on a logarithmic scale"
    )
    parser.add_argument(
        "--logy", action="store_true", help="plot y-axis on a logarithmic scale"
    )
    parser.add_argument(
        "--output", type=str, default="", help="File in which to save the graph"
    )
    print("parse args")
    args = parser.parse_args()
    print("parse args done")
    if args.ylabel is None:
        args.ylabel = get_default_ylabel(args)

    return args


def parse_input_size(name):
    splits = name.split("/")
    if len(splits) == 1:
        return 1
    return int(splits[1])


def read_data(args):
    """Read and process dataframe using commandline args"""
    print("read data")
    extension = pathlib.Path(benchmark_file).suffix
    try:
        if extension == ".csv":
            print("read csv")
            data = pd.read_csv(benchmark_file, usecols=["name", args.metric])
        elif extension == ".json":
            print("read json")
            raw_data = open(benchmark_file).read()
            json_data = json.loads(raw_data)
            print("read json done")
            data = pd.DataFrame(json_data["benchmarks"])
        else:
            logging.error("Unsupported file extension '{}'".format(extension))
            exit(1)
    except ValueError:
        logging.error(
            'Could not parse the benchmark data. Did you forget "--benchmark_format=[csv|json] when running the benchmark"?'
        )
        print("error", ValueError)
        exit(1)
    data["label"] = data["name"].apply(lambda x: x.split("/")[0])
    data["input"] = data["name"].apply(parse_input_size)
    data[args.metric] = data[args.metric].apply(TRANSFORMS[args.transform])
    return data


def plot_groups(label_groups, args):
    """Display the processed data"""
    for label, group in label_groups.items():
        plt.plot(group["input"], group[args.metric], label=label, marker=".")
    if args.logx:
        plt.xscale("log")
    if args.logy:
        plt.yscale("log")
    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)
    plt.title(args.title)
    plt.legend()
    if args.output:
        logging.info("Saving to %s" % args.output)
        plt.savefig(args.output)
    else:
        plt.show()


def plot_groups_table(label_groups, args):
    """Display the processed data"""
    for label, group in label_groups.items():
        #pltt.plot(group["input"], group[args.metric], label=label, marker=".")
        pltt.table(cellText=group[args.metric], rowLabels=group["input"], colLabels=label)
    if args.output:
        logging.info("Saving to %s" % args.output)
        pltt.show()


def main():
    """Entry point of the program"""

    logging.error("%s", str("hello"))
    print("hello")
    args = parse_args()
    run_benchmark(args)
    data = read_data(args)
    print("data", data)
    label_groups = {}
    for label, group in data.groupby("label"):
        label_groups[label] = group.set_index("input", drop=False)
    if args.relative_to is not None:
        try:
            baseline = label_groups[args.relative_to][args.metric].copy()
        except KeyError as key:
            msg = "Key %s is not present in the benchmark output"
            logging.error(msg, str(key))
            exit(1)

    if args.relative_to is not None:
        for label in label_groups:
            label_groups[label][args.metric] /= baseline
    plot_groups(label_groups, args)


if __name__ == "__main__":
    main()
