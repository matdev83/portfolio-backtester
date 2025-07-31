"""Print top cumulative functions from a cProfile stats file.
Usage: python profile_summary.py profile.prof [N]
"""
import sys
import pstats
import argparse


def main():
    parser = argparse.ArgumentParser(description="Print top cumulative time functions from a cProfile stats file")
    parser.add_argument("profile", help="Path to .prof file")
    parser.add_argument("-n", "--num", type=int, default=40, help="Number of rows to print (default 40)")
    args = parser.parse_args()

    stats = pstats.Stats(args.profile)
    stats.sort_stats("cumulative").print_stats(args.num)


if __name__ == "__main__":
    main()
