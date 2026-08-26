#!/usr/bin/env python3
import argparse

from dsmc_v2_contracts import finalize_run


def main():
    parser = argparse.ArgumentParser(description="Validate and finalize one HS_CTC_v2 output directory")
    parser.add_argument("run_directory")
    args = parser.parse_args()
    qa = finalize_run(args.run_directory)
    print(f"{args.run_directory}: {qa['status']} ({qa['n_attempts']} attempts, {qa['n_hits']} hits)")


if __name__ == "__main__":
    main()

