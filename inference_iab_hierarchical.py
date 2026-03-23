import argparse
import json

from iab_hierarchy import predict_iab_content_hierarchical


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hierarchical IAB inference for a query.")
    parser.add_argument("text", help="Query text")
    args = parser.parse_args()
    print(json.dumps(predict_iab_content_hierarchical(args.text), indent=2))


if __name__ == "__main__":
    main()
