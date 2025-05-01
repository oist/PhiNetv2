from eval.start import main as eval_with_cropmae_script

if __name__ == "__main__":
    import argparse
    from eval.util_eval import get_args_parser

    parser = get_args_parser()
    args = parser.parse_args()

    # Parse the arguments
    args = vars(args)
    if args["jhmdb"] or args["vip"]:
        eval_with_cropmae_script(args)
    else:
        # If neither jhmdb nor vip is specified, print an error message
        print("Please specify either --jhmdb or --vip to evaluate on the respective dataset.")
        # exit(1)