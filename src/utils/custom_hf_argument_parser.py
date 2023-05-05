import json
import logging
import sys
from typing import Tuple

from transformers import HfArgumentParser
from transformers.hf_argparser import DataClass


class CustomHfArgumentParser(HfArgumentParser):
    def parse_dictionary_and_args(self) -> Tuple[DataClass, ...]:
        """
        Alternative helper method that does not use `argparse` at all, instead loading a json file and populating the
        dataclass types.
        """
        args = []
        data = {}
        for i in range(1, len(sys.argv)):
            if not sys.argv[i].endswith('.json'):
                break

            with open(sys.argv[i]) as f:
                new_data = json.load(f)
            conflicting_keys = set(new_data.keys()).intersection(data.keys())
            if len(conflicting_keys) > 0:
                raise ValueError(f'There are conflicting keys in the config files: {conflicting_keys}')
            data.update(new_data)

        for k, v in data.items():
            # if any options were given explicitly through the CLA then they override anything defined in the config files
            if f'--{k}' in sys.argv:
                logging.info(f'While {k}={v} was given in a config file, a manual override was set through the CLA')
                continue
            args.extend(
                ["--" + k, *(v if isinstance(v, list) else [str(v)])]
            )  # add the file arguments first so command line args has precedence
        args += sys.argv[i:]

        return self.parse_args_into_dataclasses(args=args, look_for_args_file=False)