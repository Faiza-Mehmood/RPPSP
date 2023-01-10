#!usr/bin/env python
# coding:utf-8

"""
Copyright (C) 2019 THL A29 Limited, a Tencent company
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import json
from pathlib import Path
import codecs
import os

BASE_DIR = Path(__file__).resolve().parent.parent

class Config:

    def __init__(self,config_path=None):
        # config_path = os.path.join(BASE_DIR, "Config/DataReaderConfig.json")
        self.data = ConfigReader(config_file=config_path)
        # data_conf = "Config/DataReaderConfig.json"

        # self.data_reader= self.ConfigReader()


class ConfigReader(object):
    """Config load from json file
    """
    def __init__(self, config=None, config_file=None):
        if config_file:
            with open(config_file, 'r') as fin:
                config = json.load(fin)

        self.dict = config
        if config:
            self._update(config)

    def __getitem__(self, key):
        return self.dict[key]

    def __contains__(self, item):
        return item in self.dict

    def items(self):
        return self.dict.items()

    def add(self, key, value):
        """Add key value pair
        """
        self.__dict__[key] = value

    def _update(self, config):
        if not isinstance(config, dict):
            return

        for key in config:
            if isinstance(config[key], dict):
                config[key] = ConfigReader(config[key])

            if isinstance(config[key], list):
                config[key] = [ConfigReader(x) if isinstance(x, dict) else x for x in
                               config[key]]

        self.__dict__.update(config)





