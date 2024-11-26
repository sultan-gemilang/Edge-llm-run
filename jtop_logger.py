#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# This file is part of the jetson_stats package (https://github.com/rbonghi/jetson_stats or http://rnext.it).
# Copyright (c) 2019-2023 Raffaello Bonghi.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

from jtop import jtop, JtopException
import csv
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple jtop logger')
    # Standard file to store the logs
    parser.add_argument('--file', action="store", dest="file", default="log.csv")
    parser.add_argument('--interval', type=float, default=1)
    args = parser.parse_args()
    
    save_path = './logger/'
    save_file = os.path.join(save_path, args.file)

    print("Simple jtop logger")
    print(f"Saving log on {save_file}")
    print(f'Process PID {os.getpid()}')
    
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    try:
        with jtop(interval=args.interval) as jetson:
            # Make csv file and setup csv
            with open(save_file, 'w') as csvfile:
                stats = jetson.stats
                # Initialize cws writer
                writer = csv.DictWriter(csvfile, fieldnames=stats.keys())
                # Write header
                writer.writeheader()
                # Write first row
                writer.writerow(stats)
                # Start loop
                while jetson.ok():
                    stats = jetson.stats
                    # Write row
                    writer.writerow(stats)
                    #print("Log at {time}".format(time=stats['time']))
    except JtopException as e:
        print(e)
    except KeyboardInterrupt:
        print("Closed with CTRL-C")
    except IOError:
        print("I/O error")
# EOF
