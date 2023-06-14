import glob
import os
import json
from loguru import logger
import click

@logger.catch
@click.command()
@click.option("--pcd_dir", default=None, type=str)
@click.option("--save_dir", default=None, type=str)


def main(root_path, save_path):
    logger.info("this is a main function")
    pcd_path = root_path
    write_path = save_path
    pcds = glob.glob(os.path.join(pcd_path, '*.pcd'))
    for pcd in pcds:
        timestamp = os.patpih.basename(pcd)
        URL = {"point_cloud_Url": "/root/pointcloud/531/" + "{}".format(timestamp)}
        write_file = os.path.join(write_path, timestamp.split('.')[0] + '.json')
        with open(write_file, 'w') as f:
            json.dump(URL, f, indent=4)


if __name__ == "__main__":
    main()
