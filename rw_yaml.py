import yaml
my_yaml = "/home/example.yaml"
out_yaml = '/home/out.yaml'


def read_yaml(yaml_path):
    return yaml.load(open(yaml_path, encoding="utf-8", mode="r"), Loader=yaml.FullLoader)


def write_yaml(yaml_path,data):
    yaml.dump(data, stream=open(yaml_path, encoding="utf-8", mode="w"), allow_unicode=False)


if __name__ == '__main__':
    my_dict = read_yaml(my_yaml)
    for key, value in my_dict.items():
        o = value[0]
    for p, q in o.items():
        if p == 'cards':
            obj = q
    for i in range(len(obj)):
        tmp = obj[i]
        tt = {}
        tt['card_id'] = tmp
        tt['project'] = 'hd'
        tt['media_name'] = 'label_3d_object'
        obj[i] = tt
    write_yaml(out_yaml, my_dict)