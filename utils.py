def load_required_classes(file):
    class_names = []
    with open(file) as file:
        for line in file.readlines():
            if line and not line.startswith('#'):
                class_names.append(line.strip("\n"))

    return class_names


def load_class_ids(file):
    classes = dict()
    with open(file) as file:
        for line in file.readlines():
            parts = line.split("\t")
            classes[int(parts[0])] = parts[1].strip("\n")
    return classes

def make_reversed_dict(dictionary: dict):
    return dict(zip(dictionary.values(), dictionary.keys()))
