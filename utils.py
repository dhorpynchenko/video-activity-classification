def load_required_classes(file):
    class_names = []
    with open(file) as file:
        for line in file.readline():
            if line and not line.startswith('#'):
                class_names.append(line)

    return class_names
