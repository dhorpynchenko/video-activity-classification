import json
from PIL import Image, ImageDraw
from validation_example import MaskExam as Mask

path = "/home/students/Desktop/WD"
jsonName = "export.json"
nomeBase = "image"

ret1 = []
ret2 = []

jsonPath = path + "/" + jsonName
b = json.load(open(jsonPath))

immNum = 0
vector = []

for x in b[immNum]['Label'].keys():
    name = x

nameApp = name

lim1 = len(b[immNum]['Label'].keys())
print("Number of keys: "+ str(lim1))

iter = 0

idx = 0
tot = len(b)
idx_2 = 0
success = 0
total = 1
classes = {}
#b= b[:43]
#b = b[36:]
# start the loop for every entity in the json list
for json_elem in b[:10]:

    if json_elem['Label'] == 'Skip':
        continue

    # add class to class list
    for label in json_elem['Label'].keys():
        if label not in classes.keys():
            classes[label] = len(classes) + 1

def coordToMatrix(coord, w, h):
    img_size = (w, h)
    poly = Image.new("RGB", img_size)
    pdraw = ImageDraw.Draw(poly)
    pdraw.polygon(coord,
                  fill=(255,255,255), outline=(255,255,255))
    poly = poly.transpose(Image.FLIP_LEFT_RIGHT)
    poly = poly.rotate(180)
    #pix = np.array(poly.getdata()).reshape(w, h)
    return poly


def find_centroid(im):
    width, height = im.size
    XX, YY, count = 0, 0, 0
    for x in range(0, width, 1):
        for y in range(0, height, 1):
            if im.getpixel((x, y)) == (255,255,255):
                XX += x
                YY += y
                count += 1
    return XX/count, YY/count

def compute_area(im):
    width, height = im.size
    area = 0
    for x in range(0, width, 1):
        for y in range(0, height, 1):
            if im.getpixel((x, y)) == (255,255,255):
                area += 1
    return area

def find_max_coord(x, y):
    x_max = 0
    x_min = 10000000
    y_max = 0
    y_min = 10000000
    for indice in range(len(x)):
        if x[indice] < x_min:
            x_min = x[indice]
        if y[indice] < y_min:
            y_min = y[indice]
        if x[indice] > x_max:
            x_max = x[indice]
        if y[indice] > x_max:
            x_max = y[indice]
    return [x_max, x_min, y_max, y_min]

def cade_internamente(max, centroide):
    if centroide[0]< max[0] and centroide[0] > max[1]:
        if centroide[1]< max[2] and centroide[1] > max[3]:
            # return attributes
            return True
    return False

iter= 0


for json_elem in b[:10]:
    if iter ==1 or iter == 6 or iter== 8:
        iter+=1
        continue
    if json_elem['Label'] == 'Skip':
        continue
    # read image for sizes
    img_path = "/home/students/Desktop/provaZ/image" + str(iter) + ".png"
    iter+= 1
    try:
        im = Image.open(img_path)
        w, h = im.size
        seg = json_elem["Label"]
        maskMat = []
        idClassi = []

        idss = []
        centroidi_lista = []
        aree = []
        max_coord = []
        for i in seg.keys():
            name = str(i)
            class_id_name = classes[name]

            for j in range(len(seg[name])):
                idClassi.append(classes.get(name))
                x_coord = []
                y_coord = []
                for k in range(len(seg[name][j])):
                    y_coord.append(seg[name][j][k]['y'])
                    x_coord.append(seg[name][j][k]['x'])
                coord = []
                for ind in range(len(x_coord)):
                    coord.append(x_coord[ind])
                    coord.append(y_coord[ind])
                immagine = coordToMatrix(coord, w, h)
                centroidi_lista.append(find_centroid(immagine))
                aree.append(compute_area(immagine))
                idss.append(class_id_name)
                max_coord.append(find_max_coord(x_coord, y_coord))


        centroidi_lista_mask, idss_mask, aree_mask = Mask.centreAnalisi(im, w, h)
        for indice in range(len(idss)):
            total += 1
            for indice_mask in range(len(idss_mask)):
                if (aree[indice] *0.5)< aree_mask[indice_mask] and aree_mask[indice_mask] < (aree[indice] *1.5):
                    if cade_internamente(max_coord[indice], centroidi_lista_mask[indice_mask]):
                        if idss_mask[indice_mask] == idss[indice]:
                            success += 1
    except Exception as e:
        print(e)


print("Numero di successi: " + str(success))
print("Numero totale label: " + str(total))
print("Percentuale di successo: "+ str(float(success) / float(total) ) + "%")
