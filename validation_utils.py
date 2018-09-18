from PIL import Image, ImageDraw


def coordToMatrix(coord, w, h):
    img_size = (w, h)
    poly = Image.new("RGB", img_size)
    pdraw = ImageDraw.Draw(poly)
    pdraw.polygon(coord,
                  fill=(255, 255, 255), outline=(255, 255, 255))
    poly = poly.transpose(Image.FLIP_LEFT_RIGHT)
    poly = poly.rotate(180)
    # pix = np.array(poly.getdata()).reshape(w, h)
    return poly


def find_centroid(im):
    width, height = im.size
    XX, YY, count = 0, 0, 0
    for x in range(0, width, 1):
        for y in range(0, height, 1):
            if im.getpixel((x, y)) == (255, 255, 255):
                XX += x
                YY += y
                count += 1
    return (XX / count, YY / count) if count > 0 else (0, 0)


def compute_area(im):
    width, height = im.size
    area = 0
    for x in range(0, width, 1):
        for y in range(0, height, 1):
            if im.getpixel((x, y)) == (255, 255, 255):
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
        if y[indice] > y_max:
            y_max = y[indice]
    return [x_max, x_min, y_max, y_min]

def cade_internamente(max, centroide, margin):
    if centroide[0] < max[0] + margin and centroide[0] > max[1] - margin:
        if centroide[1] < max[2] + margin and centroide[1] > max[3] - margin:
            # return attributes
            return True
    return False
