from PIL import Image

#untuk membuka gambar
img = Image.open("image.jpeg")

img.save("image_new.png") #untuk menyimpan gambar dengan format berbeda
img.show() # untuk menampilkan Gambar
print(img.format) # untuk menampilkan format gambar
print(img.size) # untuk mengetahui ukuran gambar