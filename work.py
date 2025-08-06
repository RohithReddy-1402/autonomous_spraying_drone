from PIL import Image
img = Image.open("cursor.png")
img = img.resize((32,32)) 
img.save("cursor-small.png")
