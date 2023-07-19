import torch
import clip
import os
from PIL import Image

DEBUG = False

def process_image(img, labels):
    global device, model, preprocess

    image = preprocess(Image.open(img)).unsqueeze(0).to(device)
    text = clip.tokenize(labels).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
    
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    #save values and indices for label with confidence > 0.5
    values = []
    indices = []
    for i in range(len(labels)):
        if similarity[0][i] > 0.2:
            values.append(similarity[0][i])
            indices.append(i)

    return values, indices

def process_tray(tray, labels):
    global input_folder, output_folder

    # create output folder
    if not os.path.exists(output_folder+tray):
        os.makedirs(output_folder+tray)

    if DEBUG: print('Processing', tray)

    # process food_image
    if DEBUG: print('   Food image')
    max = ("", 0)
    for img in os.listdir(path := input_folder+tray+'/food_image/'):
        if DEBUG: print('     ',img)

        values, indices = process_image(path+img, labels)

        #print(values, indices)

        for index in indices:
            if index == 0 and values[0] > max[1]:
                max = (img, values[0].item())

        if not os.path.exists(out := output_folder+tray+'/food_image/'):
            os.makedirs(out)

        if DEBUG:
            for i in range(len(indices)):
                print('        ',labels[indices[i]], values[i].item())
            print()
    
    if DEBUG: print(max)
    if max[0] != "":
            with open(out+max[0]+'.txt', 'w') as f:
                f.write(str(max[1]))

    # process leftovers
    for i in range(1,4):
        if DEBUG: print('   leftover', i)
        max = ("", 0)
        for img in os.listdir(path := input_folder+tray+'/leftover'+str(i)+'/'):
            if DEBUG: print('     ',img)

            values, indices = process_image(path+img, labels)

            #print(values, indices)

            for index in indices:
                if index == 0 and values[0] > max[1]:
                    max = (img, values[0].item())

            if not os.path.exists(out := output_folder+tray+'/leftover'+str(i)+'/'):
                os.makedirs(out)

            if DEBUG: 
                for j in range(len(indices)):
                    print('        ',labels[indices[j]], values[j].item())
                print()
        
        if DEBUG: print(max)
        if max[0] != "":
            with open(out+max[0]+'.txt', 'w') as f:
                f.write(str(max[1]))

# the first run of this script will download the model
def main( i : int = None ):

    global device, model, preprocess, input_folder, output_folder

    input_folder = './bread/'
    output_folder = './bread_output/'

    labels = [
        "bread or white bread or canteen bread (also partially eaten)",
        "other stuff on a food tray that is NOT bread",
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    i = 5

    if i is None:
        for tray in os.listdir(input_folder):
            process_tray(tray, labels)
    else:
        process_tray("tray"+str(i), labels)


if __name__ == "__main__":
    main()