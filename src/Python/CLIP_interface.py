import torch
import clip
import os
from PIL import Image

DEBUG = False

def constrained(values, indices):
    map = [(indices[i], values[i].item(), values[i]) for i in range(len(indices))]

    map_1 = [x for x in map if x[0] in [0,1,2,3,4]]
    map_2 = [x for x in map if x[0] in [5,6,7,8]]
    map_3 = [x for x in map if x[0] in [9,10,11,12]]

    #max from map_1
    map_1 = max(map_1, key=lambda x: x[1]) if len(map_1) > 0 else []
    map_2 = max(map_2, key=lambda x: x[1]) if len(map_2) > 0 else []

    #merge 
    v, i = [], []
    if len(map_1) > 0:
        v.append(map_1[2])
        i.append(map_1[0])
    if len(map_2) > 0:
        v.append(map_2[2])
        i.append(map_2[0])
    for x in map_3:
        v.append(x[2])
        i.append(x[0])
    
    return v, i

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
        if similarity[0][i] > 0.02:
            values.append(similarity[0][i])
            indices.append(i)

    return values, indices

def process_tray(tray, labels):
    global input_folder, output_folder

    # create output folder
    if not os.path.exists(output_folder+tray):
        os.makedirs(output_folder+tray)

    if DEBUG: print('Processing', tray)

    new_labels = []

    # process food_image
    if DEBUG: print('   Food image')
    for img in os.listdir(path := input_folder+tray+'/food_image/'):
        if DEBUG: print('     ',img)

        values, indices = process_image(path+img, labels)

        values, indices = constrained(values, indices)

        if not os.path.exists(out := output_folder+tray+'/food_image/'):
            os.makedirs(out)

        with open(out+img+'.txt', 'w') as f:
            for i in range(len(indices)):
                f.write(str(labels.index(labels[indices[i]])+1)+'\n')

        for i in range(len(indices)):
            new_labels.append(labels[indices[i]]) if labels[indices[i]] not in new_labels else None 

        if DEBUG:
            for i in range(len(indices)):
                print('     ',labels[indices[i]], values[i].item())
            print()

    # process leftovers
    for i in range(1,4):
        if DEBUG: print('   leftover', i)
        for img in os.listdir(path := input_folder+tray+'/leftover'+str(i)+'/'):
            if DEBUG: print('     ',img)

            values, indices = process_image(path+img, new_labels)

            indices = [labels.index(new_labels[indices[i]]) for i in range(len(indices))]

            values, indices = constrained(values, indices)

            if not os.path.exists(out := output_folder+tray+'/leftover'+str(i)+'/'):
                if DEBUG: print(out)
                os.makedirs(out)

            with open(out+img+'.txt', 'w') as f:
                for index in indices:
                    f.write(str(index+1)+'\n')

            if DEBUG: 
                for j in range(len(indices)):
                    print('     ',labels[indices[j]], values[j].item())
                print()

# the first run of this script will download the model
def main(i : int = None):
    global device, model, preprocess, input_folder, output_folder

    input_folder = './plates/'
    output_folder = './output/'

    labels = [
        "pasta with pesto",
        "pasta with tomato sauce",
        "pasta with meat sauce",
        "pasta with clams and mussels",
        "pilaw rice with peppers and peas",
        "pork meat slices",
        "fish cutlet",
        "rabbit with bones",
        "cuttlefish food",
        "brown beans",
        "potatoes",
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    if i is not None:
	    process_tray("tray"+str(i), labels)

    else:
        for tray in os.listdir(input_folder):
            process_tray(tray, labels)

if __name__ == "__main__":
    main()