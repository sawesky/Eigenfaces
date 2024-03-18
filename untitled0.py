import matplotlib.pyplot as plt
import numpy as np
import os
import random 

def plot_portraits(images, titles, h, w, n_row, n_col, plotnum):
    plt.figure(plotnum, figsize = (2.2 * n_col, 2.2 * n_row))
    plt.subplots_adjust(bottom = 0, left = .01, right = .99, top = .90, hspace = .20)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap = plt.cm.gray)
        plt.title(titles[i])
        plt.xticks(())
        plt.yticks(())
        

def pca(X, titles, h, w, n_pc):
    n_features, n_samples = X.shape
    mean = np.mean(X, axis=1)
    plotmean = mean[:, np.newaxis]
    
    plt.figure(10)
    plt.imshow(plotmean.reshape((h, w)), cmap = plt.cm.gray)
    plt.title('Srednje lice')
    plt.xticks(())
    plt.yticks(())
    
    centered_data = X - mean[:, np.newaxis]
    plot_portraits(centered_data.T, titles, h, w, 4, 4, 11)
    
    U, S, V = np.linalg.svd(centered_data)
    U = U.T
    components = U[:n_pc]
    
    plt.figure(12, figsize = (12, 7))
    t = range(len(S))
    plt.subplot(2, 1, 1)
    plt.title('Sopstvene vrednosti')
    plt.plot(t, S**2, '*')
    plt.subplot(2, 1, 2)
    plt.title('Kumulativna objasnjena varijansa')
    plt.plot(t, np.cumsum(S**2)/np.sum(S**2))
    return components, mean, centered_data


def reconstruction(Y, C, M, h, w, image_index):
    
    n_samples, n_features = Y.shape
    weights = np.dot(Y.T, C.T)
    centered_vector=np.dot(weights[image_index, :], C)
    recovered_image=(M+centered_vector).reshape(h, w)
    return recovered_image

def recognize_face(i, C, M, h, w, images, celebrity_names, weights):
    
    input_image = images[i]
    input_face = input_image.reshape(h * w, 1)
    centered_input_face = input_face - M[:, np.newaxis]
    weights_new = np.dot(C, centered_input_face)
    reconstructed_input_face = M[:, np.newaxis] + np.dot(C.T, weights_new)
    distances = [np.linalg.norm(weights_new.T - weights[i, :]) for i in range(len(weights))]
    #distances = np.linalg.norm(centered_input_face - reconstructed_input_face.flatten()[:, np.newaxis], axis=0)
    reconstructed_input_face = reconstructed_input_face.reshape((h, w))
    closest_match_index = np.argmin(distances)
    return closest_match_index, input_image, reconstructed_input_face, images[closest_match_index]

def test_rec_face(input_img, C, M, h, w, weights):
    
    input_face = input_img.reshape(h*w, 1)
    centered_input_face = input_face - M[:, np.newaxis]
    weights_new = np.dot(C, centered_input_face)
    reconstructed_input_face = M[:, np.newaxis] + np.dot(C.T, weights_new)
    distances = [np.linalg.norm(weights_new.T - weights[i, :]) for i in range(len(weights))]
    reconstructed_input_face = reconstructed_input_face.reshape((h, w))
    closest_match_index = np.argmin(distances)
    return closest_match_index, reconstructed_input_face, images[closest_match_index]


def plot_face_recognition_results(input_image, reconstructed_image, closest_match_image, celebrity_names, closest_match_index):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(input_image, cmap=plt.cm.gray)
    plt.title('Ulaz: ' + celebrity_names[closest_match_index])
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(reconstructed_image, cmap=plt.cm.gray)
    plt.title('Rekonstruisano')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(closest_match_image, cmap=plt.cm.gray)
    plt.title('Prepoznato:' + celebrity_names[closest_match_index])
    plt.axis('off')

    plt.show()


def plot_face_recognition_results_test(input_image, reconstructed_image,
                                       closest_match_image, celebrity_names, closest_match_index, test_name):
    
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(input_image, cmap=plt.cm.gray)
    plt.title('Ulaz: ' + test_name)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(reconstructed_image, cmap=plt.cm.gray)
    plt.title('Rekonstruisano')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(closest_match_image, cmap=plt.cm.gray)
    plt.title('Prepoznato: ' + celebrity_names[closest_match_index])
    plt.axis('off')

    plt.show()


plotnum = 1;
direc = 'faces'
celebrity_photos = os.listdir(direc)
celebrity_photos = random.sample(celebrity_photos, 1000)
celebrity_images = [direc + '/' + photo for photo in celebrity_photos]
images = np.array([plt.imread(image) for image in celebrity_images], dtype = np.float64)
celebrity_names = [name[:name.find('0') - 1].replace("_", " ") for name in celebrity_photos]
n_samples, h, w = images.shape
plot_portraits(images, celebrity_names, h, w, n_row = 4, n_col = 4, plotnum=plotnum)
plotnum += 1


n_components = 20
X = images.reshape(n_samples, h*w).T 
C, M, Y= pca(X, celebrity_names, h, w, n_pc=n_components)
eigenfaces =  C.reshape((n_components, h, w))
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_portraits(eigenfaces, eigenface_titles, h, w, 4, 4, plotnum=plotnum) 
plotnum += 1
recovered_images=[reconstruction(Y, C, M, h, w, i) for i in range(len(images))]
plot_portraits(recovered_images, celebrity_names, h, w, n_row=4, n_col=4, plotnum=plotnum)
plotnum += 1

n_components = 100
C, M, Y= pca(X, celebrity_names, h, w, n_pc=n_components)
recovered_images=[reconstruction(Y, C, M, h, w, i) for i in range(len(images))]
plot_portraits(recovered_images, celebrity_names, h, w, n_row=4, n_col=4, plotnum=plotnum)
plotnum += 1
    
n_components = 200
C, M, Y= pca(X, celebrity_names, h, w, n_pc=n_components)
recovered_images=[reconstruction(Y, C, M, h, w, i) for i in range(len(images))]
plot_portraits(recovered_images, celebrity_names, h, w, n_row=4, n_col=4, plotnum=plotnum)
plotnum += 1
        
weights = np.dot(Y.T, C.T)


#%%

chosen_image_index = 35  
closest_match_index, input_image, reconstructed_input_face, closest_match_image = recognize_face(
    chosen_image_index, C, M, h, w, images, celebrity_names, weights)

plot_face_recognition_results(input_image, reconstructed_input_face, closest_match_image, celebrity_names, closest_match_index)

celebrity_photos = os.listdir(direc)
test_sample = random.sample(celebrity_photos, 1)[0]
test_celebrity = direc + '/' + test_sample
test_img = np.array(plt.imread(test_celebrity), dtype = np.float64)
test_name = test_sample[:test_sample.find('0') - 1].replace("_", " ")

closest_match_index, reconstructed_input_face, closest_match_image = test_rec_face(test_img, C, M, h, w, weights)

plot_face_recognition_results_test(test_img, reconstructed_input_face, closest_match_image, celebrity_names, closest_match_index, test_name)


