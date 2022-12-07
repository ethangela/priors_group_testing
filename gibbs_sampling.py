import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import os 



def load_image(mode):
    if mode == 'grid_28_28':
        img_gray = np.ones((28,28))
    elif mode == 'grid_72_72':
        img_gray = np.ones((72,72))
    elif mode == 'grid_28_28_block_7_grid':
        img_gray = np.ones((28+3,28+3))
        for i in range(28+3):
            for j in range(28+3):
                if i%8==7 or j%8==7:
                    img_gray[i,j] = 0
    elif mode == 'grid_72_72_block_18_grid':
        img_gray = np.ones((72+3,72+3))
        for i in range(72+3):
            for j in range(72+3):
                if i%19==18 or j%19==18:
                    img_gray[i,j] = 0
    img_padded = np.zeros([img_gray.shape[0] + 2, img_gray.shape[1] + 2])
    img_padded[1:-1, 1:-1] = img_gray
    return img_padded



class Ising:
    def __init__(self, beta, ita, U):
        self.beta = beta
        self.ita = ita
        self.U = U
        
    def posterior(self,i,j,Y):
        markov_blanket = [Y[i-1,j]*self.U[i-1,j], Y[i,j-1]*self.U[i,j-1], Y[i,j+1]*self.U[i,j+1], Y[i+1,j]*self.U[i+1,j]]
        term = self.beta * np.sum(markov_blanket) * 1 - self.ita * 1
        return 1 / (1 + math.exp(-2*term))



def gibbs_sampling(mode, burn_in_steps, total_samples, beta, ita):
    
    U = load_image(mode) #graph bone 
    Y = np.random.choice([-1, 1], size=U.shape) #graph random initialization 
    ising_model = Ising(beta, ita, U)
    
    for step in range(burn_in_steps + total_samples):
        print(f'Step {step} ...')
        for i in range(1, U.shape[0]-1):
            for j in range(1, U.shape[1]-1):
                posterior_prob = ising_model.posterior(i,j,Y) #ising_model.posterior_qp(i,j,Y)
                y = (np.random.rand() < posterior_prob) * 2 -1
                Y[i, j] = y

    Y = Y[1:-1, 1:-1]
    
    if mode == 'grid_28_28_block_7_grid':
        Y = np.delete(Y, [7, 15, 23], 0)
        Y = np.delete(Y, [7, 15, 23], 1)
    if mode == 'grid_72_72_block_18_grid':
        Y = np.delete(Y, [18, 37, 56], 0)
        Y = np.delete(Y, [18, 37, 56], 1)
    
    return Y



def facebook(seed, burn_in_steps, total_samples, beta, ita):
    # adjacency = np.load(f'game/period_{sec}_secs.npy') 
    adj = np.load(f'fb/network_500_seed_{seed}.npy') 
    Y = np.random.choice([-1, 1], size=adj.shape[0]) 
    adjacency = adj + adj.T

    for step in range(burn_in_steps + total_samples):
        print(f'Step {step} ...')
        for i in range(adjacency.shape[0]):
            markov_blanket = 0
            for j in range(adjacency.shape[0]):
                if adjacency[i][j] == 1:
                    markov_blanket += Y[j]
            term = beta * markov_blanket - ita 
            posterior = 1 / (1 + math.exp(-2*term))
            y = (np.random.rand() < posterior) * 2 -1
            Y[i] = y
    
    return Y




def save_image(image, title):
    images = np.tile(image,(30,1))
    plt.imshow(images, cmap='gray')
    plt.savefig(title + '.png')
    plt.close()



if __name__ == '__main__':

    #grid graph
    GRAPH = 'grid_28_28'
    total_samples = 500
    burn_in_steps = 500

    for round in range(5):
        for BETA in [0.5]:
            for ITA in [0.006]:
                result_img = gibbs_sampling(GRAPH, burn_in_steps, total_samples, BETA, ITA)
                
                path_dir = './grid_28_28'
                if not os.path.exists(path_dir):
                    os.makedirs(path_dir)
                np.save( os.path.join(path_dir, f'{GRAPH}_ROUND_{round}_beta_{BETA}_ita_{ITA}.npy'), result_img )
                save_image(result_img, os.path.join(path_dir, f'{GRAPH}_ROUND_{round}_beta_{BETA}_ita_{ITA}'))




    #facebook graph
    total_samples = 500
    burn_in_steps = 500
    seed = 0

    for rounds in range(5):
        for BETA in [1]:
            for ITA in [1.5]:
                result_img = facebook(seed, burn_in_steps, total_samples, BETA, ITA)
                infec = int(np.sum( (result_img - (-1)) / 2 ))
                path_dir = f'./fb/samples'
                if not os.path.exists(path_dir):
                    os.makedirs(path_dir)
                np.save( os.path.join(path_dir, f'seed_{seed}_ROUND_{rounds}_beta_{BETA}_ita_{ITA}_k_{infec}.npy'), result_img )
                save_image(result_img, os.path.join(path_dir, f'seed_{seed}_ROUND_{rounds}_beta_{BETA}_ita_{ITA}_k{infec}'))



