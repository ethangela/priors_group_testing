import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import timeit
import sys
import random


def save_image(image, title):
    plt.imshow(image, cmap='gray')
    plt.savefig(title + '.png')
    plt.close()



def test_matrix(prob, num_item, num_test, ground_truth, test_matirx_path, flip=0):
    text_dir = 'test_matrix'
    if not os.path.isdir(text_dir):
        os.mkdir(text_dir)
    test_matirx_path = os.path.join(text_dir, test_matirx_path)
    if not os.path.exists(test_matirx_path):
        for test in range(num_test):
            array = list(np.random.binomial(n=1, p=prob, size=num_item))
            if sum(array[idx] for idx in np.where(ground_truth == 1)[0]) > 0:
                single_result_real = 1
                if flip == 0:
                    single_result = 1
                else:
                    if np.random.rand() > flip: #10
                        single_result = 1 
                    else: #11
                        single_result = 0 
            else:
                single_result_real = 0
                if flip == 0:
                    single_result = 0 
                else:
                    if np.random.rand() > flip: #00
                        single_result = 0
                    else: #01
                        single_result = 1
            if test == 0:
                matrix = [array]
                result = [single_result]
                result_real = [single_result_real]
            else:
                matrix.append(array)
                result.append(single_result)
                result_real.append(single_result_real)
        np.save(test_matirx_path, np.array(matrix))
        save_image(np.array(matrix), test_matirx_path[:-4])
        test_matirx = np.array(matrix)
        # print('SHAPE CHCECK', test_matirx.shape)

    
    else:
        test_matirx = np.load(test_matirx_path)
        for test in range(num_test):
            array = list(test_matirx[test])
            if sum(array[idx] for idx in np.where(ground_truth == 1)[0]) > 0:
                single_result_real = 1
                if flip == 0:
                    single_result = 1
                else:
                    if np.random.rand() > flip: #10
                        single_result = 1 
                    else: #11
                        single_result = 0 
            else:
                single_result_real = 0
                if flip == 0:
                    single_result = 0 
                else:
                    if np.random.rand() > flip: #00
                        single_result = 0
                    else: #01
                        single_result = 1
            if test == 0:
                result = [single_result]
                result_real = [single_result_real]
            else:
                result.append(single_result)
                result_real.append(single_result_real)
        # print('SHAPE CHCECK', test_matirx.shape)
    
    return test_matirx, np.array(result), np.array(result_real)



def programming(gibbs_sample, mode='grid', skelton_h=28, skelton_w=28, optimizer='qp', num_tests=1000, beta=0.5, eta=0.006, pickle_file_path='oct10_edge_new.pkl', binary=0, noise_flip=0, num_exps=1, poly=1, octep=None, octprob=None):
    



    # Test matrix
    sample = np.load(f'{mode}/{gibbs_sample}.npy') 
    gt = sample.reshape(-1)
    # print('SHAPE CHCECK', gt.shape)
    gt_norm = (gt-gt.min()) / (gt.max()-gt.min())
    X, ys, ys_real = test_matrix(prob=np.log(2)/int(sum(gt_norm)), num_item=len(gt_norm), num_test=num_tests, ground_truth=gt_norm, test_matirx_path=f'{skelton_h}_test_matirx_{num_tests}_{num_exps}.npy', flip=noise_flip)
    # print('SHAPE CHCECK', X.shape)
    # sys.exit("Debugging")
    is0 = np.where(ys == 0)[0] #(n0,)
    is1 = np.where(ys == 1)[0] #(n1,)
    n0 = len(is0)
    n1 = len(is1)
    X0 = X[is0,:] #(n0,p)
    X1 = X[is1,:] #(n1,p)
    k = int(sum(gt_norm))
    ys_real_0 = ys_real[is0]
    ys_real_1 = ys_real[is1]



    # Create a new model
    start0 = timeit.default_timer()
    import gurobipy as gp
    from gurobipy import GRB
    if optimizer == 'qp':
        model = gp.Model("qp")
        model.params.NonConvex = 2 
    elif optimizer == 'lp':
        model = gp.Model("lp")
    elif optimizer == 'poly':
        model = gp.Model("poly")



    # graph 
    if mode == 'grid_28_28':
        adjacency = np.zeros((skelton_h*skelton_w, skelton_h*skelton_w))
        for i in range(skelton_h*skelton_w):
            for j in range(skelton_h*skelton_w):
                if (j-i==1 and j%skelton_w!=0) or j-i==skelton_h:
                    adjacency[i][j] = 1
        adjacency2 = adjacency + adjacency.T
    elif mode == 'grid_28_28_block_7':
        block = 7
        adjacency = np.zeros((skelton_h*skelton_w, skelton_h*skelton_w))
        for i in range(skelton_h*skelton_w):
            for j in range(skelton_h*skelton_w):
                if (j-i==1 and j%skelton_w!=0 and j%block!=0) or (j-i==skelton_w and j%(skelton_w*block)>=skelton_w):
                    adjacency[i][j] = 1
        adjacency2 = adjacency + adjacency.T
    
    if octep == 'loss':
        #loss edge test
        adjacency_f = adjacency.reshape(-1)
        kt = 0
        ct = 0
        for idx in range(skelton_h*skelton_w*skelton_h*skelton_w):
            if adjacency_f[idx] == 1:
                kt += 1
                if random.randint(1, 10000) <= int(10001*octprob):
                    ct += 1
                    adjacency_f[idx] = 0
            else:
                continue
        adjacency = adjacency_f.reshape(skelton_h*skelton_w, skelton_h*skelton_w)
        loss_rate = ct/kt        
    if octep == 'loss_and_add':
        #loss edge and add edge test
        num_edges = int(adjacency.sum())
        M = int(num_edges * octprob)
        if M != 0:
            loss_idx = np.random.choice(num_edges, M, replace=False)
            add_idx = np.random.choice(int((skelton_h*skelton_w*skelton_h*skelton_w-skelton_h*skelton_w)/2-num_edges), M, replace=False)
            idx1 = -1
            idx0 = -1
            for i in range(skelton_h*skelton_w):
                for j in range(skelton_h*skelton_w):
                    if j > i:
                        if adjacency[i][j] == 1:
                            idx1 += 1
                            if idx1 in loss_idx:
                                adjacency[i][j] = 0
                        elif adjacency[i][j] == 0:
                            idx0 += 1
                            if idx0 in add_idx:
                                adjacency[i][j] = 1
            assert idx1+1 == num_edges
            assert idx0+1 == int((skelton_h*skelton_w*skelton_h*skelton_w-skelton_h*skelton_w)/2-num_edges)
        loss_rate = octprob

            



    # Create variables
    var_dic = dict()
    for i in range(skelton_h):
        for j in range(skelton_w):
            if binary:
                var_dic[f"x_{i}_{j}"] = model.addVar(vtype=GRB.BINARY, name=f"u_{i}_{j}")
            else:
                var_dic[f"x_{i}_{j}"] = model.addVar(lb=0, ub=1, name=f"x_{i}_{j}")
    if optimizer == 'poly':
        poly_dic = dict()
        # Degree = 0
        degree = list()
        for i in range(skelton_h*skelton_w):
            for j in range(skelton_h*skelton_w):
                if adjacency[i][j] == 1:
                    irow, icol = i//skelton_h, i%skelton_h
                    jrow, jcol = j//skelton_w, j%skelton_w
                    #joint variable
                    if binary:
                        poly_dic[f"xx_{irow}_{icol}_{jrow}_{jcol}"] = model.addVar(vtype=GRB.BINARY, name=f"xx_{irow}_{icol}_{jrow}_{jcol}")
                    else:
                        poly_dic[f"xx_{irow}_{icol}_{jrow}_{jcol}"] = model.addVar(lb=0, ub=1, name=f"xx_{irow}_{icol}_{jrow}_{jcol}")
                    #degree
                    # Degree += var_dic[f"x_{irow}_{icol}"] + var_dic[f"x_{jrow}_{jcol}"]
                    degree.append(var_dic[f"x_{irow}_{icol}"])
                    degree.append(var_dic[f"x_{jrow}_{jcol}"])
    if noise_flip != 0:
        flip_dict = dict()
        for t in range(num_tests):
            if binary:
                flip_dict[f"y_{t}"] = model.addVar(vtype=GRB.BINARY, name=f"y_{t}")
            else:
                flip_dict[f"y_{t}"] = model.addVar(lb=0, ub=1, name=f"y_{t}") 



    # Set objective: 
    var_array = np.array(list(var_dic.values()))
    len_var = len(var_array)
    var_array_variant = 2 * var_array - 1
    if optimizer == 'poly':
        poly_array = np.array(list(poly_dic.values()))
        len_poly = len(poly_array)
        degree_array = np.array(degree)
        len_degree = len(degree_array)
    if noise_flip == 0:
        if optimizer == 'qp':
            obj = -1 * ( beta * var_array_variant.dot(adjacency).dot(var_array_variant.T) - eta * var_array_variant.dot(np.ones(len_var).T) )
        elif optimizer == 'lp':
            obj = var_array_variant.dot(np.ones(len_var).T)
        elif optimizer == 'poly':
            obj = -1 * ( 4*beta * poly_array.dot(np.ones(len_poly).T) - 2*beta * degree_array.dot(np.ones(len_degree).T)  - eta * var_array_variant.dot(np.ones(len_var).T) ) #
    else:
        flip_array = np.array(list(flip_dict.values()))
        len_flip = len(flip_array)
        prob = np.log(2)/int(sum(gt_norm))
        if optimizer == 'qp':
            obj = -1 * ( beta * var_array_variant.dot(adjacency).dot(var_array_variant.T) - eta * var_array_variant.dot(np.ones(len_var).T) ) + np.log((1-noise_flip)/noise_flip) * flip_array.dot(np.ones(len_flip).T)
        elif optimizer == 'lp':
            obj = var_array_variant.dot(np.ones(len_var).T) + np.log((1-noise_flip)/noise_flip)/np.log((1-prob)/prob) * flip_array.dot(np.ones(len_flip).T)
        elif optimizer == 'poly':
            obj = -1 * ( 4*beta * poly_array.dot(np.ones(len_poly).T) - 2*beta * degree_array.dot(np.ones(len_degree).T)  - eta * var_array_variant.dot(np.ones(len_var).T) ) + np.log((1-noise_flip)/noise_flip) * flip_array.dot(np.ones(len_flip).T)
    
    model.setObjective(obj, GRB.MINIMIZE)    

    

    # Add constraint:    
    if noise_flip == 0:
        X0_array = X0.dot(var_array)
        X1_array = X1.dot(var_array)
        for i in range(n0):
            model.addConstr(X0_array[i] == 0, f"c_non_{i}")
        for i in range(n1):
            model.addConstr(X1_array[i] >= 1, f"c_def_{i}")
    else:
        X_array = X.dot(var_array)
        for i in range(n0+n1):
            if i in is0:
                model.addConstr(X_array[i] == flip_array[i], f"c_non_{i}")
            elif i in is1:
                model.addConstr(X_array[i] + flip_array[i] >= 1, f"c_def_{i}")        
    if optimizer == 'poly': #default 1
        if poly == 1:
            for i in range(skelton_h*skelton_w):
                for j in range(skelton_h*skelton_w):
                    if adjacency[i][j] == 1:
                        irow, icol = i//skelton_h, i%skelton_h
                        jrow, jcol = j//skelton_w, j%skelton_w                       
                        model.addConstr(var_dic[f"x_{irow}_{icol}"] >= poly_dic[f"xx_{irow}_{icol}_{jrow}_{jcol}"], f"u1_{irow}_{icol}_{jrow}_{jcol}")
                        model.addConstr(var_dic[f"x_{jrow}_{jcol}"] >= poly_dic[f"xx_{irow}_{icol}_{jrow}_{jcol}"], f"u2_{irow}_{icol}_{jrow}_{jcol}")
                        model.addConstr(var_dic[f"x_{irow}_{icol}"] + var_dic[f"x_{jrow}_{jcol}"] - poly_dic[f"xx_{irow}_{icol}_{jrow}_{jcol}"]  <= 1, f"uu_{irow}_{icol}_{jrow}_{jcol}")            
        elif poly == 2:
            for i in range(skelton_h*skelton_w):
                poly_sum = 0
                poly_count = 0
                for j in range(skelton_h*skelton_w):
                    if adjacency[i][j] == 1:      
                        model.addConstr(var_dic[f"x_{irow}_{icol}"] + var_dic[f"x_{jrow}_{jcol}"] - poly_dic[f"xx_{irow}_{icol}_{jrow}_{jcol}"]  <= 1, f"uu_{irow}_{icol}_{jrow}_{jcol}")
                    if adjacency2[i][j] == 1: 
                        poly_count += 1
                        if (irow+icol) < (jrow+jcol) :
                            poly_sum += poly_dic[f"xx_{irow}_{icol}_{jrow}_{jcol}"]
                        else:
                            poly_sum += poly_dic[f"xx_{jrow}_{jcol}_{irow}_{icol}"]
                model.addConstr(poly_count * var_dic[f"x_{irow}_{icol}"] >= poly_sum, f"u_{irow}_{icol}")
        elif poly == 3:
            for i in range(skelton_h*skelton_w):
                for j in range(skelton_h*skelton_w):
                    if adjacency[i][j] == 1:
                        irow, icol = i//skelton_h, i%skelton_h
                        jrow, jcol = j//skelton_w, j%skelton_w                       
                        model.addConstr(var_dic[f"x_{irow}_{icol}"] + var_dic[f"x_{jrow}_{jcol}"] - poly_dic[f"xx_{irow}_{icol}_{jrow}_{jcol}"]  <= 1, f"uu_{irow}_{icol}_{jrow}_{jcol}")
                        model.addConstr(- var_dic[f"x_{irow}_{icol}"] - var_dic[f"x_{jrow}_{jcol}"] + 2 * poly_dic[f"xx_{irow}_{icol}_{jrow}_{jcol}"]  <= 0, f"uu2_{irow}_{icol}_{jrow}_{jcol}")



    # optimize model
    start1 = timeit.default_timer()
    model.optimize()
    stop = timeit.default_timer()
    comp_time = stop - start1
    total_time = stop - start0



    # result
    nfp,nfn = 0,0
    qfp,qfn = 0,0
    mfp,mfn = 0,0
    for round in ['normal', 'quite', 'most']:
        for i,v in enumerate(model.getVars()):
            if i < skelton_h*skelton_w:
                if (v.x != 0) and (v.x != 1):
                    print(f'MODE {mode}; DECODE {optimizer}; (ATL_{poly}); NOISE {noise_flip}; TEST {num_tests}; TRIAL {num_exps}; REAL_OUTPUT {v.x}')
                if round == 'normal':
                    v_x = int(v.x+0.5)
                elif round == 'quite':
                    if v.x < 0.9:
                        v_x = 0
                    else:
                        v_x = 1
                elif round == 'most':
                    if v.x < 0.99:
                        v_x = 0
                    else:
                        v_x = 1
                if gt_norm[i] == 0 and v_x == 1:
                    if round == 'normal':
                        nfp += 1
                    elif round == 'quite':
                        qfp += 1
                    elif round == 'most':
                        mfp += 1
                if gt_norm[i] == 1 and v_x == 0:
                    if round == 'normal':
                        nfn += 1
                    elif round == 'quite':
                        qfn += 1
                    elif round == 'most':
                        mfn += 1
            else:
                break 

        # opt_arr = np.array(opt_arr)
        # opt_arr_variant = 2 * opt_arr - 1
        # qp = -1 * (beta * 2 * opt_arr_variant.dot(adjacency).dot(opt_arr_variant.T) - eta * opt_arr_variant.sum())

        # consistent, inconsistent = 0, 0
        # for test in range(num_tests):
        #     array = X[test]      
        #     if array.dot(gt_norm.T) == 0:
        #         if array.dot(opt_arr.T) == 0:
        #             consistent += 1
        #         else:
        #             inconsistent += 1
        #     elif array.dot(gt_norm.T) > 0:
        #         if array.dot(opt_arr.T) > 0:
        #             consistent += 1
        #         else:
        #             inconsistent += 1

    # opt_arr_variant = opt_arr_variant.reshape((28,28))

 

    fpl = [nfp/k, qfp/k, mfp/k]
    fnl = [nfn/k, qfn/k, mfn/k]
    if not os.path.exists(pickle_file_path):
        d = {'mode':[mode], 'sample':[gibbs_sample], 'optimizer':[optimizer], 'flip':[noise_flip], 'num_tests':[num_tests], 'num_exps':[num_exps], 'objective_value':[obj.getValue()], 
            'normal_fp':[fpl[0]], 'normal_fn':[fnl[0]], 'quite_fp':[fpl[1]], 'quite_fm':[fnl[1]], 'most_fp':[fpl[2]], 'most_fn':[fnl[2]], 'binary':[binary], 'defective':[k], 'time':[comp_time], 'total_time':[total_time], 'poly':[poly],
            'oct_exp':[octep], 'oct_prob':[octprob], 'oct_real_rate':[loss_rate]}
        df = pd.DataFrame(data=d)
        df.to_pickle(pickle_file_path)
    else:
        d = {'mode':mode, 'sample':gibbs_sample, 'optimizer':optimizer, 'flip':noise_flip, 'num_tests':num_tests, 'num_exps':num_exps, 'objective_value':obj.getValue(), 
            'normal_fp':fpl[0], 'normal_fn':fnl[0], 'quite_fp':fpl[1], 'quite_fm':fnl[1], 'most_fp':fpl[2], 'most_fn':fnl[2], 'binary':binary, 'defective':k, 'time':comp_time, 'total_time':total_time, 'poly':poly,
            'oct_exp':octep, 'oct_prob':octprob, 'oct_real_rate':loss_rate}
        df = pd.read_pickle(pickle_file_path)
        df = df.append(d, ignore_index=True)
        df.to_pickle(pickle_file_path)




       

if __name__ == '__main__':

    # ### oct exp 2 ###
    # SKH=28
    # SKW=28
    # BETA=0.52
    # # ETA=0.009
    # # MODE = 'grid_72_72_block_18'
    # # SAMPLE = f'grid_72_72_block_18_grid_ROUND_1_beta_{BETA}_ita_{ETA}'

    # for lptp in ['qp', 'lp', 'poly']:
    #     for flip in [0, 0.01]: 
    #         for bi in [0,1]:
    #             for p in [0.1, 0.3, 0.5, 0.7]:
    #                 for test in range(100,900,100):
    #                     for exp in range(50,100):
    #                         if lptp == 'poly':
    #                             programming(gibbs_sample='grid_28_28_ROUND_31_beta_0.5_ita_0.006', mode='grid_28_28', 
    #                                 skelton_h=SKH, skelton_w=SKH, optimizer=lptp, num_tests=test, 
    #                                 beta=0.5, eta=0.006, binary=bi, noise_flip=flip, num_exps=exp, poly=1, octep='loss', octprob=p)
    #                             programming(gibbs_sample='grid_28_28_block_7_grid_ROUND_11_beta_0.6_ita_0.035', mode='grid_28_28_block_7', 
    #                                 skelton_h=SKH, skelton_w=SKH, optimizer=lptp, num_tests=test, 
    #                                 beta=0.6, eta=0.035, binary=bi, noise_flip=flip, num_exps=exp, poly=1, octep='loss', octprob=p)
    #                         else:
    #                             programming(gibbs_sample='grid_28_28_ROUND_31_beta_0.5_ita_0.006', mode='grid_28_28', 
    #                                 skelton_h=SKH, skelton_w=SKH, optimizer=lptp, num_tests=test, 
    #                                 beta=0.5, eta=0.006, binary=bi, noise_flip=flip, num_exps=exp, poly=0, octep='loss', octprob=p)
    #                             programming(gibbs_sample='grid_28_28_block_7_grid_ROUND_11_beta_0.6_ita_0.035', mode='grid_28_28_block_7', 
    #                                 skelton_h=SKH, skelton_w=SKH, optimizer=lptp, num_tests=test, 
    #                                 beta=0.6, eta=0.035, binary=bi, noise_flip=flip, num_exps=exp, poly=0, octep='loss', octprob=p)

    # df = pd.read_pickle('oct10_edge_new.pkl') #'to_produce_table_chart_aug19.pkl'
    # csv_name = 'chart_'+'oct10_edge_new.pkl'[:-4]+'.csv'
    # df.to_csv(csv_name, index=False)





    ### oct exp 3 ###
    SKH=28
    SKW=28
    # BETA=0.52
    # ETA=0.009
    # MODE = 'grid_72_72_block_18'
    # SAMPLE = f'grid_72_72_block_18_grid_ROUND_1_beta_{BETA}_ita_{ETA}'

    for lptp in ['lp', 'poly']:#['qp', 'lp', 'poly']:
        for flip in [0, 0.01]: 
            for bi in [0,1]:
                for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0]:
                    for test in [300, 500]:#range(100,900,100):
                        for exp in range(50,100):
                            if lptp == 'poly':
                                # programming(gibbs_sample='grid_28_28_ROUND_31_beta_0.5_ita_0.006', mode='grid_28_28', 
                                #     skelton_h=SKH, skelton_w=SKW, optimizer=lptp, num_tests=test, 
                                #     beta=0.5, eta=0.006, binary=bi, noise_flip=flip, num_exps=exp, poly=1, octep='loss_and_add', octprob=p)
                                programming(gibbs_sample='grid_28_28_block_7_grid_ROUND_11_beta_0.6_ita_0.035', mode='grid_28_28_block_7', 
                                    skelton_h=SKH, skelton_w=SKW, optimizer=lptp, num_tests=test, 
                                    beta=0.6, eta=0.035, binary=bi, noise_flip=flip, num_exps=exp, poly=1, octep='loss_and_add', octprob=p)
                            else:
                                # programming(gibbs_sample='grid_28_28_ROUND_31_beta_0.5_ita_0.006', mode='grid_28_28', 
                                #     skelton_h=SKH, skelton_w=SKW, optimizer=lptp, num_tests=test, 
                                #     beta=0.5, eta=0.006, binary=bi, noise_flip=flip, num_exps=exp, poly=0, octep='loss_and_add', octprob=p)
                                programming(gibbs_sample='grid_28_28_block_7_grid_ROUND_11_beta_0.6_ita_0.035', mode='grid_28_28_block_7', 
                                    skelton_h=SKH, skelton_w=SKW, optimizer=lptp, num_tests=test, 
                                    beta=0.6, eta=0.035, binary=bi, noise_flip=flip, num_exps=exp, poly=0, octep='loss_and_add', octprob=p)

    df = pd.read_pickle('oct10_edge_new.pkl') #'to_produce_table_chart_aug19.pkl'
    csv_name = 'chart_'+'oct10_edge_new.pkl'[:-4]+'.csv'
    df.to_csv(csv_name, index=False)




















    # # COMP DD debug
    # for mode in ['grid_28_28']:
    # # for mode in ['grid_28_28_block_7']:
    #     for gibbs_sample in ['graph_ROUND_4_beta_0.5_ita_0.006', 'graph_ROUND_1_beta_0.5_ita_0.006', 'graph_ROUND_3_beta_0.5_ita_0.007']:
    #     # for gibbs_sample in ['block_ROUND_0_beta_0.55_ita_0.02', 'block_ROUND_0_beta_0.6_ita_0.02', 'block_ROUND_1_beta_0.55_ita_0.01']:
    #         for optimizer in ['qp', 'lp']:
    #             for num_tests in [250, 500, 750, 1000, 1250, 1500, 1750, 2000]:
                    
    #                 sample = np.load(f'{mode}/{gibbs_sample}.npy') 
    #                 gt = sample.reshape(-1)
    #                 gt = (gt-gt.min()) / (gt.max()-gt.min())
    #                 X, ys = test_matrix(prob=np.log(2)/int(sum(gt)), num_item=len(gt), num_test=num_tests, ground_truth=gt, test_matirx_path=f'test_matirx_{num_tests}.npy')
    #                 is0 = np.where(ys == 0)[0] #(n0,)
    #                 is1 = np.where(ys == 1)[0] #(n1,)
    #                 n0 = len(is0)
    #                 n1 = len(is1)
    #                 X0 = X[is0,:] #(n0,p)
    #                 X1 = X[is1,:] #(n1,p)
                    
    #                 PD = np.ones(784)
    #                 for array in X0:
    #                     for idx,item in enumerate(array):
    #                         if item == 1:
    #                             PD[idx] = 0
    #                 np.save(f'{mode}/{gibbs_sample}_COMP_test_{num_tests}.npy', PD)

    #                 DD = np.zeros(784)
    #                 for array in X1:
    #                     sum_ = 0
    #                     idx_ = -1
    #                     for idx,item in enumerate(array):
    #                         if item == 1 and PD[idx] == 1:
    #                             sum_ += 1
    #                             idx_ = idx
    #                     if sum_ == 1:
    #                         DD[idx_] = 1
    #                 np.save(f'{mode}/{gibbs_sample}_DD_test_{num_tests}.npy', DD)

            
    




