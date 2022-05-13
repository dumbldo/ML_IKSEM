import math
import torch
from torch.nn import functional as F
import numpy as np

# Ce fichier contient les différentes fonction de génération



def generate_beam(sentence,encoder, decoder, string2code, id2lettre, eos, k, maxlen=200, nucleus = None, spaccing = " "):
    """
        Génere une séquence en beam-search : à chaque itération, on explore pour chaque candidat les k symboles les plus probables; puis seuls les k meilleurs candidats de l'ensemble des séquences générées sont conservés (au sens de la vraisemblance) pour l'itération suivante.
        * sentence  : phrase à traduire sous forme de liste de tags
        * encoder : l'encodeur
        * decoder : le decodeur
        * string2code : transform a string to a code for traduced vocabulary
        * id2lettre : transform a id to a letter for traduced vocabulary
        * eos : ID du token end of sequence
        * k : le paramètre du beam-search
        * spaccing : chaine entre chaque token
        * maxlen : longueur maximale
    """
    def add_best(new_p, new_best, old_h ,k_best_new, k_best_p_new, previous_h):
        '''
            choose if add or not the new sentence new_best of probabily new_p to the k best list 
        '''
        if len(k_best_new) < k:
            k_best_new += [new_best]
            k_best_p_new += [new_p]
            previous_h += [old_h]
            return k_best_new, k_best_p_new, previous_h, len(k_best_new) - 1
        worst = np.argmin(k_best_new)
        if k_best_p_new[worst] < new_p:
            k_best_new[worst] = new_best
            k_best_p_new[worst] = new_p
            previous_h[worst] = old_h
            return k_best_new, k_best_p_new, previous_h, worst
        return k_best_new, k_best_p_new, previous_h, -1

    
    
    k_best = [[""],[]]
    k_best_p = [[1],[]]
    
    batch_size = sentence.batch_sizes
    sentence = sentence.unsqueeze(1)
    sentence.batch_sizes = batch_size

    # encode sentence
    h, _ = encoder.forward((sentence,None))
    # decode SOS
    h = decoder.generate(h,len_seq = 1,return_hidden = True)[1] # add SOS
    k_best_h = [h]

    previous_h = []

    for _ in range(maxlen):
        # print(k_best[0])
        for j in range(len(k_best[0])): # for each element
            if k_best[0][j].endswith(id2lettre(eos)+spaccing): # if end => stop 
                k_best[1],k_best_p[1],previous_h,idx = add_best(
                    k_best_p[0][j], # new proba
                    k_best[0][j] , # new sentence
                    k_best_h[j], # h used previously
                    k_best[1], # where to stock new best sentence
                    k_best_p[1], # where to compare and stock new proba
                    previous_h) # where to stock the h used previously
                continue
            
            # probability of next elements from h (classifieur => softmax)
            
            if nucleus is None:
                p_x = torch.nn.functional.softmax(decoder.classifieur(k_best_h[j])[0][0],dim=0)
            else:
                p_x = nucleus(k_best_h[j])

            # get best elements first
            bests = torch.argsort(p_x, descending=True) # if determinist else int(torch.multinomial(p_x, k)[0])
            idx = 0
            t = 0
            
            while idx != -1: 
                # add them for next steps
                k_best[1],k_best_p[1],previous_h,idx = add_best(
                    k_best_p[0][j] * p_x[bests[t]], # new proba
                    k_best[0][j] + id2lettre(int(bests[t])) + spaccing , # new sentence
                    k_best_h[j], # h used previously
                    k_best[1], # where to stock new best sentence
                    k_best_p[1], # where to compare and stock new proba
                    previous_h) # where to stock the h used previously
                t += 1
        # update h for next step
        k_best_h = [] 
        # print(k_best[1])
        for j in range(len(k_best[1])):
            
            last_word = string2code(k_best[1][j].split(spaccing)[-2])
            k_best_h += [decoder.generate(previous_h[j],len_seq = 1,return_hidden = True, start = last_word)[1]]

        k_best[0] = k_best[1]
        k_best[1] = []

        k_best_p[0] = k_best_p[1]
        k_best_p[1] = []

        previous_h = []
        
    return k_best[0]



# p_nucleus
def p_nucleus(decoder, alpha: float):
    """Renvoie une fonction qui calcule la distribution de probabilité sur les sorties

    Args:
        * decoder: renvoie les logits étant donné l'état du RNN
        * alpha (float): masse de probabilité à couvrir
    """
    def compute(h):
        """Calcule la distribution de probabilité sur les sorties

        Args:
           * h (torch.Tensor): L'état à décoder
        """
        # print(decoder.classifieur(h)[0][0])
        p_x = torch.nn.functional.softmax(decoder.classifieur(h)[0][0],dim = 0)
        # print(p_x)
        l = torch.argsort(p_x,descending=True)
        tot = 0
        res = torch.zeros(p_x.shape)
        for i in l:
            tot += p_x[i]
            res[i] = p_x[i]
            if tot >= alpha:
                break
        return res/tot
    return compute
