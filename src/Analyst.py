import numpy as np
import itertools


m = 6 #Number of voting alternatives
n = 6 #Number of voters

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

#Binary matrices representing the voting schemes (Examples given for a 4*5 matrix)
plurality_voting = np.append(np.ones((1,n)),np.zeros((m-1,n))).reshape((m,n)).astype(int)
#[[1 1 1 1 1]
# [0 0 0 0 0]
# [0 0 0 0 0]
# [0 0 0 0 0]]

voting_for_two = np.append(np.ones((2,n)),np.zeros((m-2,n))).reshape((m,n)).astype(int)
#[[1 1 1 1 1]
# [1 1 1 1 1]
# [0 0 0 0 0]
# [0 0 0 0 0]]

anti_plurality_voting = np.append(np.ones((m-1,n)),np.zeros((1,n))).reshape((m,n)).astype(int)
#[[1 1 1 1 1]
# [1 1 1 1 1]
# [1 1 1 1 1]
# [0 0 0 0 0]]


#Generating the borda voting matrix
borda_voting = np.arange(0,m,1)
borda_check = lambda x: x > 1

preferences = np.arange(1,m+1,1)#MxN matrix of characters representing the voting alternatives and their ranking
np.random.shuffle(preferences)#Randomly shuffle

#Generate borda & preferences matrix
for i in range(1,n):
        borda_voting = np.append(borda_voting,np.arange(0,m,1)) #np.arange(start,stop,step_size)\
        nxt_pref = np.arange(1,m+1,1)
        np.random.shuffle(nxt_pref)
        preferences = np.append(preferences, nxt_pref)

#Transformation done to correct dimensions
borda_voting = (m-1)-borda_voting.reshape((n,m)).T
#[[3 3 3 3 3]
# [2 2 2 2 2]
# [1 1 1 1 1]
# [0 0 0 0 0]]

#Transformation done to correct dimensions
preferences = preferences.reshape((n,m)).T

def voting_analysis(pref_matrix, voting_scheme):
        #Outputs 1 and 2 completed
        outcome, happy = true_voting(pref_matrix, voting_scheme)
        print("True Voting outcome: ")
        print(outcome)
        print("OVERALL HAPPINESS:")
        print(happy)
        #For each voter a strategic voter preference of form (modified_pref_list, Outcome of modification, Overall happiness, advantage for voter i)
        strategic_voting(pref_matrix, voting_scheme, outcome)

#Checks the true voting value
def true_voting(pref_matrix, voting_scheme):
        #Counter for each alternative
        counts = np.zeros(m)
        chars = alphabet[0:m]

        if(np.any(borda_check(voting_scheme))): #if the voting scheme is Borda voting
                for i in range(0,m):
                        for j in range(0, n):
                                if(voting_scheme[i][j] >= 1):
                                        counts[pref_matrix[i][j]-1]=counts[pref_matrix[i][j]-1]+voting_scheme[i][j]
                unique = np.arange(0,m,1)
                unique = [chr(int(x) + 65) for x in unique]
                counts = [int(x) for x in counts]
        else:
                res_matrix = pref_matrix*voting_scheme

                unique, cnts = np.unique(res_matrix, return_counts=True)
                unique = [chr(int(x)+64) for x in unique]

                for i in range(1, len(unique)):
                        counts[chars.index(unique[i])] = cnts[i]

        #Zipping it and putting it in a dictionary for nice viewing purposes :)
        outcome = dict(zip(chars, counts))
        #Initialise outcome vector
        outcome_vector = np.array([])

        #Collects all the outcome values
        for key in outcome:
                outcome_vector = np.append(outcome_vector,int(outcome[key]))

        #Rankes outcome vector based on the scores
        #Double flipping is done to make sure tie breaking is done correctly by preferring A over B
        ranked_outcome = np.flip(rank_vector(np.flip(outcome_vector)))

        #Calculate total happiness
        overall_happiness = 0
        for i in range(0,n-1):
                overall_happiness = overall_happiness + happiness_level(i,pref_matrix,ranked_outcome)

        return ranked_outcome, overall_happiness

def strategic_voting(pref_matrix, voting_scheme, ranked_outcome):
        out = []
        #For each voter
        for i in range(0,n):
                #Bullet
                happy_orig = happiness_level(i, pref_matrix, ranked_outcome)

                print("==========================")
                print("VOTER " + str(i))
                print("True Happiness: " + str(happy_orig))
                print("True voting: " + str(pref_matrix[:, i]))
                print("True Outcome: "+str(ranked_outcome))

                pref_bull = pref_matrix[:,i].copy()
                compromising_vote(pref_bull)
                pref_bull = bullet_vote(pref_bull)

                bull_pref_matrix = pref_matrix.copy()
                bull_pref_matrix[:,i] = pref_bull
                outcome_bull, _ = true_voting(bull_pref_matrix,voting_scheme)
                happy_bull = happiness_level(i, bull_pref_matrix, outcome_bull)
                print("==========================")
                print("BULLET")
                print("Bullet Happiness: "+str(happy_bull))
                print("Bullet Voting: "+str(pref_bull))
                print("Bullet Outcome: "+str(outcome_bull))
                print("==========================")
                print("COMPROMISING")
                poss = compromising_vote(pref_matrix[:,i].copy())
                happy_com = []
                outcome_com = []

                for j in range(len(poss[0])):
                        com_pref_matrix = pref_matrix.copy()
                        com_pref_matrix[:, i] = poss[:,j]
                        outcome_com, _ = true_voting(com_pref_matrix, voting_scheme)
                        happy_com = happy_com + [happiness_level(i, com_pref_matrix, outcome_com)]

                best_i = happy_com.index(max(happy_com))
                print("Comp Happiness: "+str(max(happy_com)))
                print(pref_matrix[:,i].copy())
                print("Comp Voting: "+str(poss[:,best_i]))
                print("Comp Outcome: "+str(outcome_com))
                print("==========================")

                tuple = ()
                pref = []
                output = []
                happiness = 0
                #If bullet is better than original
                if(happy_orig<happy_bull):
                        #Compromising better than bullet?
                        if(happy_bull < max(happy_com)):
                                pref = poss[:,best_i]
                                output = outcome_com
                                for j in range(0,n):
                                        happiness += happiness_level(j,com_pref_matrix,outcome_com)
                                #tuple = (poss[:,happy_com.index(max(happy_com))],outcome_com,happiness_level(i,com_pref_matrix,outcome_com))
                        else:
                                pref = pref_bull
                                output = outcome_bull
                                for j in range(0, n):
                                        happiness += happiness_level(j,bull_pref_matrix,outcome_bull)
                else:
                        pref = pref_matrix[:,i].copy()
                        output = ranked_outcome
                        for j in range(0, n):
                                happiness += happiness_level(j,pref_matrix,ranked_outcome)
                tuple = (str(pref), str(output), happiness, happiness_level(i,pref_matrix,output)-happy_orig)
                print("Output: (preferences, output, happiness, happiness-gain)")
                print("preferences, output: highest number preferred")
                print(tuple)
                print("RESULT BEFORE: " + str(ranked_outcome))



#Ranks a given vector based on the values, ranked from 1->n with 1 being the worst rank
def rank_vector(vector):
        temp = vector.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(vector))
        return ranks+1

#Calculate happiness level based on weights(rank) and distance between preference and outcome
def happiness_level(voter_id, pref_matrix, voting_outcome):
        dist = np.multiply(pref_matrix[:,voter_id],voting_outcome-pref_matrix[:,voter_id])
        return 1/(1+abs(sum(dist))) #Keep happiness score within [0..1] interval

def bullet_vote(pref_vector):
        res  = pref_vector
        res[res < max(res)] = 0
        return res

def get_alternatives(pref_vector):
        return list(itertools.permutations(pref_vector))

def compromising_vote(pref_vector):
        print(max(pref_vector))
        best = np.argmax(pref_vector)
        perm = get_alternatives(pref_vector[:])

        leng = len(perm)
        res = []
        for i in range(leng):
                if(perm[i][best] == pref_vector[best]):
                        res = res + [i]
        perm = [perm[i] for i in res]
        perm = np.array(perm)

        perm = np.reshape(perm, (len(res),m)).T
        return perm


#Printing stuff for clarity
print("Input:")
print("Preferences")
#print(np.array([[3,2,3,2,2],[1,4,4,4,3],[4,3,1,3,4],[2,1,2,1,1]])) #Test case given in the helper slides
print(preferences)
print("=================")
print("Voting scheme")
print(voting_for_two)
#print(happiness_level(i, new_pref_matrix, outcome))
voting_analysis(preferences,voting_for_two)

