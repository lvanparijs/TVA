import numpy as np

m = 4 #Number of voting alternatives
n = 5 #Number of voters

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

#Binary matrices representing the voting schemes (Examples given for a 4*5 matrix)
plurality_voting = np.append(np.ones((1,n)),np.zeros((m-1,n))).reshape((m,n))
#[[1 1 1 1 1]
# [0 0 0 0 0]
# [0 0 0 0 0]
# [0 0 0 0 0]]

voting_for_two = np.append(np.ones((2,n)),np.zeros((m-2,n))).reshape((m,n))
#[[1 1 1 1 1]
# [1 1 1 1 1]
# [0 0 0 0 0]
# [0 0 0 0 0]]

anti_plurality_voting = np.append(np.ones((m-1,n)),np.zeros((1,n))).reshape((m,n))
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

#Checks the true voting value
def true_voting(pref_matrix, voting_scheme):
        #Counter for each alternative
        counts = np.zeros(m)
        chars = alphabet[0:m]
        cnts = np.zeros(m)
        if(np.any(borda_check(voting_scheme))): #if the voting scheme is Borda voting
                print("BORDA")
                for i in range(0,m):
                        for j in range(0, n):
                                if(voting_scheme[i][j] >= 1):
                                        counts[pref_matrix[i][j]-1]=counts[pref_matrix[i][j]-1]+voting_scheme[i][j]
                unique = np.arange(0,m,1)
                unique = [chr(int(x) + 65) for x in unique]
                counts = [int(x) for x in counts]
                print(counts)
                print(unique)
        else:
                res_matrix = pref_matrix*voting_scheme
                unique, cnts = np.unique(res_matrix, return_counts=True)
                unique = [chr(int(x)+64) for x in unique]

                for i in range(0, len(unique)):
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

        print("True Voting outcome: ")
        print(outcome)
        print("OVERALL HAPPINESS:")
        print(overall_happiness)

#Ranks a given vector based on the values, ranked from 1->n with 1 being the worst rank
def rank_vector(vector):
        temp = vector.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(vector))
        return ranks+1

#Calculate happiness level based on weights(rank) and distance between preference and outcome
def happiness_level(voter_id, pref_matrix, voting_outcome):
        dist = np.multiply(voting_outcome,voting_outcome-pref_matrix[:,voter_id])
        return 1/(1+abs(sum(dist))) #Keep happiness score within [0..1] interval

#Printing stuff for clarity
print("Input:")
print("Preferences")
#print(np.array([[3,2,3,2,2],[1,4,4,4,3],[4,3,1,3,4],[2,1,2,1,1]])) #Test case given in the helper slides
print(preferences)
print("=================")
print("Voting scheme")
print(borda_voting)
true_voting(preferences,borda_voting)

