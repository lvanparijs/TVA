import numpy as np

m = 4 #Number of voting alternatives
n = 5 #Number of voters

#Binary matrices representing the voting schemes
plurality_voting = np.append(np.ones((1,n)),np.zeros((m-1,n))).reshape((m,n))
voting_for_two = np.append(np.ones((2,n)),np.zeros((m-2,n))).reshape((m,n))
anti_plurality_voting = np.append(np.ones((m-1,n)),np.zeros((1,n))).reshape((m,n))
#Generating the borda voting matrix
borda_voting = np.arange(0,m,1)
borda_check = lambda x: x > 1

preferences = np.arange(0,m,1)#MxN matrix of characters representing the voting alternatives and their ranking
np.random.shuffle(preferences)#Randomly shuffle

#Generate borda & preferences matrix
for i in range(1,n):
        borda_voting = np.append(borda_voting,np.arange(0,m,1)) #np.arange(start,stop,step_size)\
        nxt_pref = np.arange(0,m,1)
        np.random.shuffle(nxt_pref)
        preferences = np.append(preferences, nxt_pref)
borda_voting = (m-1)-borda_voting.reshape((n,m)).T
preferences = preferences.reshape((n,m)).T

#Checks the true voting value
def true_voting(pref_matrix, voting_scheme):
        counts = np.zeros(m)
        if(np.any(borda_check(voting_scheme))): #Borda voting
                for i in range(0,m):
                        for j in range(0, n):
                                if(voting_scheme[i][j] >= 1):
                                        counts[pref_matrix[i][j]-1]=counts[pref_matrix[i][j]-1]+voting_scheme[i][j]
                unique = np.arange(0,m,1)
                unique = [chr(int(x) + 65) for x in unique]
                counts = [int(x) for x in counts]
        else:
                res_matrix = pref_matrix*voting_scheme
                unique, counts = np.unique(res_matrix, return_counts=True)
                unique = [chr(int(x)+64) for x in unique]
        outcome = dict(zip(unique, counts))
        print("True Voting outcome: ")
        print(outcome)

print("Input:")
print("Preferences")
print(np.array([[3,2,3,2,2],[1,4,4,4,3],[4,3,1,3,4],[2,1,2,1,1]]))
print("=================")
print("Voting scheme")
print(borda_voting)
true_voting(np.array([[3,2,3,2,2],[1,4,4,4,3],[4,3,1,3,4],[2,1,2,1,1]]),borda_voting)

